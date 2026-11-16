"""Attribut-Klassifikator: MobileNetV3-small mit 4 Köpfen à 3 Klassen.

Eingabe: entzerrte Karte im Hochformat, 128 x 192 (B x H), RGB, ImageNet-normiert.
Ausgabe: dict Attribut -> Logits (N, 3), dazu "card" -> Logits (N, 2): ist das überhaupt eine Karte?
"""
from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from .common import COLORS, FILLS, SHAPES

ATTRS = {"color": COLORS, "shape": SHAPES, "fill": FILLS, "count": ["1", "2", "3"]}
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
IN_W, IN_H = 128, 192


class CardNet(nn.Module):
    def __init__(self, pretrained: bool = True, dropout: float = 0.2):
        super().__init__()
        net = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None)
        self.features = net.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        c = net.classifier[0].in_features  # 576
        self.neck = nn.Sequential(nn.Flatten(), nn.Linear(c, 256), nn.Hardswish(), nn.Dropout(dropout))
        self.heads = nn.ModuleDict({k: nn.Linear(256, len(v)) for k, v in ATTRS.items()})
        self.heads["card"] = nn.Linear(256, 2)  # 0 = keine Karte (Fehlalarm des Detektors), 1 = Karte

    def forward(self, x):
        f = self.neck(self.pool(self.features(x)))
        return {k: h(f) for k, h in self.heads.items()}


def read_label_csv(csv_path: Path) -> list[tuple[str, list[int]]]:
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            rows.append((r["file"], [ATTRS[k].index(str(r[k])) for k in ATTRS]))
    return rows


def to_tensor(img_rgb_u8: np.ndarray) -> torch.Tensor:
    x = (img_rgb_u8.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy())


class CardDataset(Dataset):
    """Bilder + Labels aus <dir>/<split>.csv bzw. einer CSV mit Bildordner."""

    def __init__(self, img_dir: Path, csv_path: Path, augment: bool = False, limit: int | None = None):
        self.img_dir = Path(img_dir)
        self.rows = read_label_csv(csv_path)
        if limit:
            self.rows = self.rows[:limit]
        self.augment = augment

    def __len__(self):
        return len(self.rows)

    def _aug(self, im: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        h, w = im.shape[:2]
        # leichte Perspektive (Eckfehler des Detektors) und Skalierung
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = src + rng.normal(0, 0.012 * w, src.shape).astype(np.float32)
        M = cv2.getPerspectiveTransform(src, dst)
        im = cv2.warpPerspective(im, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        # Helligkeit/Kontrast/Sättigung (Farbton bewusst nicht: rot != lila)
        x = im.astype(np.float32)
        x = x * rng.uniform(0.8, 1.2) + rng.uniform(-15, 15)
        g = x.mean(-1, keepdims=True)
        x = g + (x - g) * rng.uniform(0.8, 1.2)
        x *= rng.uniform(0.94, 1.06, 3)  # leichter Kanal-Stich
        im = np.clip(x, 0, 255).astype(np.uint8)
        if rng.random() < 0.2:
            im = cv2.GaussianBlur(im, (0, 0), rng.uniform(0.3, 1.0))
        if rng.random() < 0.3:  # erneut komprimieren / Auflösung verlieren
            s = rng.uniform(0.5, 1.0)
            im = cv2.resize(cv2.resize(im, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA), (w, h),
                            interpolation=cv2.INTER_LINEAR)
        return im

    def __getitem__(self, i):
        fn, y = self.rows[i]
        im = cv2.imread(str(self.img_dir / fn), cv2.IMREAD_COLOR)[..., ::-1]
        if im.shape[:2] != (IN_H, IN_W):
            im = cv2.resize(im, (IN_W, IN_H), interpolation=cv2.INTER_AREA)
        if self.augment:
            rng = np.random.default_rng()
            im = self._aug(np.ascontiguousarray(im), rng)
            if rng.random() < 0.5:  # Lage 0°/180° ist bei der Entzerrung nicht festgelegt
                im = im[::-1, ::-1]
        return to_tensor(np.ascontiguousarray(im)), torch.tensor(y)


class NegativeDataset(Dataset):
    """Kartenförmige Ausschnitte OHNE Karte: Untergrund aus synthetischen Szenen und echten Hintergründen.

    Simuliert Fehlalarme des Detektors. Ausschnitte dürfen Karten anschneiden, ihre Mitte liegt aber
    nie auf einer Karte. Label: alle Attribute -1 (ignoriert), card = 0.
    """

    def __init__(self, img_dirs: list[Path], label_dirs: list[Path | None], n: int, seed: int | None = None,
                 paper_from: "CardDataset | None" = None, paper_frac: float = 0.25):
        self.paper_from, self.paper_frac = paper_from, paper_frac
        self.items = []
        for d, ld in zip(img_dirs, label_dirs):
            for f in sorted(Path(d).glob("*.jpg")):
                self.items.append((f, Path(ld) / (f.stem + ".txt") if ld else None))
        self.n, self.seed = n, seed

    def __len__(self):
        return self.n

    @staticmethod
    def _boxes(lbl: Path | None, w: int, h: int) -> np.ndarray:
        if lbl is None or not lbl.exists():
            return np.zeros((0, 4))
        b = []
        for line in lbl.read_text().split("\n"):
            v = line.split()
            if len(v) >= 5:
                cx, cy, bw, bh = (float(x) for x in v[1:5])
                b.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
        return np.array(b).reshape(-1, 4)

    def __getitem__(self, i):
        rng = np.random.default_rng(None if self.seed is None else [self.seed, i])
        if self.paper_from is not None and rng.random() < self.paper_frac:
            # leeres Kartenpapier: schmaler Rand oben/unten einer echten Karte, stark vergrößert
            fn, _ = self.paper_from.rows[int(rng.integers(len(self.paper_from.rows)))]
            im = cv2.imread(str(self.paper_from.img_dir / fn), cv2.IMREAD_COLOR)
            im = cv2.resize(im, (IN_W, IN_H), interpolation=cv2.INTER_AREA)
            band = int(rng.integers(10, 18))
            part = im[:band] if rng.random() < 0.5 else im[-band:]
            x0 = int(rng.integers(0, IN_W // 3))
            part = part[:, x0:x0 + int(rng.integers(IN_W // 2, IN_W - x0 + 1))]
            crop = cv2.resize(part, (IN_W, IN_H), interpolation=cv2.INTER_CUBIC)[..., ::-1].astype(np.float32)
            crop += rng.normal(0, rng.uniform(1, 6), crop.shape)
            crop = np.clip(crop * rng.uniform(0.75, 1.2), 0, 255).astype(np.uint8)
            return to_tensor(np.ascontiguousarray(crop)), torch.full((len(ATTRS),), -1, dtype=torch.long), \
                torch.tensor(0)
        for _ in range(20):
            f, lbl = self.items[int(rng.integers(len(self.items)))]
            im = cv2.imread(str(f), cv2.IMREAD_COLOR)
            if im is None:
                continue
            h, w = im.shape[:2]
            boxes = self._boxes(lbl, w, h)
            short = rng.uniform(0.06, 0.45) * min(h, w)
            c = np.array([rng.uniform(0.1, 0.9) * w, rng.uniform(0.1, 0.9) * h])
            if len(boxes) and ((c[0] > boxes[:, 0]) & (c[0] < boxes[:, 2]) & (c[1] > boxes[:, 1])
                               & (c[1] < boxes[:, 3])).any():
                continue
            ang = rng.uniform(0, np.pi)
            u, v = np.array([np.cos(ang), np.sin(ang)]), np.array([-np.sin(ang), np.cos(ang)])
            sx, sy = short / 2 * rng.uniform(0.8, 1.2), short * 1.53 / 2 * rng.uniform(0.6, 1.2)
            q = np.array([c - sx * u - sy * v, c + sx * u - sy * v, c + sx * u + sy * v, c - sx * u + sy * v])
            q += rng.normal(0, 0.06 * short, q.shape)
            dst = np.float32([[0, 0], [IN_W, 0], [IN_W, IN_H], [0, IN_H]]) - 0.5
            M = cv2.getPerspectiveTransform(q.astype(np.float32), dst)
            crop = cv2.warpPerspective(im, M, (IN_W, IN_H), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)
            x = crop[..., ::-1].astype(np.float32)
            x = x * rng.uniform(0.7, 1.3) + rng.uniform(-20, 20)  # auch helle, papierähnliche Flächen
            crop = np.clip(x, 0, 255).astype(np.uint8)
            return to_tensor(np.ascontiguousarray(crop)), torch.full((len(ATTRS),), -1, dtype=torch.long), \
                torch.tensor(0)
        return self[i + 1]


class WithCard(Dataset):
    """Hängt an ein CardDataset das Label card = 1 an."""

    def __init__(self, ds: Dataset):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        x, y = self.ds[i]
        return x, y, torch.tensor(1)
