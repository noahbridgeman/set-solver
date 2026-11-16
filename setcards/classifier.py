"""Attribut-Klassifikator: MobileNetV3-small mit 4 Köpfen à 3 Klassen.

Eingabe: entzerrte Karte im Hochformat, 128 x 192 (B x H), RGB, ImageNet-normiert.
Ausgabe: dict Attribut -> Logits (N, 3).
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
