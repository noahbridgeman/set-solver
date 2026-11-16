"""Phase 5: Echtzeit-Erkennung. Detektor -> Entzerren (volle Auflösung) -> Klassifikator -> Tracking -> Sets -> Overlay.

Tracking: Erkennungen werden von Bild zu Bild über die Kartenmitte zugeordnet (Hungarian). Ecken und
Klassen-Wahrscheinlichkeiten werden geglättet; dadurch flackern die Labels nicht und einzelne
unscharfe Bilder (Bewegung) fallen weniger ins Gewicht. Sets werden nur aus sicher erkannten Karten gebildet.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from .classifier import ATTRS, CardNet, to_tensor
from .common import ROOT
from .solver import find_sets
from .synth import warp_crop

ATTR_KEYS = list(ATTRS)
DEFAULT_DET = ROOT / "models" / "detector_v1.pt"
DEFAULT_CLS = ROOT / "models" / "classifier_v1.pt"


def pick_device(name: str = "auto") -> str:
    if name != "auto":
        return name
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


# ================================================================= Erkennung
class Recognizer:
    def __init__(self, det: Path = DEFAULT_DET, cls: Path = DEFAULT_CLS, device: str = "auto",
                 imgsz: int = 640, conf: float = 0.6):
        from ultralytics import YOLO
        self.device = pick_device(device)
        self.det = YOLO(str(det))
        self.imgsz, self.conf = imgsz, conf
        ck = torch.load(str(cls), map_location="cpu", weights_only=False)
        self.cls = CardNet(pretrained=False)
        self.cls.load_state_dict(ck["model"])
        self.cls.eval().to(self.device)

    def detect(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """-> Ecken (N,4,2) in Bildkoordinaten (k0-k1 = obere kurze Kante), Konfidenz (N,)."""
        r = self.det.predict(frame, imgsz=self.imgsz, conf=self.conf, device=self.device, verbose=False)[0]
        if r.keypoints is None or len(r.boxes) == 0:
            return np.zeros((0, 4, 2)), np.zeros(0)
        return r.keypoints.xy.cpu().numpy().astype(np.float64), r.boxes.conf.cpu().numpy()

    @torch.no_grad()
    def classify(self, frame: np.ndarray, kps: np.ndarray) -> np.ndarray:
        """-> Wahrscheinlichkeiten (N, 4 Attribute, 3 Klassen), Mittel über 0°/180°."""
        if len(kps) == 0:
            return np.zeros((0, len(ATTR_KEYS), 3))
        crops = [to_tensor(np.ascontiguousarray(warp_crop(frame, k)[..., ::-1])) for k in kps]
        x = torch.stack(crops).to(self.device)
        o1, o2 = self.cls(x), self.cls(torch.flip(x, dims=(2, 3)))
        p = torch.stack([(o1[k].softmax(1) + o2[k].softmax(1)) / 2 for k in ATTR_KEYS], 1)
        return p.cpu().numpy()


# ================================================================== Tracking
@dataclass
class Track:
    id: int
    kp: np.ndarray            # (4,2)
    probs: np.ndarray         # (4,3)
    hits: int = 1
    misses: int = 0
    n_good: int = 0           # Bilder mit sicherer Einzelerkennung (Karte komplett im Bild)
    confirmed: bool = False   # einmal bestätigt bleibt die Karte sichtbar, solange sie sicher bleibt

    @property
    def label(self) -> tuple[int, ...]:
        return tuple(int(i) for i in self.probs.argmax(1))

    @property
    def confidence(self) -> float:
        return float(self.probs.max(1).min())

    def text(self) -> str:
        return " ".join(str(ATTRS[k][i]) for k, i in zip(ATTR_KEYS, self.label))


def _short_edge(kp: np.ndarray) -> float:
    return float(min(np.linalg.norm(kp[1] - kp[0]), np.linalg.norm(kp[3] - kp[2])))


class Tracker:
    def __init__(self, gate: float = 0.6, max_misses: int = 4, kp_alpha: float = 0.7, prob_alpha: float = 0.35,
                 min_conf: float = 0.8, confirm_frames: int = 2, drop_conf: float = 0.5):
        self.gate, self.max_misses = gate, max_misses
        self.min_conf, self.confirm_frames, self.drop_conf = min_conf, confirm_frames, drop_conf
        self.kp_alpha, self.prob_alpha = kp_alpha, prob_alpha
        self.tracks: list[Track] = []
        self._next = 0

    def reset(self) -> None:
        self.tracks.clear()

    def _observe(self, t: Track, p: np.ndarray, valid: bool) -> None:
        if valid:
            if p.max(1).min() >= self.min_conf:
                t.n_good += 1
        if t.n_good >= self.confirm_frames and t.confidence >= self.min_conf:
            t.confirmed = True
        elif t.confidence < self.drop_conf:
            t.confirmed = False

    def update(self, kps: np.ndarray, probs: np.ndarray, valid: np.ndarray | None = None) -> list[Track]:
        """valid[j] = False: Karte ragt aus dem Bild -> Lage nachführen, Labels nicht anfassen."""
        T, D = len(self.tracks), len(kps)
        if valid is None:
            valid = np.ones(D, bool)
        matched_t, matched_d = set(), set()
        if T and D:
            tc = np.array([t.kp.mean(0) for t in self.tracks])
            dc = kps.mean(1)
            scale = np.array([_short_edge(t.kp) for t in self.tracks])[:, None]
            cost = np.linalg.norm(tc[:, None] - dc[None], axis=2) / np.maximum(scale, 1)
            for i, j in zip(*linear_sum_assignment(cost)):
                if cost[i, j] < self.gate:
                    t, k = self.tracks[i], kps[j]
                    # Reihenfolge kann um 180° springen -> an alte Ecken angleichen
                    k = min((np.roll(k, s, 0) for s in (0, 2)), key=lambda q: np.linalg.norm(q - t.kp))
                    t.kp = self.kp_alpha * k + (1 - self.kp_alpha) * t.kp
                    if valid[j]:
                        t.probs = self.prob_alpha * probs[j] + (1 - self.prob_alpha) * t.probs
                    self._observe(t, probs[j], bool(valid[j]))
                    t.hits += 1
                    t.misses = 0
                    matched_t.add(i)
                    matched_d.add(j)
        for i, t in enumerate(self.tracks):
            if i not in matched_t:
                t.misses += 1
        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]
        for j in range(D):
            if j not in matched_d:
                t = Track(self._next, kps[j].copy(), probs[j].copy())
                self._observe(t, probs[j], bool(valid[j]))
                self.tracks.append(t)
                self._next += 1
        return self.tracks


# =================================================================== Overlay
SET_COLORS = [(255, 0, 255), (255, 255, 0), (0, 165, 255), (0, 255, 0), (255, 100, 0), (0, 255, 255),
              (180, 105, 255), (128, 0, 255)]  # BGR


@dataclass
class FrameResult:
    cards: list[Track]
    sets: list[tuple[int, int, int]]  # Indizes in cards
    timings: dict = field(default_factory=dict)


class LivePipeline:
    def __init__(self, recognizer: Recognizer, min_conf: float = 0.8, min_hits: int = 2, max_misses_show: int = 2):
        self.rec = recognizer
        self.tracker = Tracker(min_conf=min_conf, confirm_frames=min_hits)
        self.min_conf, self.min_hits, self.max_misses_show = min_conf, min_hits, max_misses_show

    def process(self, frame: np.ndarray) -> FrameResult:
        t0 = time.perf_counter()
        kps, _ = self.rec.detect(frame)
        t1 = time.perf_counter()
        probs = self.rec.classify(frame, kps)
        t2 = time.perf_counter()
        h, w = frame.shape[:2]
        tol = np.array([_short_edge(k) * 0.02 for k in kps]).reshape(-1, 1)
        x, y = kps[..., 0], kps[..., 1]
        valid = ((x >= -tol) & (x <= w - 1 + tol) & (y >= -tol) & (y <= h - 1 + tol)).all(1)
        tracks = self.tracker.update(kps, probs, valid)
        cards = [t for t in tracks if t.confirmed and t.misses <= self.max_misses_show]
        sets = find_sets([c.label for c in cards])
        return FrameResult(cards, sets, {"det": t1 - t0, "cls": t2 - t1})


def _scaled(kp: np.ndarray, f: float) -> np.ndarray:
    c = kp.mean(0)
    return c + (kp - c) * f


def draw(frame: np.ndarray, res: FrameResult, min_conf: float = 0.8, show_labels: bool = True,
         only_set: int | None = None) -> np.ndarray:
    """only_set: nur dieses Set (Index, modulo Anzahl) zeigen, sonst alle."""
    out = frame.copy()
    lw = max(2, round(frame.shape[1] / 500))
    fs = frame.shape[1] / 1800
    for c in res.cards:
        col = (200, 200, 200) if c.confidence >= min_conf else (0, 220, 255)
        cv2.polylines(out, [c.kp.astype(np.int32)], True, col, max(1, lw // 2), cv2.LINE_AA)
    # Sets: jede Karte bekommt pro Set einen eigenen, etwas größeren Rahmen
    layer = {}
    sets = list(enumerate(res.sets))
    if only_set is not None and sets:
        sets = [sets[only_set % len(sets)]]
    for si, s in sets:
        col = SET_COLORS[si % len(SET_COLORS)]
        for i in s:
            n = layer.get(i, 0)
            layer[i] = n + 1
            cv2.polylines(out, [_scaled(res.cards[i].kp, 1.04 + 0.05 * n).astype(np.int32)], True, col, lw + 1,
                          cv2.LINE_AA)
    if show_labels:
        for c in res.cards:
            txt = c.text() + ("" if c.confidence >= min_conf else f" ?{c.confidence:.2f}")
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
            x, y = c.kp.mean(0).astype(int)
            x -= tw // 2
            cv2.rectangle(out, (x - 3, y - th - 4), (x + tw + 3, y + 5), (30, 30, 30), -1)
            cv2.putText(out, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), max(1, round(fs * 2)),
                        cv2.LINE_AA)
    return out


def hud(img: np.ndarray, res: FrameResult, fps: float, only_set: int | None = None) -> None:
    n_sets = len(res.sets)
    msg = f"{len(res.cards)} Karten   {n_sets} Set{'s' if n_sets != 1 else ''}" if n_sets else \
        f"{len(res.cards)} Karten   kein Set"
    if only_set is not None and n_sets:
        msg += f"   (zeige {only_set % n_sets + 1}/{n_sets})"
    det_ms = res.timings.get("det", 0) * 1000
    cls_ms = res.timings.get("cls", 0) * 1000
    lines = [msg, f"{fps:4.1f} fps  (Detektor {det_ms:.0f} ms, Klassifikator {cls_ms:.0f} ms)"]
    fs = img.shape[1] / 1400
    y = int(40 * fs * 1.4)
    for i, t in enumerate(lines):
        s = fs * (1.0 if i == 0 else 0.6)
        (tw, th), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, s, 2)
        cv2.rectangle(img, (8, y - th - 8), (16 + tw, y + 8), (0, 0, 0), -1)
        cv2.putText(img, t, (12, y), cv2.FONT_HERSHEY_SIMPLEX, s, (255, 255, 255), 2, cv2.LINE_AA)
        y += th + 22
