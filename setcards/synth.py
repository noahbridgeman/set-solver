"""Synthetische Szenen mit SET-Karten für Detektor (YOLO-pose, 4 Ecken) und Klassifikator.

Ablauf einer Szene
1. Kamera: Lochkamera mit zufälligem Bildwinkel, Neigung (0-55°), Azimut, Rollwinkel.
2. Karten liegen auf der Tischebene (Raster wie im Spiel oder verstreut, auch überlappend),
   jede leicht gewölbt. Jede Karte wird mit eigener Druckvariation gerendert (render.Style)
   und per Strahl-Flächen-Schnitt exakt (inkl. Wölbung) ins Bild projiziert.
3. Licht: Schlagschatten der Karten, Schattenflecken, Helligkeitsverlauf, Glanzlichter,
   Lichtfarbe mit unvollständigem Weißabgleich, Belichtung (inkl. Clipping).
4. Kamera: Defokus, Bewegungsunschärfe, Sensorrauschen, Tonkurve, Sättigung, JPEG.

Alles bis Schritt 3 läuft in linearem Licht.

Eck-Reihenfolge der Keypoints (für Detektor und spätere Entzerrung):
im Bild im Uhrzeigersinn, k0-k1 ist die kurze Kante, deren Mitte weiter oben im Bild liegt,
k0 ist deren erster Punkt. Entzerren k0..k3 -> (0,0),(W,0),(W,H),(0,H) ergibt das Hochformat.
Bei horizontaler Spiegelung gilt flip_idx = [1, 0, 3, 2].
"""
from __future__ import annotations

import string
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .common import CARD_H, CARD_W, lin_to_srgb, srgb_to_lin
from .render import Card, Style, all_cards, load_geometry, render_raster

CROP_W, CROP_H = 128, 192
ALL_CARDS = all_cards()


# ====================================================================== Druckvariation
def random_style(rng: np.random.Generator) -> Style:
    return Style(
        offset=(float(rng.normal(0, 0.45)), float(rng.normal(0, 0.6))),
        rot_deg=float(rng.normal(0, 0.3)),
        symbol_scale=float(rng.normal(1, 0.008)),
        pitch_scale=float(rng.normal(1, 0.01)),
        stroke_delta=float(rng.normal(0, 0.06)),
        outset_delta=float(rng.normal(0, 0.04)),
        stripe_period_scale=float(rng.normal(1, 0.01)),
        stripe_width_delta=float(rng.normal(0, 0.04)),
        stripe_phase_delta=float(rng.uniform(-0.3, 0.3)),
        ink_dlab=tuple(rng.normal(0, [2.5, 3.0, 3.0]).tolist()),
        line_dlab=tuple(rng.normal(0, [3.0, 3.5, 3.5]).tolist()),
        paper_lab=(float(rng.uniform(88, 97)), float(rng.normal(0, 0.8)), float(rng.normal(1.5, 1.5))),
        print_blur_mm=float(abs(rng.normal(0, 0.04))),
    )


# ====================================================================== Hintergründe
def _noise(h: int, w: int, rng, scale: float, octaves: int = 4) -> np.ndarray:
    """Fraktales Rauschen in [-1, 1] (Summe hochskalierter Zufallsraster)."""
    out = np.zeros((h, w), np.float32)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        gs = max(2, int(max(h, w) / (scale / 2 ** o)))
        g = rng.normal(0, 1, (max(2, gs * h // max(h, w) + 2), gs + 2)).astype(np.float32)
        out += amp * cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC)
        tot += amp
        amp *= 0.5
    out /= tot
    return np.clip(out / (np.abs(out).max() + 1e-6), -1, 1)


def _rand_color(rng, dark: bool = False) -> np.ndarray:
    hsv = np.uint8([[[rng.integers(0, 180), rng.integers(0, 256), rng.integers(30 if dark else 60, 140 if dark else 256)]]])
    return srgb_to_lin(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0] / 255.0)


def bg_procedural(h: int, w: int, rng) -> np.ndarray:
    kind = rng.choice(["flat", "wood", "fabric", "tiles", "gradient"], p=[0.2, 0.25, 0.25, 0.15, 0.15])
    if kind == "flat":
        c = _rand_color(rng, dark=rng.random() < 0.4)
        img = c * (1 + 0.15 * _noise(h, w, rng, 300))[..., None]
    elif kind == "wood":
        base = srgb_to_lin(np.array([rng.uniform(0.35, 0.85), rng.uniform(0.2, 0.6), rng.uniform(0.08, 0.4)]))
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        a = rng.uniform(0, np.pi)
        u = np.cos(a) * xx + np.sin(a) * yy
        warp = 40 * _noise(h, w, rng, 400)
        grain = np.sin((u + warp) * rng.uniform(0.03, 0.12)) * 0.5 + 0.5
        fine = _noise(h, w, rng, 30, 2)
        img = base * (0.75 + 0.25 * grain[..., None] + 0.06 * fine[..., None])
    elif kind == "fabric":
        c = _rand_color(rng, dark=rng.random() < 0.5)
        fine = rng.normal(0, 1, (h, w)).astype(np.float32)
        fine = cv2.GaussianBlur(fine, (0, 0), rng.uniform(0.6, 2.0))
        img = c * (1 + 0.25 * fine / (fine.std() + 1e-6) * 0.4 + 0.1 * _noise(h, w, rng, 200))[..., None]
    elif kind == "tiles":
        img = np.empty((h, w, 3), np.float32)
        img[:] = _rand_color(rng)
        s = int(rng.uniform(40, 300))
        c2 = _rand_color(rng)
        yy, xx = np.mgrid[0:h, 0:w]
        if rng.random() < 0.5:
            m = ((xx // s + yy // s) % 2) == 1
        else:
            m = (xx // s) % 2 == 1
        img[m] = c2
        img = cv2.GaussianBlur(img, (0, 0), 1.0)
    else:
        c1, c2 = _rand_color(rng), _rand_color(rng)
        t = np.linspace(0, 1, w, dtype=np.float32)[None, :, None]
        img = np.broadcast_to(c1 * (1 - t) + c2 * t, (h, w, 3)).copy()
        M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(0, 360), 1.5)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    return np.clip(img, 0, 1).astype(np.float32)


def add_clutter(img: np.ndarray, rng, n: int) -> None:
    """Störobjekte: Flächen, Linien, Text (keine kartenähnlichen weißen Rechtecke)."""
    h, w = img.shape[:2]
    for _ in range(n):
        c = tuple(float(v) for v in _rand_color(rng))
        k = rng.integers(0, 4)
        p = (int(rng.uniform(0, w)), int(rng.uniform(0, h)))
        if k == 0:
            cv2.circle(img, p, int(rng.uniform(10, 0.15 * w)), c, -1, cv2.LINE_AA)
        elif k == 1:
            q = (int(rng.uniform(0, w)), int(rng.uniform(0, h)))
            cv2.line(img, p, q, c, int(rng.uniform(2, 20)), cv2.LINE_AA)
        elif k == 2:
            pts = (np.array(p) + rng.normal(0, 0.08 * w, (int(rng.integers(3, 7)), 2))).astype(np.int32)
            cv2.fillPoly(img, [pts], c, cv2.LINE_AA)
        else:
            txt = "".join(rng.choice(list(string.ascii_letters + string.digits + "  "), int(rng.integers(3, 14))))
            m = np.zeros((h, w), np.uint8)  # putText kann (OpenCV 5) nur 8-Bit-Bilder
            cv2.putText(m, txt, p, int(rng.integers(0, 8)), rng.uniform(0.8, 4), 255, int(rng.integers(1, 6)),
                        cv2.LINE_AA)
            a = (m / 255.0).astype(np.float32)[..., None]
            img[:] = img * (1 - a) + np.array(c, np.float32) * a


class Backgrounds:
    def __init__(self, folder: Path | None):
        exts = {".jpg", ".jpeg", ".png", ".webp"}
        self.files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in exts) if folder and folder.exists() else []

    def sample(self, h: int, w: int, rng) -> np.ndarray:
        if self.files and rng.random() < 0.6:
            im = cv2.imread(str(self.files[rng.integers(len(self.files))]), cv2.IMREAD_COLOR)
            if im is not None:
                im = srgb_to_lin(im[..., ::-1] / 255.0).astype(np.float32)
                if rng.random() < 0.5:
                    im = im[:, ::-1]
                im = np.rot90(im, int(rng.integers(0, 4)))
                ih, iw = im.shape[:2]
                s = max(h / ih, w / iw) * rng.uniform(1.0, 1.8)
                im = cv2.resize(np.ascontiguousarray(im), (int(iw * s) + 1, int(ih * s) + 1), interpolation=cv2.INTER_AREA
                                if s < 1 else cv2.INTER_LINEAR)
                y0 = int(rng.integers(0, im.shape[0] - h + 1))
                x0 = int(rng.integers(0, im.shape[1] - w + 1))
                img = im[y0:y0 + h, x0:x0 + w].copy()
                img *= rng.uniform(0.7, 1.3, 3).astype(np.float32)  # Farbstich der Vorlage variieren
                return np.clip(img, 0, 1)
        img = bg_procedural(h, w, rng)
        if rng.random() < 0.5:
            add_clutter(img, rng, int(rng.integers(1, 12)))
        return img


# ====================================================================== Kamera und Geometrie
@dataclass
class Camera:
    K: np.ndarray
    R: np.ndarray  # Welt -> Kamera
    C: np.ndarray  # Kamerazentrum (Welt, mm)
    w: int
    h: int

    def project(self, P: np.ndarray) -> np.ndarray:
        Xc = (np.asarray(P, np.float64) - self.C) @ self.R.T
        uv = Xc @ self.K.T
        return uv[..., :2] / uv[..., 2:3]

    def rays(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """Blickrichtungen (Welt) für Pixelmittelpunkte."""
        pix = np.stack([xs + 0.5, ys + 0.5, np.ones_like(xs)], -1)
        d = pix @ np.linalg.inv(self.K).T
        return d @ self.R  # R^T * d


def make_camera(w: int, h: int, target: np.ndarray, dist: float, tilt: float, azim: float, roll: float,
                hfov_deg: float) -> Camera:
    f = 0.5 * max(w, h) / np.tan(np.radians(hfov_deg) / 2)
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], np.float64)
    dvec = np.array([np.sin(tilt) * np.cos(azim), np.sin(tilt) * np.sin(azim), np.cos(tilt)])
    C = target + dist * dvec
    z = -dvec
    x = np.cross(z, [0, 0, 1.0])
    x = x / np.linalg.norm(x) if np.linalg.norm(x) > 1e-6 else np.array([1.0, 0, 0])
    if np.linalg.norm(np.cross(z, [0, 0, 1.0])) <= 1e-6:
        x = np.array([np.cos(azim), np.sin(azim), 0.0])
    y = np.cross(z, x)
    cr, sr = np.cos(roll), np.sin(roll)
    x, y = cr * x + sr * y, -sr * x + cr * y
    return Camera(K, np.stack([x, y, z]), C, w, h)


@dataclass
class CardPose:
    card: Card
    center: np.ndarray            # Tischkoordinaten (mm)
    phi: float                    # Drehung in der Tischebene
    z0: float                     # Höhe (Stapelreihenfolge)
    bend: float                   # Wölbung (mm, Mitte gegenüber Rand)
    bend_axis: int                # 0: Wölbung entlang u, 1: entlang v
    corner_lift: tuple = (0, 0.0)  # (Ecke, Höhe mm)
    style: Style = field(default_factory=Style)

    def height(self, u, v):
        """Höhe der Kartenoberfläche über dem Tisch an Kartenkoordinaten (u, v), Ursprung = Kartenmitte."""
        t = (2 * u / CARD_W) if self.bend_axis == 0 else (2 * v / CARD_H)
        z = self.z0 + self.bend * (1 - np.clip(t, -1, 1) ** 2)
        ci, lift = self.corner_lift
        if lift:
            sx, sy = [(-1, -1), (1, -1), (1, 1), (-1, 1)][ci]
            r = np.clip((sx * u / (CARD_W / 2) + sy * v / (CARD_H / 2) - 1.2) / 0.8, 0, None)
            z = z + lift * r ** 2
        return z - (self.bend if self.bend < 0 else 0)  # nach unten gewölbt: Ränder liegen auf

    def to_world(self, u, v):
        c, s = np.cos(self.phi), np.sin(self.phi)
        # Karten-v zeigt im Bild nach unten -> auf dem Tisch nach -y (sonst wäre die Karte gespiegelt)
        x = self.center[0] + c * u - s * (-v)
        y = self.center[1] + s * u + c * (-v)
        return np.stack([x, y, self.height(u, v)], -1)

    def to_card(self, x, y):
        c, s = np.cos(self.phi), np.sin(self.phi)
        dx, dy = x - self.center[0], y - self.center[1]
        return c * dx + s * dy, -(-s * dx + c * dy)


CORNERS_UV = np.array([[-CARD_W / 2, -CARD_H / 2], [CARD_W / 2, -CARD_H / 2],
                       [CARD_W / 2, CARD_H / 2], [-CARD_W / 2, CARD_H / 2]])


def order_keypoints(p: np.ndarray) -> np.ndarray:
    """Ecken (Kartenreihenfolge TL,TR,BR,BL) -> Keypoint-Reihenfolge laut Modulbeschreibung."""
    area = 0.5 * np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1])
    if area < 0:  # im Bild gegen den Uhrzeigersinn -> umdrehen, kurze Kante bleibt 0-1
        p = p[[1, 0, 3, 2]]
    # kurze Kanten sind (0,1) und (2,3); die obere (kleineres y der Mitte) wird k0-k1
    if (p[0, 1] + p[1, 1]) <= (p[2, 1] + p[3, 1]):
        return p
    return p[[2, 3, 0, 1]]


# ====================================================================== Layout
def _overlap_ok(poses: list[CardPose], new: CardPose, max_frac: float) -> bool:
    if not poses:
        return True
    s = 0.5  # px/mm
    pts_new = new.to_world(CORNERS_UV[:, 0], CORNERS_UV[:, 1])[:, :2]
    allp = np.vstack([pts_new] + [p.to_world(CORNERS_UV[:, 0], CORNERS_UV[:, 1])[:, :2] for p in poses])
    lo = allp.min(0) - 2
    size = np.ceil((allp.max(0) - lo + 4) * s).astype(int)

    def mask(pts):
        m = np.zeros((size[1], size[0]), np.uint8)
        cv2.fillPoly(m, [np.round((pts - lo) * s).astype(np.int32)], 1)
        return m.astype(bool)
    mn = mask(pts_new)
    an = mn.sum()
    for p in poses:
        mo = mask(p.to_world(CORNERS_UV[:, 0], CORNERS_UV[:, 1])[:, :2])
        inter = (mn & mo).sum()
        if inter > max_frac * min(an, mo.sum()):
            return False
    return True


def make_layout(rng) -> list[CardPose]:
    mode = rng.choice(["grid", "scatter", "overlap"], p=[0.55, 0.3, 0.15])
    poses: list[CardPose] = []

    def pose(center, phi, z0):
        bend = float(np.clip(rng.normal(0, 0.9), -3, 3)) if rng.random() < 0.8 else 0.0
        lift = (int(rng.integers(0, 4)), float(rng.uniform(0.5, 4))) if rng.random() < 0.12 else (0, 0.0)
        return CardPose(card=None, center=np.asarray(center, float), phi=float(phi), z0=z0, bend=bend,
                        bend_axis=int(rng.integers(0, 2)), corner_lift=lift, style=random_style(rng))

    if mode == "grid":
        rows, cols = [(3, 4), (3, 4), (3, 5), (3, 6), (4, 3), (3, 3), (2, 3), (3, 7), (1, 3), (2, 2)][rng.integers(10)]
        landscape = rng.random() < 0.25
        cw, ch = (CARD_H, CARD_W) if landscape else (CARD_W, CARD_H)
        gx, gy = rng.uniform(3, 25), rng.uniform(3, 25)
        rot = rng.uniform(0, 2 * np.pi)
        drop = rng.random(rows * cols) < (0.15 if rng.random() < 0.4 else 0.0)
        for r in range(rows):
            for c in range(cols):
                if drop[r * cols + c]:
                    continue
                lx = (c - (cols - 1) / 2) * (cw + gx) + rng.normal(0, 3)
                ly = (r - (rows - 1) / 2) * (ch + gy) + rng.normal(0, 3)
                x = np.cos(rot) * lx - np.sin(rot) * ly
                y = np.sin(rot) * lx + np.cos(rot) * ly
                phi = rot + (np.pi / 2 if landscape else 0) + rng.normal(0, np.radians(4)) + np.pi * rng.integers(0, 2)
                p = pose((x, y), phi, 0.0)
                if _overlap_ok(poses, p, 0.02):
                    poses.append(p)
    else:
        n = int(rng.integers(1, 16)) if mode == "scatter" else int(rng.integers(2, 8))
        spread = rng.uniform(60, 260)
        max_frac = 0.0 if mode == "scatter" else 0.5
        for i in range(n * 6):
            if len(poses) >= n:
                break
            p = pose(rng.normal(0, spread, 2), rng.uniform(0, 2 * np.pi), 0.35 * len(poses))
            if _overlap_ok(poses, p, max_frac):
                poses.append(p)
    cards = rng.choice(len(ALL_CARDS), size=len(poses), replace=False)  # wie im Spiel: keine Duplikate
    for p, k in zip(poses, cards):
        p.card = ALL_CARDS[int(k)]
    return poses


# ====================================================================== Licht
def kelvin_rgb(T: float) -> np.ndarray:
    """Grobe Schwarzkörperfarbe (linear, auf G normiert)."""
    T = T / 100
    r = 255 if T <= 66 else 329.7 * (T - 60) ** -0.1332
    g = 99.47 * np.log(T) - 161.1 if T <= 66 else 288.1 * (T - 60) ** -0.0755
    b = 255 if T >= 66 else (0 if T <= 19 else 138.5 * np.log(T - 10) - 305.0)
    c = srgb_to_lin(np.clip(np.array([r, g, b]) / 255.0, 0.01, 1))
    return c / c[1]


def soft_blobs(h, w, rng, n, scale) -> np.ndarray:
    """Weiche Flecken (Schatten, Glanz); in 1/8 Auflösung berechnet, da sehr unscharf."""
    f = 8
    hs, ws = max(h // f, 2), max(w // f, 2)
    m = np.zeros((hs, ws), np.float32)
    for _ in range(n):
        pts = (np.array([rng.uniform(-0.2, 1.2) * ws, rng.uniform(-0.2, 1.2) * hs])
               + rng.normal(0, scale * max(hs, ws), (int(rng.integers(3, 8)), 2))).astype(np.int32)
        cv2.fillPoly(m, [pts], 1.0)
    k = rng.uniform(0.01, 0.08) * max(hs, ws)
    return cv2.resize(cv2.GaussianBlur(m, (0, 0), max(k, 0.5)), (w, h), interpolation=cv2.INTER_LINEAR)


# ====================================================================== Szene
@dataclass
class SceneResult:
    image: np.ndarray            # sRGB uint8 (H, W, 3), RGB, nach Kamera-Pipeline
    keypoints: list[np.ndarray]  # je Karte 4x3 (x, y, sichtbar 0/1/2)
    boxes: list[np.ndarray]      # je Karte x0, y0, x1, y1
    cards: list[Card]
    visible_frac: list[float]
    inside: list[bool]           # alle Ecken im Bild


def scene_setup(rng: np.random.Generator) -> tuple[int, int, list[CardPose]]:
    """Bildgröße und Kartenlayout. Erste Zufallsziehungen einer Szene: Reihenfolge nicht ändern,
    sonst lassen sich Labels bestehender Datensätze nicht mehr rekonstruieren (gen_synth --rebuild-csv)."""
    long_side = int(rng.choice([1280, 1600, 1920]))
    aspect = float(rng.choice([4 / 3, 16 / 9, 3 / 2]))
    W, H = (long_side, int(long_side / aspect)) if rng.random() < 0.6 else (int(long_side / aspect), long_side)
    poses = make_layout(rng) if rng.random() > 0.04 else []
    return W, H, poses


def depth_of_field(img: np.ndarray, cam: Camera, rng, sigma_max: float, levels: int = 5) -> np.ndarray:
    """Unschärfe wächst mit dem Abstand (in 1/Entfernung) zur Fokusebene auf dem Tisch (z = 0)."""
    H, W = img.shape[:2]
    ys, xs = np.mgrid[0:H:8, 0:W:8].astype(np.float64)
    d = cam.rays(xs, ys)
    dz = np.minimum(d[..., 2], -1e-3)              # Strahlen über dem Horizont -> sehr weit weg
    inv = np.clip(-dz / max(cam.C[2], 1.0), 1e-6, None)   # 1/Entfernung bis zum Tisch
    focus = float(np.quantile(inv, rng.uniform(0.2, 0.8)))
    blur = np.abs(inv - focus) / max(focus, 1e-6)
    blur = np.clip(blur / max(np.quantile(blur, 0.95), 1e-6), 0, 1)
    blur = cv2.resize(blur.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR) * (levels - 1)
    stack = [img] + [cv2.GaussianBlur(img, (0, 0), sigma_max * k / (levels - 1)) for k in range(1, levels)]
    lo = np.floor(blur).astype(int).clip(0, levels - 2)
    w = (blur - lo)[..., None]
    out = np.empty_like(img)
    for k in range(levels - 1):
        m = lo == k
        out[m] = (stack[k][m] * (1 - w[m]) + stack[k + 1][m] * w[m])
    return out


def render_scene(rng: np.random.Generator, bgs: Backgrounds, g: dict | None = None) -> SceneResult:
    g = g or load_geometry()
    W, H, poses = scene_setup(rng)
    # Kamera: Abstand so wählen, dass das Layout einen Zielanteil des Bildes füllt
    # v2: ein gutes Drittel der Szenen mit flachem Blick über den Tisch (Live-Test: 65-75° kamen vor)
    if rng.random() < 0.35:
        tilt = np.radians(rng.uniform(40, 78))
    else:
        tilt = np.radians(min(abs(rng.normal(0, 25)), 58))
    azim, roll = rng.uniform(0, 2 * np.pi), rng.uniform(0, 2 * np.pi)
    hfov = rng.uniform(50, 75)
    if poses:
        world = np.vstack([p.to_world(CORNERS_UV[:, 0], CORNERS_UV[:, 1]) for p in poses])
        target = world.mean(0) * [1, 1, 0]
        ext = np.ptp(world[:, :2], 0).max() + CARD_H
    else:
        target, ext = np.zeros(3), 300.0
    target = target + np.r_[rng.normal(0, 0.12 * ext, 2), 0]
    # v2: ein Viertel der Szenen nah dran, Karten am Bildrand angeschnitten
    fill = rng.uniform(1.2, 2.3) if rng.random() < 0.25 else rng.uniform(0.5, 1.15)
    dist = 600.0
    for _ in range(3):
        cam = make_camera(W, H, target, dist, tilt, azim, roll, hfov)
        if not poses:
            break
        pr = cam.project(world)
        span = max(np.ptp(pr[:, 0]) / W, np.ptp(pr[:, 1]) / H)
        dist = float(np.clip(dist * span / fill, 120, 4000))
    cam = make_camera(W, H, target, dist, tilt, azim, roll, hfov)

    img = bgs.sample(H, W, rng)
    light_dir = np.array([rng.normal(0, 0.5), rng.normal(0, 0.5), 1.0])
    light_dir /= np.linalg.norm(light_dir)
    shadow_strength = rng.uniform(0.05, 0.45)
    ids = np.full((H, W), -1, np.int32)
    full_px, kps, boxes, cards = [], [], [], []

    for i, p in enumerate(poses):
        # Umriss projizieren -> Bildbereich der Karte
        t = np.linspace(0, 1, 12)
        edge = np.vstack([np.c_[-CARD_W / 2 + CARD_W * t, np.full_like(t, -CARD_H / 2)],
                          np.c_[np.full_like(t, CARD_W / 2), -CARD_H / 2 + CARD_H * t],
                          np.c_[CARD_W / 2 - CARD_W * t, np.full_like(t, CARD_H / 2)],
                          np.c_[np.full_like(t, -CARD_W / 2), CARD_H / 2 - CARD_H * t]])
        outline = cam.project(p.to_world(edge[:, 0], edge[:, 1]))
        corners = cam.project(p.to_world(CORNERS_UV[:, 0], CORNERS_UV[:, 1]))
        x0, y0 = np.floor(outline.min(0)).astype(int) - 2
        x1, y1 = np.ceil(outline.max(0)).astype(int) + 3
        bx0, by0, bx1, by1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
        cards.append(p.card)
        kps.append(corners)
        boxes.append(np.array([x0, y0, x1, y1], float))
        # Fläche der ganzen Karte im Bild (auch außerhalb), für Sichtbarkeit
        full_px.append(abs(cv2.contourArea(outline.astype(np.float32))))
        if bx1 <= bx0 or by1 <= by0:
            continue

        # Texturauflösung ~ projizierter Maßstab (px/mm) an der Karte
        ctr = cam.project(p.to_world(np.array([0.0, 1.0]), np.array([0.0, 0.0])))
        scale = float(np.linalg.norm(ctr[1] - ctr[0]))
        corner_scales = [np.linalg.norm(corners[(k + 1) % 4] - corners[k]) / (CARD_W if k % 2 == 0 else CARD_H)
                         for k in range(4)]
        ppmm = float(np.clip(max(scale, *corner_scales) * 1.2, 1.5, 24))
        rgb, alpha = render_raster(p.card, ppmm, g, p.style)
        # Papiermaserung / leichte Verschmutzung
        grain = 1 + 0.015 * rng.normal(0, 1, alpha.shape).astype(np.float32) \
            + 0.02 * _noise(alpha.shape[0], alpha.shape[1], rng, 200, 3)
        rgb = rgb * grain[..., None]

        # Strahl-Flächen-Schnitt: Pixel -> Kartenkoordinaten (iterativ wegen Wölbung)
        ys, xs = np.mgrid[by0:by1, bx0:bx1].astype(np.float64)
        d = cam.rays(xs, ys)
        zc = np.full(xs.shape, p.z0 + max(p.bend, 0) * 0.5)
        for _ in range(4):
            tt = (zc - cam.C[2]) / d[..., 2]
            X = cam.C[0] + tt * d[..., 0]
            Y = cam.C[1] + tt * d[..., 1]
            u, v = p.to_card(X, Y)
            zc = p.height(u, v)
        mx = ((u + CARD_W / 2) * ppmm - 0.5).astype(np.float32)
        my = ((v + CARD_H / 2) * ppmm - 0.5).astype(np.float32)
        tex = cv2.remap(np.dstack([rgb, alpha]), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                        borderValue=0)
        a = tex[..., 3]
        # Schattierung durch Wölbung (Lambert relativ zur flachen Karte)
        eps = 0.5
        dzdu = (p.height(u + eps, v) - p.height(u - eps, v)) / (2 * eps)
        dzdv = (p.height(u, v + eps) - p.height(u, v - eps)) / (2 * eps)
        c_, s_ = np.cos(p.phi), np.sin(p.phi)
        # Gradient in Weltkoordinaten (u -> (c, s), v -> (s, -c))
        gx = dzdu * c_ + dzdv * s_
        gy = dzdu * s_ - dzdv * c_
        n = np.stack([-gx, -gy, np.ones_like(gx)], -1)
        n /= np.linalg.norm(n, axis=-1, keepdims=True)
        shade = np.clip((n @ light_dir) / light_dir[2], 0.6, 1.3)[..., None]

        # Schlagschatten auf den Untergrund
        sh = cv2.GaussianBlur(a, (0, 0), max(1.0, scale * rng.uniform(0.5, 2.0)))
        off = (light_dir[:2] / light_dir[2]) * scale * (1 + p.z0 + max(p.bend, 0)) * -1
        M = np.float32([[1, 0, off[0]], [0, 1, off[1]]])
        sh = cv2.warpAffine(sh, M, (sh.shape[1], sh.shape[0]))
        reg = img[by0:by1, bx0:bx1]
        reg *= (1 - shadow_strength * sh)[..., None]
        reg[:] = reg * (1 - a[..., None]) + tex[..., :3] * shade * a[..., None]
        ids[by0:by1, bx0:bx1][a > 0.5] = i

    # Glanzlichter auf den Karten
    card_mask = (ids >= 0).astype(np.float32)
    if poses and rng.random() < 0.3:
        gl = soft_blobs(H, W, rng, int(rng.integers(1, 3)), 0.06) * card_mask
        img += (rng.uniform(0.1, 0.5) * gl)[..., None]
    # Beleuchtung: Verlauf, Vignette, Schattenflecken
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    ang = rng.uniform(0, 2 * np.pi)
    grad = ((np.cos(ang) * (xx / W - 0.5) + np.sin(ang) * (yy / H - 0.5)) * rng.uniform(0, 0.8))
    vig = ((xx / W - 0.5) ** 2 + (yy / H - 0.5) ** 2) * rng.uniform(0, 0.8)
    light = np.exp(grad - vig)
    if rng.random() < 0.4:
        light *= 1 - rng.uniform(0.3, 0.75) * soft_blobs(H, W, rng, int(rng.integers(1, 3)), 0.15)
    img *= light[..., None]
    # Lichtfarbe (Mischlicht möglich) und unvollständiger Weißabgleich der Kamera
    T1 = rng.uniform(2300, 7500)
    col = kelvin_rgb(T1)
    if rng.random() < 0.25:
        col2 = kelvin_rgb(rng.uniform(2300, 7500))
        mix = (np.cos(ang + 1) * (xx / W - 0.5) + np.sin(ang + 1) * (yy / H - 0.5) + 0.5).clip(0, 1)[..., None]
        colmap = col * (1 - mix) + col2 * mix
    else:
        colmap = col
    wb = rng.uniform(0.5, 1.0)  # 1 = perfekter Weißabgleich auf die Hauptlichtquelle
    img = img * colmap / (col ** wb)
    # Belichtung: helle Kartenpixel (Papier) landen auf einem Zielwert; > 1 heißt leichtes Ausfressen
    if poses and card_mask.any():
        paper = np.percentile(img[::4, ::4][card_mask[::4, ::4] > 0].max(-1), 75)
        img *= rng.uniform(0.45, 1.1) / max(paper, 1e-4)
    else:
        img *= rng.uniform(0.5, 1.4)

    # ---------------- Kamera
    s = W / 1920
    # v2: Tiefenunschärfe bei schrägem Blick (Schärfe nur in einer Entfernung)
    if tilt > np.radians(30) and rng.random() < 0.5:
        img = depth_of_field(img, cam, rng, sigma_max=rng.uniform(2, 9) * s)
    if rng.random() < 0.7:
        img = cv2.GaussianBlur(img, (0, 0), float(np.clip(abs(rng.normal(0, 0.8)) * s * 1.5, 0.3, 3.0)))
    if rng.random() < 0.15:
        L = int(rng.uniform(3, 10) * s) | 1
        k = np.zeros((L, L), np.float32)
        k[L // 2, :] = 1
        k = cv2.warpAffine(k, cv2.getRotationMatrix2D((L / 2 - 0.5, L / 2 - 0.5), rng.uniform(0, 180), 1), (L, L))
        img = cv2.filter2D(img, -1, k / k.sum())
    shot, read = rng.uniform(0.0003, 0.01), rng.uniform(1e-6, 2e-4)
    img = img + np.sqrt(np.clip(img, 0, None) * shot + read) * rng.normal(0, 1, img.shape).astype(np.float32)
    img = lin_to_srgb(np.clip(img, 0, 1))
    c = rng.uniform(-0.15, 0.25)  # Tonkurve
    img = np.clip(img + c * img * (1 - img) * (2 * img - 1) * -2, 0, 1)
    sat = rng.uniform(0.9, 1.35)
    gray = img.mean(-1, keepdims=True)
    img = np.clip(gray + (img - gray) * sat, 0, 1)
    img8 = (img * 255 + 0.5).astype(np.uint8)
    q = int(rng.uniform(40, 96))
    ok, buf = cv2.imencode(".jpg", img8[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, q])
    img8 = cv2.imdecode(buf, cv2.IMREAD_COLOR)[..., ::-1]

    # ---------------- Labels
    out_kps, vis, inside = [], [], []
    for i, (corners, fp) in enumerate(zip(kps, full_px)):
        k = order_keypoints(corners)
        inb = (k[:, 0] >= 0) & (k[:, 0] < W) & (k[:, 1] >= 0) & (k[:, 1] < H)
        v = np.where(inb, 2, 0).astype(float)
        for j in range(4):  # verdeckte Ecke (eine spätere Karte liegt darüber): sichtbar = 1
            if inb[j] and ids[int(k[j, 1]), int(k[j, 0])] > i:
                v[j] = 1
        out_kps.append(np.c_[k, v])
        vis.append(float((ids == i).sum() / max(fp, 1)))
        inside.append(bool(inb.all()))
    return SceneResult(img8, out_kps, boxes, cards, vis, inside)


def warp_crop(img8: np.ndarray, kp: np.ndarray, rng=None, jitter: float = 0.0,
              size: tuple[int, int] = (CROP_W, CROP_H)) -> np.ndarray:
    """Karte wie in der Inferenz entzerren (k0..k3 -> Hochformat), optional mit Eckrauschen."""
    k = kp[:, :2].astype(np.float64).copy()
    if rng is not None and jitter > 0:
        short = np.linalg.norm(k[1] - k[0])
        k += rng.normal(0, jitter * short, k.shape)
    w2, h2 = size[0] * 2, size[1] * 2
    dst = np.array([[0, 0], [w2, 0], [w2, h2], [0, h2]], np.float64) - 0.5
    M = cv2.getPerspectiveTransform(k.astype(np.float32), dst.astype(np.float32))
    big = cv2.warpPerspective(img8, M, (w2, h2), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return cv2.resize(big, size, interpolation=cv2.INTER_AREA)
