"""Vektor-Renderer für die 81 SET-Karten aus data/geometry.json.

- render_svg(): SVG in mm (viewBox 0 0 60 92), Füllungen über clipPath.
- render_raster(): Rasterbild in beliebiger Auflösung (px/mm). Kanten werden
  analytisch über Distanzfelder geglättet, Streifen box-gefiltert; gemischt wird
  in linearem Licht. Ergebnis: lineares RGB + Deckkraft (Kartenform).

Karte im Hochformat, Symbole liegen quer. Die Welle ist nicht punktsymmetrisch;
gerendert wird die gemessene Lage ("kanonisch").
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np

from .common import (CARD_H, CARD_R, CARD_W, COLORS, COUNTS, FILLS, GEOMETRY_JSON, SHAPES, lab_to_lin_rgb,
                     lin_to_srgb)
from .shapes import sdf_rounded_rect, shape_path, shape_sdf


@dataclass(frozen=True)
class Card:
    color: str
    shape: str
    fill: str
    count: int

    @property
    def name(self) -> str:
        return f"{self.color}_{self.shape}_{self.fill}_{self.count}"


def all_cards() -> list[Card]:
    return [Card(c, s, f, n) for c in COLORS for s in SHAPES for f in FILLS for n in COUNTS]


@lru_cache(maxsize=1)
def load_geometry(path: str = str(GEOMETRY_JSON)) -> dict:
    return json.loads(open(path, encoding="utf-8").read())


def symbol_centers(g: dict, count: int, shape: str) -> list[tuple[float, float]]:
    L = g["layout"]
    xc, yc = L["center_x"], L["center_y"]
    p = L["pitch"][shape]
    if count == 1:
        ys = [yc]
    elif count == 2:
        ys = [yc - p["2"] / 2, yc + p["2"] / 2]
    else:
        ys = [yc - p["3"], yc, yc + p["3"]]
    return [(xc, y) for y in ys]


def _outset(g: dict, fill: str) -> float:
    return 0.0 if fill == "voll" else float(g["outline_outset"][fill])


def _ink_lin(g: dict, color: str, line: bool = False) -> np.ndarray:
    """Tintenfarbe (linear). line=True: Farbe für Konturen/Streifen (dunkler gedruckt)."""
    c = g["colors"][color]
    return np.clip(lab_to_lin_rgb(np.array(c["line_lab"] if line and "line_lab" in c else c["lab"])), 0, 1)


def _hex(lin: np.ndarray) -> str:
    s = lin_to_srgb(lin)
    return "#" + "".join(f"{int(round(float(v) * 255)):02x}" for v in s)


# ---------------------------------------------------------------- SVG
def render_svg(card: Card, g: dict | None = None) -> str:
    g = g or load_geometry()
    spec = g["shapes"][card.shape]
    ink = _hex(_ink_lin(g, card.color, line=card.fill != "voll"))
    paper = _hex(np.clip(lab_to_lin_rgb(np.array(g["colors"]["paper"]["lab"])), 0, 1))
    w = g["stroke_width"]
    st = g["stripes"]
    T, lw = st["period"], st["line_width"]
    phase = st["phase_by_shape"][card.shape] * T
    d = _outset(g, card.fill)

    defs, body = [], []
    for i, (cx, cy) in enumerate(symbol_centers(g, card.count, card.shape)):
        path = shape_path(spec, cx, cy, d)
        if card.fill == "voll":
            body.append(f'<path d="{path}" fill="{ink}"/>')
            continue
        cid = f"s{i}"
        defs.append(f'<clipPath id="{cid}"><path d="{path}"/></clipPath>')
        g_parts = [f'<path d="{path}" fill="none" stroke="{ink}" stroke-width="{2 * w:.4f}"/>']
        if card.fill == "gestreift":
            # Streifen über die Symbolbreite, clip schneidet auf die Form zu
            half = spec.get("half_width", 0) or max(abs(p[0]) for p in spec["points"])
            k0 = int(np.floor((-half - 1 - phase) / T))
            k1 = int(np.ceil((half + 1 - phase) / T))
            ys, h = cy - 15, 30
            rects = "".join(f'<rect x="{cx + phase + k * T - lw / 2:.4f}" y="{ys:.4f}" width="{lw:.4f}" height="{h}"/>'
                            for k in range(k0, k1 + 1))
            g_parts.append(f'<g fill="{ink}">{rects}</g>')
        body.append(f'<g clip-path="url(#{cid})">{"".join(g_parts)}</g>')

    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{CARD_W}mm" height="{CARD_H}mm" '
            f'viewBox="0 0 {CARD_W} {CARD_H}">\n'
            f'<title>{card.name}</title>\n'
            f'<defs>{"".join(defs)}</defs>\n'
            f'<rect width="{CARD_W}" height="{CARD_H}" rx="{CARD_R}" ry="{CARD_R}" fill="{paper}"/>\n'
            + "\n".join(body) + "\n</svg>\n")


# ---------------------------------------------------------------- Raster
def _cov(sdf: np.ndarray, px: float) -> np.ndarray:
    """Flächendeckung eines Pixels aus der Distanz zur Kante (lineare Rampe über 1 px)."""
    return np.clip(0.5 - sdf / px, 0.0, 1.0)


def _stripe_cov(u: np.ndarray, T: float, lw: float, px: float) -> np.ndarray:
    """Box-gefilterte Deckung periodischer Streifen (Breite lw, Periode T, Zentrum bei u=0)."""
    def F(v):  # Stammfunktion der Streifen-Indikatorfunktion
        v = v + lw / 2
        k = np.floor(v / T)
        return k * lw + np.minimum(v - k * T, lw)
    return (F(u + px / 2) - F(u - px / 2)) / px


@dataclass
class Style:
    """Druckvariation einer einzelnen Karte (alle Werte in mm bzw. Lab). Standard = gemessenes Modell."""
    offset: tuple[float, float] = (0.0, 0.0)  # Versatz des Drucks gegenüber dem Kartenschnitt
    rot_deg: float = 0.0                       # Drehung des Drucks
    symbol_scale: float = 1.0
    pitch_scale: float = 1.0
    stroke_delta: float = 0.0                  # Strichstärke +/-
    outset_delta: float = 0.0                  # Außenkante leer/gestreift +/-
    stripe_period_scale: float = 1.0
    stripe_width_delta: float = 0.0
    stripe_phase_delta: float = 0.0            # Verschiebung der Streifen (mm)
    ink_dlab: tuple[float, float, float] = (0.0, 0.0, 0.0)
    line_dlab: tuple[float, float, float] = (0.0, 0.0, 0.0)
    paper_lab: tuple[float, float, float] | None = None
    print_blur_mm: float = 0.0                 # Tintenverlauf / Druckunschärfe (Gauß-Sigma)


def render_raster(card: Card, px_per_mm: float = 20.0, g: dict | None = None,
                  style: Style | None = None) -> tuple[np.ndarray, np.ndarray]:
    """-> (lineares RGB HxWx3, Deckkraft HxW). Außerhalb der Karte ist die Deckkraft 0."""
    g = g or load_geometry()
    st_ = style or Style()
    spec = g["shapes"][card.shape]
    px = 1.0 / px_per_mm
    W, H = round(CARD_W * px_per_mm), round(CARD_H * px_per_mm)
    x = ((np.arange(W) + 0.5) * px).astype(np.float32)
    y = ((np.arange(H) + 0.5) * px).astype(np.float32)
    X, Y = np.meshgrid(x, y)
    # Druckkoordinaten: Drehung um die Kartenmitte und Versatz
    if st_.rot_deg or st_.offset != (0.0, 0.0):
        r = np.radians(st_.rot_deg)
        Xc, Yc = X - CARD_W / 2 - st_.offset[0], Y - CARD_H / 2 - st_.offset[1]
        Xd = np.cos(r) * Xc + np.sin(r) * Yc + CARD_W / 2
        Yd = -np.sin(r) * Xc + np.cos(r) * Yc + CARD_H / 2
    else:
        Xd, Yd = X, Y

    sc = st_.symbol_scale
    w = g["stroke_width"] + st_.stroke_delta
    st = g["stripes"]
    T = st["period"] * st_.stripe_period_scale
    lw = max(st["line_width"] + st_.stripe_width_delta, 0.05)
    phase = st["phase_by_shape"][card.shape] * T + st_.stripe_phase_delta
    d = _outset(g, card.fill) + (st_.outset_delta if card.fill != "voll" else 0.0)

    cx0, cy0 = g["layout"]["center_x"], g["layout"]["center_y"]
    alpha = np.zeros((H, W), np.float32)
    for cx, cy in symbol_centers(g, card.count, card.shape):
        cy = cy0 + (cy - cy0) * st_.pitch_scale
        # nur das Umfeld des Symbols auswerten (großzügig wegen Versatz/Drehung)
        y0, y1 = max(int((cy - 16) / px), 0), min(int((cy + 16) / px) + 1, H)
        x0, x1 = max(int((cx - 28) / px), 0), min(int((cx + 28) / px) + 1, W)
        xs, ys = Xd[y0:y1, x0:x1] - cx, Yd[y0:y1, x0:x1] - cy
        sdf = shape_sdf(spec, xs / sc, ys / sc) * sc
        outer = _cov(sdf - d, px)
        if card.fill == "voll":
            a = outer
        else:
            inner = _cov(sdf - d + w, px)
            a = outer - inner
            if card.fill == "gestreift":
                a = a + inner * _stripe_cov(xs - phase, T, lw, px)
        alpha[y0:y1, x0:x1] = np.maximum(alpha[y0:y1, x0:x1], a)
    if st_.print_blur_mm * px_per_mm > 0.3:
        alpha = cv2.GaussianBlur(alpha, (0, 0), st_.print_blur_mm * px_per_mm)

    col = g["colors"]
    paper_lab = np.array(st_.paper_lab if st_.paper_lab is not None else col["paper"]["lab"])
    paper = np.clip(lab_to_lin_rgb(paper_lab), 0, 1).astype(np.float32)
    line = card.fill != "voll"
    base = np.array(col[card.color]["line_lab"] if line and "line_lab" in col[card.color] else col[card.color]["lab"])
    ink_lab = base + np.array(st_.line_dlab if line else st_.ink_dlab)
    ink = np.clip(lab_to_lin_rgb(ink_lab), 0, 1).astype(np.float32)
    rgb = paper * (1 - alpha[..., None]) + ink * alpha[..., None]
    card_a = _cov(sdf_rounded_rect(X - CARD_W / 2, Y - CARD_H / 2, CARD_W / 2, CARD_H / 2, CARD_R), px)
    return rgb.astype(np.float32), card_a.astype(np.float32)


def to_srgba8(rgb_lin: np.ndarray, a: np.ndarray) -> np.ndarray:
    srgb = lin_to_srgb(rgb_lin)
    return (np.concatenate([srgb, a[..., None]], -1) * 255 + 0.5).astype(np.uint8)
