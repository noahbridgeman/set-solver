"""Symbolformen: vorzeichenbehaftete Distanzfelder (mm, innen negativ) und SVG-Pfade.

Koordinaten: Symbol-lokal in mm, Ursprung = Symbolmittelpunkt (Schwerpunkt der
vollen Form), x nach rechts, y nach unten (wie Bild/SVG).
"""
from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np


# ---------------------------------------------------------------- Distanzfelder
def sdf_rhombus(x, y, a, b, r=0.0):
    """Rhombus mit Halbachsen a (x) und b (y), Ecken mit Radius r gerundet.

    Die Außenmaße an den Kanten bleiben (a, b); die Ecken werden um r abgerundet.
    """
    n = np.array([b, a]) / np.hypot(a, b)
    r = max(float(r), 0.0)
    ai, bi = a - r / n[0], b - r / n[1]          # innerer Rhombus (Eckmittelpunkte)
    q = np.stack([np.abs(x), np.abs(y)], -1)
    bb = np.array([ai, bi])
    h = np.clip(((bb[0] - 2 * q[..., 0]) * bb[0] - (bb[1] - 2 * q[..., 1]) * bb[1]) / (bb @ bb), -1, 1)
    d = np.linalg.norm(q - 0.5 * bb * np.stack([1 - h, 1 + h], -1), axis=-1)
    sgn = np.sign(q[..., 0] * bi + q[..., 1] * ai - ai * bi)
    return d * sgn - r


def sdf_stadium(x, y, a, b):
    """Stadion mit Halblänge a (gesamt) und Halbhöhe b (Halbkreise mit Radius b)."""
    dx = np.maximum(np.abs(x) - (a - b), 0)
    return np.hypot(dx, y) - b


def sdf_rounded_rect(x, y, hw, hh, r):
    qx = np.abs(x) - (hw - r)
    qy = np.abs(y) - (hh - r)
    return np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0) - r


_POLY_SDF_RES = 40.0  # px/mm des vorberechneten Distanzfelds für Konturformen


@lru_cache(maxsize=8)
def _poly_sdf_grid(pts_key: tuple):
    pts = np.array(pts_key).reshape(-1, 2)
    margin = 4.0
    x0, y0 = pts.min(0) - margin
    x1, y1 = pts.max(0) + margin
    s = _POLY_SDF_RES
    W, H = int(np.ceil((x1 - x0) * s)), int(np.ceil((y1 - y0) * s))
    # Maske 4x überabgetastet, damit die Kante genauer als ein Rasterpunkt liegt
    up = 4
    m = np.zeros((H * up, W * up), np.uint8)
    p = (pts - [x0, y0]) * s * up - 0.5
    cv2.fillPoly(m, [np.round(p * 16).astype(np.int32)], 1, cv2.LINE_8, shift=4)
    out = cv2.distanceTransform(1 - m, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    inn = cv2.distanceTransform(m, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    sdf = ((out - inn) + np.where(m > 0, 0.5, -0.5)) / (s * up)
    sdf = cv2.resize(sdf, (W, H), interpolation=cv2.INTER_AREA)
    return sdf.astype(np.float32), float(x0), float(y0)


def sdf_polygon(x, y, pts):
    """Distanzfeld einer geschlossenen Kontur (über ein vorberechnetes Raster, Auflösung 0,025 mm)."""
    grid, x0, y0 = _poly_sdf_grid(tuple(np.round(np.asarray(pts), 5).ravel()))
    s = _POLY_SDF_RES
    mx = ((np.asarray(x, np.float32) - x0) * s - 0.5).astype(np.float32)
    my = ((np.asarray(y, np.float32) - y0) * s - 0.5).astype(np.float32)
    v = cv2.remap(grid, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=10.0)
    return v


def shape_sdf(spec: dict, x, y):
    t = spec["type"]
    if t == "rhombus":
        return sdf_rhombus(x, y, spec["half_width"], spec["half_height"], spec.get("corner_radius", 0.0))
    if t == "stadium":
        return sdf_stadium(x, y, spec["half_width"], spec["half_height"])
    if t == "fourier_contour":
        return sdf_polygon(x, y, spec["points"])
    raise ValueError(t)


# ---------------------------------------------------------------- SVG-Pfade
def _f(v: float) -> str:
    return f"{v:.4f}".rstrip("0").rstrip(".")


def _offset_closed(pts: np.ndarray, d: float) -> np.ndarray:
    """Geschlossene, glatte Kontur um d nach außen versetzen (Normalen aus zentralen Differenzen)."""
    t = np.roll(pts, -1, 0) - np.roll(pts, 1, 0)
    n = np.stack([t[:, 1], -t[:, 0]], 1)
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    area = 0.5 * np.sum(pts[:, 0] * np.roll(pts[:, 1], -1) - np.roll(pts[:, 0], -1) * pts[:, 1])
    # (dy, -dx) zeigt bei positiver Fläche (y nach unten) nach außen
    return pts + (d if area > 0 else -d) * n


def _catmull_rom_path(pts: np.ndarray, cx: float, cy: float) -> str:
    P = pts + [cx, cy]
    n = len(P)
    out = [f"M{_f(P[0, 0])},{_f(P[0, 1])}"]
    for i in range(n):
        p0, p1, p2, p3 = P[i - 1], P[i], P[(i + 1) % n], P[(i + 2) % n]
        c1 = p1 + (p2 - p0) / 6
        c2 = p2 - (p3 - p1) / 6
        out.append(f"C{_f(c1[0])},{_f(c1[1])} {_f(c2[0])},{_f(c2[1])} {_f(p2[0])},{_f(p2[1])}")
    return " ".join(out) + "Z"


def shape_path(spec: dict, cx: float, cy: float, offset: float = 0.0) -> str:
    """SVG-Pfad der Form um (cx, cy), Außenkante um `offset` mm versetzt."""
    t = spec["type"]
    if t == "stadium":
        a, b = spec["half_width"] + offset, spec["half_height"] + offset
        s = a - b
        return (f"M{_f(cx - s)},{_f(cy - b)} L{_f(cx + s)},{_f(cy - b)} "
                f"A{_f(b)},{_f(b)} 0 0 1 {_f(cx + s)},{_f(cy + b)} L{_f(cx - s)},{_f(cy + b)} "
                f"A{_f(b)},{_f(b)} 0 0 1 {_f(cx - s)},{_f(cy - b)}Z")
    if t == "rhombus":
        a, b, r = spec["half_width"], spec["half_height"], max(spec.get("corner_radius", 0.0), 0.0)
        n0, n1 = np.array([b, a]) / np.hypot(a, b)
        ai, bi = a - r / n0, b - r / n1
        R = r + offset
        V = {"r": (ai, 0.0), "b": (0.0, bi), "l": (-ai, 0.0), "t": (0.0, -bi)}
        N = {"tr": (n0, -n1), "rb": (n0, n1), "bl": (-n0, n1), "lt": (-n0, -n1)}

        def P(v, nn):
            return f"{_f(cx + V[v][0] + R * N[nn][0])},{_f(cy + V[v][1] + R * N[nn][1])}"
        if R <= 1e-6:
            return f"M{P('r', 'tr')} L{P('b', 'rb')} L{P('l', 'bl')} L{P('t', 'lt')}Z"
        A = f"A{_f(R)},{_f(R)} 0 0 1 "
        return (f"M{P('r', 'tr')} {A}{P('r', 'rb')} L{P('b', 'rb')} {A}{P('b', 'bl')} "
                f"L{P('l', 'bl')} {A}{P('l', 'lt')} L{P('t', 'lt')} {A}{P('t', 'tr')}Z")
    if t == "fourier_contour":
        pts = np.asarray(spec["points"], np.float64)
        if offset:
            pts = _offset_closed(pts, offset)
        return _catmull_rom_path(pts, cx, cy)
    raise ValueError(t)
