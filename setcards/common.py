"""Gemeinsame Hilfsfunktionen: Farbräume, Kartendetektion, Entzerrung, Weißabgleich."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
CROPS = ROOT / "data" / "crops"
TEMPLATES = ROOT / "data" / "templates"
GEOMETRY_JSON = ROOT / "data" / "geometry.json"

# Kartenmaß im Hochformat (Breite x Höhe), mm
CARD_W, CARD_H, CARD_R = 60.0, 92.0, 5.0
CROP_PX_PER_MM = 12.0  # Auflösung der entzerrten Crops

COLORS = ["rot", "gruen", "lila"]
SHAPES = ["raute", "oval", "welle"]
FILLS = ["voll", "gestreift", "leer"]
COUNTS = [1, 2, 3]

PAPER_TARGET_LIN = 0.88  # Kartenweiß nach Weißabgleich (linear, ≈ L* 95)


# ---------------------------------------------------------------- Farbräume
def srgb_to_lin(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float32)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def lin_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, np.float32), 0, 1)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)


_M_RGB2XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                       [0.2126729, 0.7151522, 0.0721750],
                       [0.0193339, 0.1191920, 0.9503041]], np.float64)
_WHITE_D65 = np.array([0.95047, 1.0, 1.08883])


def lin_rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Lineares sRGB (…,3, Reihenfolge R,G,B) -> CIELAB (D65)."""
    xyz = np.asarray(rgb, np.float64) @ _M_RGB2XYZ.T / _WHITE_D65
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], -1)


def lab_to_lin_rgb(lab: np.ndarray) -> np.ndarray:
    lab = np.asarray(lab, np.float64)
    fy = (lab[..., 0] + 16) / 116
    fx = fy + lab[..., 1] / 500
    fz = fy - lab[..., 2] / 200
    f = np.stack([fx, fy, fz], -1)
    xyz = np.where(f > 6 / 29, f ** 3, 3 * (6 / 29) ** 2 * (f - 4 / 29)) * _WHITE_D65
    return xyz @ np.linalg.inv(_M_RGB2XYZ).T


def read_lin(path: Path) -> np.ndarray:
    """Bild laden (EXIF-Rotation wird von OpenCV angewandt) -> lineares RGB float32."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return srgb_to_lin(bgr[..., ::-1] / 255.0)


def write_lin(path: Path, rgb_lin: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    srgb = (lin_to_srgb(rgb_lin) * 255 + 0.5).astype(np.uint8)
    cv2.imwrite(str(path), srgb[..., ::-1])


def read_srgb8_as_lin(path: Path) -> np.ndarray:
    return read_lin(path)


# ---------------------------------------------------------------- Detektion
def _line_fit(pts: np.ndarray) -> np.ndarray:
    vx, vy, x0, y0 = cv2.fitLine(pts.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
    # homogene Gerade a*x + b*y + c = 0
    a, b = vy, -vx
    return np.array([a, b, -(a * x0 + b * y0)])


def _refine_corners(contour: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Seiten der Karte als Geraden fitten (ohne die abgerundeten Ecken) und schneiden.

    Ergebnis sind die virtuellen, scharfen Ecken des Kartenrechtecks.
    """
    c = contour.reshape(-1, 2).astype(np.float64)
    lines = []
    for i in range(4):
        p, q = quad[i], quad[(i + 1) % 4]
        d = q - p
        L = np.linalg.norm(d)
        t = ((c - p) @ d) / L ** 2
        n = np.array([-d[1], d[0]]) / L
        dist = np.abs((c - p) @ n)
        sel = (t > 0.15) & (t < 0.85) & (dist < 0.03 * L)
        lines.append(_line_fit(c[sel]))
    corners = []
    for i in range(4):
        x = np.cross(lines[i - 1], lines[i])
        corners.append(x[:2] / x[2])
    return np.array(corners)


def _intersect_lines(lines: list[np.ndarray]) -> np.ndarray:
    out = []
    for i in range(4):
        x = np.cross(lines[i - 1], lines[i])
        out.append(x[:2] / x[2])
    return np.array(out)


def refine_corners_fullres(gray: np.ndarray, corners: np.ndarray, iters: int = 2,
                           half_len: float = 30.0, n_samples: int = 80) -> tuple[np.ndarray, float]:
    """Kartenkanten in voller Auflösung subpixelgenau nachfitten.

    Pro Seite werden Profile senkrecht zur Kante abgetastet, die Kante liegt am
    stärksten Hell->Dunkel-Übergang (nach außen). Die Punkte werden robust
    (iterativ getrimmt) zu einer Geraden gefittet. Gibt Ecken und RMS-Abweichung
    der Kantenpunkte (px) zurück.
    """
    g = cv2.GaussianBlur(gray, (0, 0), 1.2)
    c = np.asarray(corners, np.float64)
    ctr = c.mean(0)
    rms_all = []
    for _ in range(iters):
        lines, rms_all = [], []
        for i in range(4):
            p, q = c[i], c[(i + 1) % 4]
            d = (q - p) / np.linalg.norm(q - p)
            n = np.array([-d[1], d[0]])
            if n @ (ctr - (p + q) / 2) > 0:  # Normale nach außen
                n = -n
            t = np.linspace(0.1, 0.9, n_samples)
            s = np.arange(-half_len, half_len + 0.5, 0.5)
            base = p[None] + t[:, None] * (q - p)[None]
            pts = base[:, None, :] + s[None, :, None] * n[None, None, :]
            prof = cv2.remap(g, pts[..., 0].astype(np.float32), pts[..., 1].astype(np.float32), cv2.INTER_LINEAR)
            der = -np.gradient(prof, axis=1)               # hell innen -> dunkel außen = positiv
            k = np.argmax(der, axis=1)
            k = np.clip(k, 1, len(s) - 2)
            r_ = np.arange(len(t))
            y0, y1, y2 = der[r_, k - 1], der[r_, k], der[r_, k + 1]
            off = 0.5 * (y0 - y2) / np.where(np.abs(y0 - 2 * y1 + y2) < 1e-9, 1e-9, y0 - 2 * y1 + y2)
            pos = s[k] + np.clip(off, -1, 1) * 0.5
            edge = base + pos[:, None] * n[None]
            keep = np.ones(len(t), bool)
            for _k in range(3):  # getrimmter Fit: Ausreißer (Schatten, Reflexe) verwerfen
                line = _line_fit(edge[keep])
                dist = np.abs(edge @ line[:2] + line[2]) / np.hypot(*line[:2])
                thr = max(3 * np.median(dist[keep]), 0.75)
                keep = dist < thr
            lines.append(line)
            rms_all.append(float(np.sqrt(np.mean(dist[keep] ** 2))))
        # Ecken = Schnitt benachbarter Seiten (Seite i läuft von Ecke i nach i+1)
        c = _intersect_lines(lines)
    return c, float(np.max(rms_all))


def find_cards(rgb_lin: np.ndarray, expected: int | None = None, work_px: int = 1600,
               return_rms: bool = False):
    """Karten als helle Vierecke finden; Kanten danach in voller Auflösung verfeinern.

    Gibt eine Liste von 4x2-Eckarrays (Bildkoordinaten, im Uhrzeigersinn) zurück,
    mit return_rms zusätzlich die RMS-Abweichung der Kantenpunkte je Karte (px).
    """
    h, w = rgb_lin.shape[:2]
    s = work_px / max(h, w)
    small = cv2.resize(rgb_lin, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    lum = small @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    # Otsu auf der Helligkeit; Tisch/Filz am Rand werden über Form und Seitenverhältnis aussortiert.
    # (Kein Sättigungskriterium: unter warmem Kunstlicht ist das Kartenweiß selbst deutlich gesättigt.)
    l8 = (np.clip(lin_to_srgb(lum), 0, 1) * 255).astype(np.uint8)
    thr, _ = cv2.threshold(l8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = (l8 > thr).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    cands = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 0.005 * mask.size:
            continue
        (cx, cy), (rw, rh), ang = cv2.minAreaRect(c)
        if rw * rh <= 0:
            continue
        aspect = max(rw, rh) / min(rw, rh)
        fill = area / (rw * rh)
        if not (1.35 < aspect < 1.75 and fill > 0.9):
            continue
        hull = cv2.convexHull(c)
        peri = cv2.arcLength(hull, True)
        quad = None
        for eps in np.linspace(0.01, 0.08, 30):
            ap = cv2.approxPolyDP(hull, eps * peri, True)
            if len(ap) == 4:
                quad = ap.reshape(4, 2).astype(np.float64)
                break
        if quad is None:
            quad = cv2.boxPoints(((cx, cy), (rw, rh), ang)).astype(np.float64)
        # im Uhrzeigersinn ordnen
        ctr = quad.mean(0)
        quad = quad[np.argsort(np.arctan2(quad[:, 1] - ctr[1], quad[:, 0] - ctr[0]))]
        corners = _refine_corners(c, quad) / s
        cands.append((area, corners))

    cands.sort(key=lambda t: -t[0])
    if expected is not None:
        if len(cands) < expected:
            raise RuntimeError(f"nur {len(cands)} statt {expected} Karten gefunden")
        cands = cands[:expected]
    gray = (rgb_lin @ np.array([0.2126, 0.7152, 0.0722], np.float32)).astype(np.float32)
    refined = [refine_corners_fullres(gray, c) for _, c in cands]
    if return_rms:
        return [c for c, _ in refined], [r for _, r in refined]
    return [c for c, _ in refined]


def warp_card(rgb_lin: np.ndarray, corners: np.ndarray, px_per_mm: float = CROP_PX_PER_MM) -> np.ndarray:
    """Karte ins Hochformat (60 x 92 mm) entzerren. Lange Seite wird vertikal."""
    c = np.asarray(corners, np.float64)
    e01 = np.linalg.norm(c[1] - c[0])
    e12 = np.linalg.norm(c[2] - c[1])
    if e01 > e12:  # Kante 0-1 ist lang -> Startecke verschieben, damit 0-1 die kurze Oberkante ist
        c = np.roll(c, -1, axis=0)
    # Startecke = oben links (kleinste x+y unter den beiden Kandidaten mit Kante 0-1 kurz)
    if (c[0] + c[1]).sum() > (c[2] + c[3]).sum():
        c = np.roll(c, 2, axis=0)
    W, H = round(CARD_W * px_per_mm), round(CARD_H * px_per_mm)
    dst = np.array([[0, 0], [W, 0], [W, H], [0, H]], np.float64) - 0.5
    M = cv2.getPerspectiveTransform(c.astype(np.float32), dst.astype(np.float32))
    return cv2.warpPerspective(rgb_lin, M, (W, H), flags=cv2.INTER_AREA if px_per_mm < 8 else cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)


def card_mask(px_per_mm: float, inset_mm: float = 0.0) -> np.ndarray:
    """Maske der abgerundeten Karte (optional um inset_mm verkleinert)."""
    W, H = round(CARD_W * px_per_mm), round(CARD_H * px_per_mm)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    x = (xx + 0.5) / px_per_mm
    y = (yy + 0.5) / px_per_mm
    r = CARD_R
    dx = np.maximum(np.maximum(r + inset_mm - x, x - (CARD_W - r - inset_mm)), 0)
    dy = np.maximum(np.maximum(r + inset_mm - y, y - (CARD_H - r - inset_mm)), 0)
    return (dx ** 2 + dy ** 2 <= r ** 2) & (x > inset_mm) & (x < CARD_W - inset_mm) & \
        (y > inset_mm) & (y < CARD_H - inset_mm)


def white_balance(card_lin: np.ndarray, px_per_mm: float = CROP_PX_PER_MM, iters: int = 3) -> np.ndarray:
    """Weißabgleich + Flatfield über das Kartenweiß.

    Pro Kanal wird ein quadratisches 2D-Polynom an die Papierpixel gefittet und
    das Bild dadurch geteilt; Papier landet dann neutral bei PAPER_TARGET_LIN.
    """
    H, W = card_lin.shape[:2]
    inner = card_mask(px_per_mm, inset_mm=1.0)
    yy, xx = np.mgrid[0:H, 0:W]
    xn, yn = (xx / W - 0.5).ravel(), (yy / H - 0.5).ravel()
    A = np.stack([np.ones_like(xn), xn, yn, xn * xn, xn * yn, yn * yn], 1)

    # Start: Papier = Randstreifen, da sitzen keine Symbole
    paper = inner & ~card_mask(px_per_mm, inset_mm=6.0)
    out = card_lin
    for _ in range(iters):
        sub = paper.ravel()
        step = max(1, sub.sum() // 40000)
        idx = np.flatnonzero(sub)[::step]
        field = np.empty_like(card_lin)
        for ch in range(3):
            coef, *_ = np.linalg.lstsq(A[idx], card_lin.reshape(-1, 3)[idx, ch], rcond=None)
            field[..., ch] = (A @ coef).reshape(H, W)
        out = card_lin / np.maximum(field, 1e-4) * PAPER_TARGET_LIN
        # Papier neu bestimmen: hell und neutral
        lab = lin_rgb_to_lab(out)
        chroma = np.hypot(lab[..., 1], lab[..., 2])
        paper = inner & (chroma < 8) & (np.abs(lab[..., 0] - lin_rgb_to_lab(np.full(3, PAPER_TARGET_LIN))[0]) < 8)
    return out.astype(np.float32)
