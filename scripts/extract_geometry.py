"""Schritt 2: Symbolgeometrie, Strichstärke, Streifen, Layout und Farben aus den Crops messen.

Vorgehen
- Pro Karte wird ein weicher Tintenanteil alpha in [0,1] bestimmt: lineares
  Entmischen zwischen Papierweiß und der Tintenfarbe der Karte. Damit sind
  Kanten (alpha = 0,5) und Flächenintegrale (Strichstärke, Streifenbreite)
  unabhängig von Unschärfe.
- Symbol = Zusammenhangskomponente der Tinte mit gefüllten Löchern.
  Gemessen wird die Außenkontur, das Innere von leer/gestreift zählt als 1.
- Alle Instanzen einer Form (54 pro Form) werden auf den Schwerpunkt
  ausgerichtet, gemittelt und symmetrisiert (Raute/Oval: Spiegelung an
  beiden Achsen, Welle: Punktsymmetrie). Die Kontur wird dann geglättet:
  Raute = Rhombus mit gerundeten Ecken, Oval = Stadion, Welle = Fourier-Tiefpass.
- Strichstärke w aus leeren Symbolen: Ringfläche A = P*w - pi*w^2 (Steiner).
- Streifen: Periodogramm des Innenbereichs der gestreiften Symbole liefert
  Periode und Phase; Breite = mittlerer Tintenanteil * Periode.

Ausgabe: data/geometry.json (+ Diagnosebilder in data/geometry_debug/)
Aufruf:  python scripts/extract_geometry.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage, optimize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from setcards.common import (CARD_H, CARD_R, CARD_W, COLORS, COUNTS, CROP_PX_PER_MM, CROPS, FILLS,  # noqa: E402
                             GEOMETRY_JSON, PAPER_TARGET_LIN, ROOT, SHAPES, card_mask, lin_rgb_to_lab,
                             read_lin)
from setcards.shapes import sdf_rhombus, sdf_stadium  # noqa: E402

S = CROP_PX_PER_MM
UP = 4                     # Überabtastung für die gemittelte Symbolmaske
TPL_W, TPL_H = 44.0, 24.0  # mm, Fenster um ein Symbol
DEBUG = ROOT / "data" / "geometry_debug"


# ---------------------------------------------------------------- pro Karte
def ink_alpha(card: np.ndarray, ink_rgb: np.ndarray) -> np.ndarray:
    paper = np.full(3, PAPER_TARGET_LIN, np.float32)
    d = (ink_rgb - paper).astype(np.float32)
    a = ((card - paper) @ d) / float(d @ d)
    return np.clip(a, 0, 1.2)


def rough_ink(card: np.ndarray) -> np.ndarray:
    lab = lin_rgb_to_lab(card)
    paper = lin_rgb_to_lab(np.full(3, PAPER_TARGET_LIN))
    de = np.linalg.norm(lab - paper, axis=-1)
    return (de > 25) & card_mask(S, 2.5)


def symbols(card: np.ndarray, ink_rgb: np.ndarray, count: int):
    """Liefert (alpha, Liste von (Maske gefüllt, Maske Tinte-Komponente)) sortiert von oben nach unten."""
    alpha = ink_alpha(card, ink_rgb)
    ink = rough_ink(card) | ((alpha > 0.35) & card_mask(S, 2.5))
    ink = cv2.morphologyEx(ink.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)).astype(bool)
    filled = ndimage.binary_fill_holes(ink)
    lab, n = ndimage.label(filled)
    sizes = ndimage.sum(filled, lab, range(1, n + 1))
    keep = [i + 1 for i in np.argsort(-sizes)[:count]]
    if len(keep) != count or min(sizes[k - 1] for k in keep) < 0.3 * max(sizes):
        raise RuntimeError(f"{count} Symbole erwartet, Größen: {sorted(sizes)[-5:]}")
    regs = [lab == k for k in keep]
    regs.sort(key=lambda m: np.argwhere(m)[:, 0].mean())
    return alpha, regs


def shape_alpha(alpha: np.ndarray, region: np.ndarray) -> np.ndarray:
    """Weiche Außenform: Inneres = 1, am Rand der weiche Tintenanteil."""
    inner = ndimage.binary_erosion(region, iterations=3)
    soft = np.where(inner, 1.0, np.clip(alpha, 0, 1))
    near = ndimage.binary_dilation(region, iterations=4)
    return np.where(near, soft, 0.0).astype(np.float32)


def centroid(w: np.ndarray) -> tuple[float, float]:
    yy, xx = np.mgrid[0:w.shape[0], 0:w.shape[1]]
    m = w.sum()
    return float((xx * w).sum() / m), float((yy * w).sum() / m)


def principal_angle(w: np.ndarray) -> float:
    yy, xx = np.mgrid[0:w.shape[0], 0:w.shape[1]].astype(np.float64)
    cx, cy = centroid(w)
    m = w.sum()
    mxx = ((xx - cx) ** 2 * w).sum() / m
    myy = ((yy - cy) ** 2 * w).sum() / m
    mxy = ((xx - cx) * (yy - cy) * w).sum() / m
    return 0.5 * np.degrees(np.arctan2(2 * mxy, mxx - myy))


def to_template(w: np.ndarray, cx: float, cy: float, angle_deg: float) -> np.ndarray:
    """Ausschnitt um (cx,cy) in Pixeln, gedreht um -angle, auf UP-faches Raster mm-zentriert."""
    s2 = S * UP
    W, H = round(TPL_W * s2), round(TPL_H * s2)
    # Pixelmittelpunkte: Quellpixel i liegt bei i+0.5. Drehen um (cx,cy), dann skalieren,
    # sodass der Schwerpunkt auf der Fenstermitte (W/2, H/2 in Kantenkoordinaten) landet.
    R = np.vstack([cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0), [0, 0, 1]])
    # Zielindex j <-> mm (j + 0.5 - W/2) / (S*UP)  ==  (i - cx) / S
    T = np.array([[UP, 0, W / 2 - 0.5 - UP * cx],
                  [0, UP, H / 2 - 0.5 - UP * cy], [0, 0, 1]], np.float64)
    A = (T @ R)[:2]
    return cv2.warpAffine(w, A, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)


def tpl_coords():
    s2 = S * UP
    W, H = round(TPL_W * s2), round(TPL_H * s2)
    x = (np.arange(W) + 0.5 - W / 2) / s2
    y = (np.arange(H) + 0.5 - H / 2) / s2
    return np.meshgrid(x, y)


# ---------------------------------------------------------------- Formmodelle (SDF, mm)
def fit_model(target: np.ndarray, sdf, p0, X, Y):
    """Parameter so wählen, dass die weiche Modellmaske die gemittelte Maske bestmöglich trifft."""
    px = 1 / (S * UP)
    # nur ein Band um die Kante auswerten (innen/außen ist es konstant)
    edge = (target > 0.02) & (target < 0.98)
    band = ndimage.binary_dilation(edge, iterations=int(0.6 * S * UP))
    Xb, Yb, tb = X[band], Y[band], target[band]

    def loss(p):
        m = np.clip(0.5 - sdf(Xb, Yb, *p) / px, 0, 1)
        return float(((m - tb) ** 2).sum())
    res = optimize.minimize(loss, p0, method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-4, "maxiter": 2000})
    m = np.clip(0.5 - sdf(X, Y, *res.x) / px, 0, 1)
    iou = float(np.minimum(m, target).sum() / np.maximum(m, target).sum())
    return res.x, iou


def fourier_contour(mask: np.ndarray, n_harm: int, n_pts: int = 720, point_sym: bool = False) -> np.ndarray:
    """Außenkontur (mm) nach Bogenlänge resamplen und per Fourier tiefpassfiltern."""
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
    s2 = S * UP
    H, W = mask.shape
    c = np.stack([(c[:, 0] + 0.5 - W / 2) / s2, (c[:, 1] + 0.5 - H / 2) / s2], 1)
    seg = np.linalg.norm(np.diff(np.vstack([c, c[:1]]), axis=0), axis=1)
    t = np.concatenate([[0], np.cumsum(seg)])
    tt = np.linspace(0, t[-1], n_pts, endpoint=False)
    cc = np.vstack([c, c[:1]])
    z = np.interp(tt, t, cc[:, 0]) + 1j * np.interp(tt, t, cc[:, 1])
    F = np.fft.fft(z)
    k = np.fft.fftfreq(n_pts, 1 / n_pts)
    F[np.abs(k) > n_harm] = 0
    if point_sym:  # z(t + T/2) = -z(t)  <=>  nur ungerade Harmonische
        F[(k.astype(int) % 2) == 0] = 0
    z = np.fft.ifft(F)
    pts = np.stack([z.real, z.imag], 1)
    # Startpunkt: ganz links, Orientierung im Uhrzeigersinn (Bildkoordinaten)
    pts = np.roll(pts, -int(np.argmin(pts[:, 0])), axis=0)
    area = 0.5 * np.sum(pts[:, 0] * np.roll(pts[:, 1], -1) - np.roll(pts[:, 0], -1) * pts[:, 1])
    if area < 0:
        pts = np.vstack([pts[:1], pts[1:][::-1]])
    return pts


def sdf_from_mask(mask: np.ndarray) -> np.ndarray:
    """Vorzeichenbehaftete Distanz (mm, innen negativ) auf dem Template-Raster."""
    m = mask.astype(np.uint8)
    out = cv2.distanceTransform(1 - m, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    inn = cv2.distanceTransform(m, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return ((out - inn) + np.where(m > 0, 0.5, -0.5)) / (S * UP)


def poly_mask(pts_mm: np.ndarray, shape) -> np.ndarray:
    s2 = S * UP
    H, W = shape
    p = np.stack([pts_mm[:, 0] * s2 + W / 2 - 0.5, pts_mm[:, 1] * s2 + H / 2 - 0.5], 1)
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [np.round(p * 16).astype(np.int32)], 1, cv2.LINE_8, shift=4)
    return m.astype(np.float32)


# ---------------------------------------------------------------- Hauptprogramm
def main() -> None:
    DEBUG.mkdir(parents=True, exist_ok=True)
    cards = {}
    for c in COLORS:
        for sh in SHAPES:
            for f in FILLS:
                for n in COUNTS:
                    name = f"{c}_{sh}_{f}_{n}"
                    cards[name] = (c, sh, f, n, read_lin(CROPS / f"{name}.png"))

    # 1) Tintenfarben aus vollen Symbolen (Inneres, 1 mm erodiert)
    ink_samples = defaultdict(list)
    for name, (c, sh, f, n, img) in cards.items():
        if f != "voll":
            continue
        filled = ndimage.binary_fill_holes(rough_ink(img))
        core = ndimage.binary_erosion(filled, iterations=int(1.0 * S))
        ink_samples[c].append(np.median(img[core], axis=0))
    ink_rgb = {c: np.median(np.array(v), 0) for c, v in ink_samples.items()}
    ink_lab_per_card = {c: lin_rgb_to_lab(np.array(v)) for c, v in ink_samples.items()}
    # Linienfarbe (Konturen, Streifen) ist dunkler gedruckt als die Flächen: aus dem Kern der
    # Konturen leerer Symbole (>= 0,25 mm von der Strichkante entfernt)
    line_samples = defaultdict(list)
    for name, (c, sh, f, n, img) in cards.items():
        if f != "leer":
            continue
        core = ndimage.binary_erosion(rough_ink(img), iterations=3)
        line_samples[c].append(np.median(img[core], axis=0))
    line_rgb = {c: np.median(np.array(v), 0) for c, v in line_samples.items()}

    # 2) Symbole segmentieren
    inst = defaultdict(list)   # shape -> Liste dicts
    for name, (c, sh, f, n, img) in cards.items():
        alpha, regs = symbols(img, ink_rgb[c] if f == "voll" else line_rgb[c], n)
        for i, reg in enumerate(regs):
            w = shape_alpha(alpha, reg)
            cx, cy = centroid(w)
            inst[sh].append(dict(card=name, color=c, fill=f, count=n, idx=i, w=w, alpha=alpha, region=reg,
                                 cx=cx, cy=cy, ang=principal_angle(w), area=float(w.sum()) / S ** 2))

    X, Y = tpl_coords()
    result: dict = {"units": "mm", "card": {"width": CARD_W, "height": CARD_H, "corner_radius": CARD_R},
                    "source": "data/crops (81 Karten), gemessen mit scripts/extract_geometry.py"}
    shapes_out = {}
    report = []

    outsets = defaultdict(dict)
    card_flip: dict = {}
    for sh in SHAPES:
        L = inst[sh]
        angs = np.array([d["ang"] for d in L])
        mean_ang = float(np.median(angs))

        # Die Welle ist NICHT punktsymmetrisch und die Karten lagen beim Fotografieren teils um 180°
        # gedreht. Deshalb pro Karte die Lage bestimmen (Vergleich mit einer Referenzinstanz).
        for d in L:
            d["flip"] = 0
        if sh == "welle":
            def iou(p, q):
                return ((p > .5) & (q > .5)).sum() / ((p > .5) | (q > .5)).sum()
            ref = next(to_template(d["w"], d["cx"], d["cy"], d["ang"] - mean_ang) for d in L if d["fill"] == "voll")
            votes = defaultdict(float)
            for d in L:
                t = to_template(d["w"], d["cx"], d["cy"], d["ang"] - mean_ang)
                votes[d["card"]] += iou(t[::-1, ::-1], ref) - iou(t, ref)
            for d in L:
                d["flip"] = int(votes[d["card"]] > 0)
            card_flip[sh] = {k: int(v > 0) for k, v in votes.items()}

        def tpl(d):
            return to_template(d["w"], d["cx"], d["cy"], d["ang"] - mean_ang + 180 * d["flip"])

        def symmetrize(avg):
            if sh == "welle":
                return avg  # keine Symmetrie (gemessen: siehe report "welle Symmetrie")
            return 0.25 * (avg + avg[::-1] + avg[:, ::-1] + avg[::-1, ::-1])
        per_fill = {f: symmetrize(np.mean([tpl(d) for d in L if d["fill"] == f], 0)) for f in FILLS}
        if sh == "welle":
            v = per_fill["voll"]
            sym_iou = {k: round(float(((v > .5) & (w_ > .5)).sum() / ((v > .5) | (w_ > .5)).sum()), 3)
                       for k, w_ in {"punkt_180": v[::-1, ::-1], "spiegel_x": v[:, ::-1], "spiegel_y": v[::-1]}.items()}
            report.append(f"welle Symmetrie (IoU mit gespiegelter/gedrehter Mittelform): {sym_iou}")
        # Grundform aus den vollen Symbolen; leer/gestreift sind außen messbar größer (siehe outline_outset)
        sym = per_fill["voll"]
        cv2.imwrite(str(DEBUG / f"{sh}_mean.png"), (sym * 255).astype(np.uint8))
        mask = sym > 0.5
        areas = {f: float(np.mean([d["area"] for d in L if d["fill"] == f])) for f in FILLS}

        if sh == "raute":
            ys, xs = np.nonzero(mask)
            p0 = [np.abs(X[mask]).max(), np.abs(Y[mask]).max(), 0.3]
            p, iou = fit_model(sym, sdf_rhombus, p0, X, Y)
            a, b, r = map(float, p)
            model = {"type": "rhombus", "half_width": a, "half_height": b, "corner_radius": max(r, 0.0)}
        elif sh == "oval":
            p0 = [np.abs(X[mask]).max(), np.abs(Y[mask]).max()]
            p, iou = fit_model(sym, sdf_stadium, p0, X, Y)
            a, b = map(float, p)
            model = {"type": "stadium", "half_width": a, "half_height": b}
        else:
            best = None
            for nh in (8, 10, 12, 15, 18, 22, 26, 30):
                pts = fourier_contour(mask, nh, point_sym=False)
                m = poly_mask(pts, mask.shape)
                soft = cv2.GaussianBlur(m, (0, 0), 1.0)
                iou = float(np.minimum(soft, sym).sum() / np.maximum(soft, sym).sum())
                report.append(f"  welle Fourier {nh:2d} Harmonische: IoU {iou:.4f}")
                if best is None or iou > best[1] + 5e-4:
                    best = (pts, iou, nh)
            pts, iou, nh = best
            # auf 180 Punkte ausdünnen, reicht für eine glatte Catmull-Rom-Kurve
            pts = pts[::4]
            model = {"type": "fourier_contour", "harmonics": nh, "points": np.round(pts, 4).tolist()}
        # Modell-SDF auf dem Raster -> Außenversatz je Füllung fitten und Passung messen
        if sh == "raute":
            sdf_m = sdf_rhombus(X, Y, a, b, max(r, 0.0))
        elif sh == "oval":
            sdf_m = sdf_stadium(X, Y, a, b)
        else:
            sdf_m = sdf_from_mask(poly_mask(pts, X.shape) > 0.5)
        px = 1 / (S * UP)
        fit_q = {}
        for f in FILLS:
            t = per_fill[f]
            band = ndimage.binary_dilation((t > 0.02) & (t < 0.98), iterations=int(0.6 * S * UP))
            res = optimize.minimize_scalar(
                lambda dd: float(((np.clip(0.5 - (sdf_m[band] - dd) / px, 0, 1) - t[band]) ** 2).sum()),
                bounds=(-1, 1), method="bounded")
            dd = float(res.x) if f != "voll" else 0.0
            outsets[f][sh] = float(res.x)
            mb, tb = (sdf_m - dd) < 0, t > 0.5
            fit_q[f] = {"outset_mm": round(float(res.x), 3), "iou": round(float((mb & tb).sum() / (mb | tb).sum()), 4),
                        "edge_rms_mm": round(float(np.sqrt(np.mean(((sdf_m - dd)[np.abs(t - 0.5) < 0.25]) ** 2))), 3)}
        model["rotation_deg"] = 0.0
        model["measured_principal_axis_deg"] = mean_ang
        model["fit"] = fit_q
        model["mean_area_mm2_by_fill"] = {k: round(v, 2) for k, v in areas.items()}
        shapes_out[sh] = model
        report.append(f"{sh}: {json.dumps({k: v for k, v in model.items() if k != 'points'})}")

        # Diagnose: gemittelte Maske mit Modellkontur
        dbg = cv2.cvtColor((sym * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        s2 = S * UP
        if model["type"] == "fourier_contour":
            mm = np.array(model["points"])
        else:
            sdf = (sdf_rhombus(X, Y, a, b, max(r, 0)) if sh == "raute" else sdf_stadium(X, Y, a, b))
            cs, _ = cv2.findContours((sdf < 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            c0 = max(cs, key=cv2.contourArea).reshape(-1, 2)
            mm = np.stack([(c0[:, 0] + 0.5 - X.shape[1] / 2) / s2, (c0[:, 1] + 0.5 - X.shape[0] / 2) / s2], 1)
        pp = np.stack([mm[:, 0] * s2 + X.shape[1] / 2, mm[:, 1] * s2 + X.shape[0] / 2], 1)
        cv2.polylines(dbg, [np.round(pp * 16).astype(np.int32)], True, (0, 0, 255), 1, cv2.LINE_AA, shift=4)
        cv2.imwrite(str(DEBUG / f"{sh}_fit.png"), dbg)

    result["shapes"] = shapes_out
    result["outline_outset"] = {
        f: round(float(np.median(list(outsets[f].values()))), 3) for f in ("gestreift", "leer")}
    result["outline_outset"]["note"] = ("Außenkante leerer/gestreifter Symbole liegt um diesen Betrag (mm) "
                                        "außerhalb der Kontur der vollen Symbole")
    report.append(f"Außenversatz je Füllung und Form: { {f: {k: round(v, 3) for k, v in d.items()} for f, d in outsets.items()} }")

    # 3) Strichstärke aus leeren Symbolen (Steiner: A = P w - pi w^2)
    widths = []
    for sh in SHAPES:
        for d in inst[sh]:
            if d["fill"] != "leer":
                continue
            reg = d["region"]
            band = ndimage.binary_dilation(reg, iterations=4)
            A = float(np.clip(d["alpha"][band], 0, 1).sum()) / S ** 2
            cnts, _ = cv2.findContours(reg.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # Umfang der Außenkante aus der weichen Form (geglättet)
            P = cv2.arcLength(cv2.approxPolyDP(max(cnts, key=cv2.contourArea), 0.6, True), True) / S
            w = (P - np.sqrt(max(P * P - 4 * np.pi * A, 0))) / (2 * np.pi)
            widths.append((sh, d["color"], w))
    wv = np.array([w for *_, w in widths])
    stroke = float(np.median(wv))
    by_shape = {sh: round(float(np.median([w for s, _, w in widths if s == sh])), 3) for sh in SHAPES}
    by_color = {c: round(float(np.median([w for _, cc, w in widths if cc == c])), 3) for c in COLORS}
    result["stroke_width"] = round(stroke, 3)
    report.append(f"Strichstärke: {stroke:.3f} mm (IQR {np.percentile(wv, 25):.3f}-{np.percentile(wv, 75):.3f}),"
                  f" je Form {by_shape}, je Farbe {by_color}")

    # 4) Streifen aus gestreiften Symbolen
    periods = np.linspace(0.4, 1.5, 1101)
    angles = np.linspace(-3, 3, 25)
    stripe_meas = []
    for sh in SHAPES:
        for d in inst[sh]:
            if d["fill"] != "gestreift":
                continue
            reg = d["region"]
            interior = ndimage.binary_erosion(reg, iterations=int(round((stroke + 0.6) * S)))
            yy, xx = np.nonzero(interior)
            a = np.clip(d["alpha"][yy, xx], 0, 1)
            x = (xx + 0.5) / S - (d["cx"] + 0.5) / S
            y = (yy + 0.5) / S - (d["cy"] + 0.5) / S
            ac = a - a.mean()
            best = (0, 0, 0)
            binw = 0.01  # mm; Profil entlang u, danach Periodogramm auf den Bins
            for ang in angles:
                u = x * np.cos(np.radians(ang)) + y * np.sin(np.radians(ang))
                ub = np.round(u / binw).astype(int)
                ub -= ub.min()
                prof = np.bincount(ub, ac)
                uc = (np.arange(len(prof)) * binw)
                pw = np.abs(np.exp(-2j * np.pi * np.outer(1 / periods, uc)) @ prof)
                i = int(np.argmax(pw))
                if pw[i] > best[0]:
                    best = (pw[i], periods[i], ang)
            _, T, ang = best
            # Periode verfeinern (Parabel um das Maximum) und Phase exakt aus allen Pixeln
            u = x * np.cos(np.radians(ang)) + y * np.sin(np.radians(ang))
            fine = np.linspace(T - 0.003, T + 0.003, 61)
            pw = np.abs(np.exp(-2j * np.pi * np.outer(1 / fine, u)) @ ac)
            T = float(fine[int(np.argmax(pw))])
            z = (np.exp(-2j * np.pi * u / T) * ac).sum()
            phase_mm = (-np.angle(z) / (2 * np.pi) * T) % T   # Lage eines Streifenzentrums relativ zum Symbolmittelpunkt
            if d["flip"]:  # Karte lag um 180° gedreht -> in kanonische Lage umrechnen
                phase_mm = (-phase_mm) % T
            stripe_meas.append(dict(shape=sh, color=d["color"], period=T, angle=ang, duty=float(a.mean()),
                                    phase=phase_mm, card=d["card"], cx=(d["cx"] + 0.5) / S))
    T = float(np.median([m["period"] for m in stripe_meas]))
    duty = float(np.median([m["duty"] for m in stripe_meas]))
    ang = float(np.median([m["angle"] for m in stripe_meas]))
    # Phase: Lage eines Streifenzentrums relativ zur Symbolmitte, in Perioden. Crops können um 180°
    # gedreht sein (phi -> -phi), daher wird auf [0, 0.5] gefaltet und der Median je Form genommen.
    phase = {}
    for sh in SHAPES:
        ph = np.array([m["phase"] for m in stripe_meas if m["shape"] == sh]) / T % 1
        if sh == "welle":  # Lage bekannt -> vorzeichenrichtige Phase (zirkulärer Mittelwert)
            fold = ph
            phase[sh] = round(float(np.angle(np.exp(2j * np.pi * ph).mean()) / (2 * np.pi) % 1), 3)
        else:
            fold = np.minimum(ph, 1 - ph)
            phase[sh] = round(float(np.median(fold)), 3)
        report.append(f"Streifenphase {sh}: gefaltet {np.round(np.sort(fold), 2).tolist()}")
    result["stripes"] = {"orientation": "vertikal (senkrecht zur Symbol-Längsachse)", "period": round(T, 4),
                         "line_width": round(duty * T, 4), "angle_deg": round(ang, 2),
                         "phase_by_shape": phase,
                         "phase_note": "Streifenmitte bei x = phase*period relativ zur Symbolmitte"}
    report.append(f"Streifen: Periode {T:.4f} mm (IQR {np.percentile([m['period'] for m in stripe_meas], 25):.4f}-"
                  f"{np.percentile([m['period'] for m in stripe_meas], 75):.4f}), Tintenanteil {duty:.3f} -> "
                  f"Linienbreite {duty * T:.3f} mm, Winkel {ang:.2f}°, Phase je Form {phase}")

    # 5) Layout. Abstände innerhalb einer Karte sind unabhängig vom Schnittversatz der Karte und
    #    werden pro Form gemessen (der Druck hält eher konstante Lücken als ein festes Raster).
    #    Die Gruppenmitte streut mit dem Schnitt (~0,5-1 mm) und wird über alle Karten gemittelt.
    by_card = defaultdict(list)
    for sh in SHAPES:
        for d in inst[sh]:
            by_card[d["card"]].append(((d["cx"] + 0.5) / S, (d["cy"] + 0.5) / S, sh, d["count"]))
    pitch = defaultdict(lambda: defaultdict(list))
    centers = defaultdict(list)
    for name, L in by_card.items():
        L.sort(key=lambda t: t[1])
        sh, n = L[0][2], L[0][3]
        ys = [t[1] for t in L]
        centers[sh].append((np.mean([t[0] for t in L]), np.mean(ys)))
        if n == 2:
            pitch[sh]["2"].append(ys[1] - ys[0])
        if n == 3:
            pitch[sh]["3"].append(0.5 * (ys[2] - ys[0]))
    allc = np.array([c for v in centers.values() for c in v])
    xc, yc = float(np.median(allc[:, 0])), float(np.median(allc[:, 1]))
    layout = {"center_x": round(xc, 3), "center_y": round(yc, 3),
              "center_spread_mm": [round(float(v), 3) for v in allc.std(0)],
              "pitch": {sh: {k: round(float(np.median(v)), 3) for k, v in pitch[sh].items()} for sh in SHAPES},
              "pitch_spread_mm": {sh: {k: round(float(np.std(v)), 3) for k, v in pitch[sh].items()} for sh in SHAPES},
              "center_by_shape": {sh: [round(float(v), 3) for v in np.median(np.array(centers[sh]), 0)]
                                  for sh in SHAPES},
              "note": "Symbolmitten: 1 -> (cx, cy); 2 -> cy ± pitch[form]['2']/2; 3 -> cy, cy ± pitch[form]['3']"}
    result["layout"] = layout
    report.append(f"Layout: Mitte ({xc:.2f}, {yc:.2f}) mm, Streuung {layout['center_spread_mm']} mm; "
                  f"Abstände {layout['pitch']}, Streuung {layout['pitch_spread_mm']}; "
                  f"Mitte je Form {layout['center_by_shape']}")

    # 6) Farben
    paper_lab = lin_rgb_to_lab(np.full(3, PAPER_TARGET_LIN))
    colors_out = {"paper": {"lab": np.round(paper_lab, 2).tolist(),
                            "note": "per Weißabgleich festgelegt (Kartenweiß neutral, L*≈95)"}}
    for c in COLORS:
        lab = lin_rgb_to_lab(ink_rgb[c])
        spread = ink_lab_per_card[c]
        de = np.linalg.norm(spread - lab, axis=1)
        srgb = np.clip(np.where(ink_rgb[c] <= 0.0031308, ink_rgb[c] * 12.92,
                                1.055 * np.clip(ink_rgb[c], 0, 1) ** (1 / 2.4) - 0.055), 0, 1)
        line_lab = lin_rgb_to_lab(line_rgb[c])
        colors_out[c] = {"lab": np.round(lab, 2).tolist(),
                         "srgb_hex": "#" + "".join(f"{round(v * 255):02x}" for v in srgb),
                         "line_lab": np.round(line_lab, 2).tolist(),
                         "line_deltaE_vs_fill": round(float(np.linalg.norm(line_lab - lab)), 2),
                         "deltaE_cards_median": round(float(np.median(de)), 2),
                         "deltaE_cards_max": round(float(de.max()), 2)}
        report.append(f"Farbe {c}: Lab {np.round(lab, 1).tolist()} {colors_out[c]['srgb_hex']}, "
                      f"Streuung über Karten ΔE median {np.median(de):.1f}, max {de.max():.1f}; "
                      f"Linie Lab {np.round(line_lab, 1).tolist()} (ΔE {np.linalg.norm(line_lab - lab):.1f} zur Fläche)")
    result["colors"] = colors_out

    GEOMETRY_JSON.write_text(json.dumps(result, indent=1, ensure_ascii=False))
    (DEBUG / "report.txt").write_text("\n".join(report) + "\n")
    (DEBUG / "stripes.json").write_text(json.dumps(stripe_meas, indent=1))
    (DEBUG / "card_flip.json").write_text(json.dumps(card_flip, indent=1))
    print("\n".join(report))
    print(f"\n-> {GEOMETRY_JSON}")


if __name__ == "__main__":
    main()
