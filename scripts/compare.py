"""Schritt 4: Render neben Foto-Crop für alle 81 Karten, mit Kennzahlen.

Pro Karte: [Foto-Crop | Render | Überlagerung]. Überlagerung: weiß = beide,
rot = nur Foto, türkis = nur Render (Tintenmasken bei alpha 0,5, Foto um den Druckversatz
verschoben).
Die Crop-Lage (0°/180°) wird automatisch passend zum Render gewählt.

Kennzahlen (data/compare/metrics.csv):
- iou_ink:            Tintenmaske Foto vs. Render, ohne Ausrichtung (streng: 0,5 mm Versatz
                      halbiert bei 1,35 mm Strich die IoU)
- iou_shape_aligned:  gefüllte Außenform nach Ausgleich des Versatzes (misst nur die Form)
- edge_dist_mm:       mittlerer Abstand der Außenkonturen nach Ausrichtung
- shift_x/y_mm:       Versatz des Drucks gegenüber dem Modell-Layout (Schwerpunkt)
- dE_ink:             ΔE76 der Flächenfarbe (nur volle Karten)

Ausgabe: data/compare/cards/<name>.png, data/compare/<farbe>_<form>.png (je 3x3),
         data/compare/overview.png, data/compare/metrics.csv
Aufruf:  python scripts/compare.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from setcards.common import (COLORS, CROP_PX_PER_MM, CROPS, PAPER_TARGET_LIN, ROOT, SHAPES, card_mask,  # noqa: E402
                             lab_to_lin_rgb, lin_rgb_to_lab, lin_to_srgb, read_lin)
from setcards.render import all_cards, load_geometry, render_raster  # noqa: E402

OUT = ROOT / "data" / "compare"
S = CROP_PX_PER_MM


def alpha_of(img: np.ndarray, ink: np.ndarray) -> np.ndarray:
    paper = np.full(3, PAPER_TARGET_LIN, np.float32)
    d = (ink - paper).astype(np.float32)
    return np.clip(((img - paper) @ d) / float(d @ d), 0, 1)


def to8(lin: np.ndarray) -> np.ndarray:
    return (lin_to_srgb(lin) * 255 + 0.5).astype(np.uint8)


def label(img: np.ndarray, text: str) -> np.ndarray:
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 56), (235, 235, 235), -1)
    cv2.putText(img, text, (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (20, 20, 20), 2, cv2.LINE_AA)
    return img


def measure(photo: np.ndarray, a_r: np.ndarray, ink: np.ndarray, inner: np.ndarray) -> dict:
    a_p = alpha_of(photo, ink) * inner
    mp, mr = a_p > 0.5, a_r > 0.5
    iou = float((mp & mr).sum() / max((mp | mr).sum(), 1))
    # Versatz des Drucks: Schwerpunkt der gefüllten Formen (Foto minus Render)
    fp = ndimage.binary_fill_holes(mp)
    fr = ndimage.binary_fill_holes(mr)
    (py, px_), (ry, rx) = ndimage.center_of_mass(fp), ndimage.center_of_mass(fr)
    dx, dy = px_ - rx, py - ry
    M = np.float32([[1, 0, -dx], [0, 1, -dy]])
    fpa = cv2.warpAffine(fp.astype(np.float32), M, fp.shape[::-1], flags=cv2.INTER_LINEAR) > 0.5
    iou_shape = float((fpa & fr).sum() / max((fpa | fr).sum(), 1))
    # mittlerer Abstand der Außenkonturen (symmetrisch), nach Ausrichtung
    ep = fpa ^ ndimage.binary_erosion(fpa)
    er = fr ^ ndimage.binary_erosion(fr)
    edge = 0.5 * (ndimage.distance_transform_edt(~er)[ep].mean() +
                  ndimage.distance_transform_edt(~ep)[er].mean()) / S
    mp_al = cv2.warpAffine(a_p, M, a_p.shape[::-1], flags=cv2.INTER_LINEAR) > 0.5
    return dict(photo=photo, mp=mp_al, mr=mr, iou=iou, iou_shape=iou_shape, edge=float(edge), dx=dx, dy=dy)


def main() -> None:
    g = load_geometry()
    (OUT / "cards").mkdir(parents=True, exist_ok=True)
    inner = card_mask(S, 2.0)
    rows, panels = [], {}
    for card in all_cards():
        photo = read_lin(CROPS / f"{card.name}.png")
        rgb, _ = render_raster(card, S, g)
        col = g["colors"][card.color]
        ink = np.clip(lab_to_lin_rgb(np.array(col["lab"] if card.fill == "voll" else col["line_lab"])), 0, 1)
        a_r = alpha_of(rgb, ink) * inner
        # Lage des Crops wählen (Karten lagen teils um 180° gedreht): die mit der besseren Form
        res = [measure(np.ascontiguousarray(p), a_r, ink, inner) for p in (photo, photo[::-1, ::-1])]
        rot = int(res[1]["edge"] < res[0]["edge"])
        m = res[rot]
        photo, mp, mr, iou, iou_shape, edge, dx, dy = (m[k] for k in
                                                       ("photo", "mp", "mr", "iou", "iou_shape", "edge", "dx", "dy"))
        dE = ""
        if card.fill == "voll":
            core = ndimage.binary_erosion(mr, iterations=int(S))
            dE = round(float(np.linalg.norm(lin_rgb_to_lab(np.median(photo[core], 0)) -
                                            np.array(g["colors"][card.color]["lab"]))), 2)
        rows.append(dict(card=card.name, rotated_180=rot, iou_ink=round(iou, 4), iou_shape_aligned=round(iou_shape, 4),
                         edge_dist_mm=round(float(edge), 3), shift_x_mm=round(dx / S, 3), shift_y_mm=round(dy / S, 3),
                         dE_ink=dE))

        # Bild: Foto | Render | Überlagerung
        ov = np.zeros(photo.shape, np.uint8)
        ov[..., 0] = (mp * 255).astype(np.uint8)
        ov[..., 1] = ov[..., 2] = (mr * 255).astype(np.uint8)
        ov = (0.25 * to8(photo) + 0.75 * ov).astype(np.uint8)
        gap = np.full((photo.shape[0], 8, 3), 40, np.uint8)
        panel = np.hstack([label(to8(photo), "Foto"), gap, label(to8(rgb), "Render"), gap,
                           label(ov, f"ausgerichtet, Kante {edge:.2f} mm")])
        cv2.imwrite(str(OUT / "cards" / f"{card.name}.png"), panel[..., ::-1])
        panels[card.name] = panel
        print(f"{card.name:24s} IoU Tinte {iou:.3f}  Form ausgerichtet {iou_shape:.3f}  Kante {edge:.3f} mm  "
              f"Versatz ({dx / S:+.2f}, {dy / S:+.2f}) mm  {'ΔE ' + str(dE) if dE != '' else ''}")

    # Blätter je Farbe+Form (Zeile = Füllung, Spalte = Anzahl, wie die Fotos)
    for c in COLORS:
        for sh in SHAPES:
            names = [n for n in panels if n.startswith(f"{c}_{sh}_")]
            small = [cv2.resize(panels[n], None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA) for n in names]
            pad = lambda im: cv2.copyMakeBorder(im, 6, 6, 6, 6, cv2.BORDER_CONSTANT, value=(25, 25, 25))  # noqa: E731
            sheet = np.vstack([np.hstack([pad(t) for t in small[i:i + 3]]) for i in range(0, 9, 3)])
            cv2.imwrite(str(OUT / f"{c}_{sh}.png"), sheet[..., ::-1])
    # Gesamtübersicht: 81 Paare (Foto | Render), 9 x 9
    W = panels[next(iter(panels))].shape[1]
    w2 = (W - 16) // 3 * 2 + 8
    tiles = [cv2.copyMakeBorder(cv2.resize(p[:, :w2], (2 * 90 + 6, 138), interpolation=cv2.INTER_AREA),
                                3, 3, 3, 3, cv2.BORDER_CONSTANT, value=(25, 25, 25)) for p in panels.values()]
    sheet = np.vstack([np.hstack(tiles[i:i + 9]) for i in range(0, 81, 9)])
    cv2.imwrite(str(OUT / "overview.png"), sheet[..., ::-1])

    with open(OUT / "metrics.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    iou = np.array([r["iou_ink"] for r in rows])
    iou_s = np.array([r["iou_shape_aligned"] for r in rows])
    edge = np.array([r["edge_dist_mm"] for r in rows])
    sh = np.array([[r["shift_x_mm"], r["shift_y_mm"]] for r in rows])
    de = np.array([r["dE_ink"] for r in rows if r["dE_ink"] != ""])
    print(f"\nIoU Tinte (ohne Ausrichtung) median {np.median(iou):.3f} (min {iou.min():.3f})")
    print(f"IoU Form ausgerichtet median {np.median(iou_s):.4f} (min {iou_s.min():.4f}); "
          f"Konturabstand median {np.median(edge):.3f} mm (max {edge.max():.3f} mm)")
    print(f"Versatz |d| median {np.median(np.linalg.norm(sh, axis=1)):.2f} mm, max {np.linalg.norm(sh, axis=1).max():.2f} mm")
    print(f"ΔE Tinte (volle Karten) median {np.median(de):.1f}, max {de.max():.1f}")
    for r in sorted(rows, key=lambda r: -r["edge_dist_mm"])[:5]:
        print("  schwächste:", r)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
