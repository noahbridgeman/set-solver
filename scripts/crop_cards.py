"""Schritt 1: Karten aus den Fotos ausschneiden, entzerren, weißabgleichen und labeln.

9er-Fotos (Dateiname = farbe_form): Zeile = Füllung (voll/gestreift/leer),
Spalte = Anzahl (1/2/3). Ausgabe: data/crops/<farbe>_<form>_<fuellung>_<anzahl>.png

Einzelaufnahmen und farben.jpeg landen in data/crops/extra/. Deren Farbe wird
über den Farbton bestimmt, Form/Füllung/Anzahl kommen aus dem Dateinamen.
Zusätzlich wird bei allen Karten die Farbe per Farbton gegengeprüft.

Aufruf:  python scripts/crop_cards.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from setcards.common import (COLORS, CROP_PX_PER_MM, CROPS, FILLS, RAW, SHAPES, card_mask,  # noqa: E402
                             find_cards, lin_rgb_to_lab, read_lin, warp_card, white_balance, write_lin)


def ink_hue_color(card: np.ndarray) -> tuple[str, float]:
    """Farbe der Symbole über den Farbton der stark gesättigten Pixel bestimmen."""
    lab = lin_rgb_to_lab(card)
    inner = card_mask(CROP_PX_PER_MM, 3.0)
    chroma = np.hypot(lab[..., 1], lab[..., 2])
    sel = inner & (chroma > 25)
    if sel.sum() < 50:
        return "?", 0.0
    hue = np.degrees(np.arctan2(lab[..., 2][sel], lab[..., 1][sel])) % 360
    h = float(np.degrees(np.angle(np.exp(1j * np.radians(hue)).mean())) % 360)
    # Lab-Farbtöne: rot ≈ 30-50°, grün ≈ 140-170°, lila ≈ 290-330°
    refs = {"rot": 40.0, "gruen": 155.0, "lila": 305.0}
    best = min(refs, key=lambda k: abs((h - refs[k] + 180) % 360 - 180))
    return best, h


def order_grid(corner_list: list[np.ndarray], rows: int, cols: int) -> list[list[np.ndarray]]:
    centers = np.array([c.mean(0) for c in corner_list])
    order = np.argsort(centers[:, 1])
    grid = []
    for r in range(rows):
        idx = order[r * cols:(r + 1) * cols]
        idx = idx[np.argsort(centers[idx, 0])]
        grid.append([corner_list[i] for i in idx])
    return grid


def process(path: Path, n_expected: int):
    img = read_lin(path)
    corners, rms = find_cards(img, expected=n_expected, return_rms=True)
    print(f"  {path.name}: Kanten-RMS je Karte (px) {np.round(rms, 2).tolist()}")
    return img, corners


def main() -> None:
    CROPS.mkdir(parents=True, exist_ok=True)
    meta: dict[str, dict] = {}
    problems: list[str] = []

    # --- 9er-Fotos
    for color in COLORS:
        for shape in SHAPES:
            path = RAW / f"{color}_{shape}.jpeg"
            img, corners = process(path, 9)
            grid = order_grid(corners, 3, 3)
            for r, fill in enumerate(FILLS):
                for c, count in enumerate((1, 2, 3)):
                    cr = grid[r][c]
                    card = white_balance(warp_card(img, cr))
                    name = f"{color}_{shape}_{fill}_{count}"
                    write_lin(CROPS / f"{name}.png", card)
                    hue_col, hue = ink_hue_color(card)
                    if hue_col != color:
                        problems.append(f"{name}: Farbton {hue:.0f}° sieht nach {hue_col} aus")
                    meta[name] = {"source": path.name, "row": r, "col": c, "corners": cr.round(1).tolist(),
                                  "color": color, "shape": shape, "fill": fill, "count": count,
                                  "hue_deg": round(hue, 1)}
            print(f"{path.name}: 9 Karten")

    # --- Einzelaufnahmen und Farbfoto
    extra = CROPS / "extra"
    extra.mkdir(exist_ok=True)
    singles = {  # dateiname -> (form, füllung, anzahl)
        "einzel_raute_voll": ("raute", "voll", 1),
        "einzel_oval_voll": ("oval", "voll", 1),
        "einzel_welle_voll": ("welle", "voll", 1),
        "einzel_oval_leer": ("oval", "leer", 1),
        "einzel_oval_gestreift": ("oval", "gestreift", 1),
        "einzel_oval_2": ("oval", "voll", 2),
        "einzel_oval_3": ("oval", "voll", 3),
    }
    for stem, (shape, fill, count) in singles.items():
        img, corners = process(RAW / f"{stem}.jpeg", 1)
        card = white_balance(warp_card(img, corners[0]))
        color, hue = ink_hue_color(card)
        name = f"{color}_{shape}_{fill}_{count}"
        write_lin(extra / f"{stem}.png", card)
        meta[f"extra/{stem}"] = {"source": f"{stem}.jpeg", "corners": corners[0].round(1).tolist(), "color": color,
                                 "shape": shape, "fill": fill, "count": count, "hue_deg": round(hue, 1),
                                 "label": name}
        print(f"{stem}.jpeg -> {name}")

    img, corners = process(RAW / "farben.jpeg", 3)
    for i, cr in enumerate(order_grid(corners, 1, 3)[0]):
        card = white_balance(warp_card(img, cr))
        color, hue = ink_hue_color(card)
        stem = f"farben_{i + 1}_{color}"
        write_lin(extra / f"{stem}.png", card)
        meta[f"extra/{stem}"] = {"source": "farben.jpeg", "corners": cr.round(1).tolist(), "color": color,
                                 "shape": "oval", "fill": "voll", "count": 1, "hue_deg": round(hue, 1)}
        print(f"farben.jpeg Karte {i + 1} -> {color} ({hue:.0f}°)")

    (CROPS / "labels.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    print(f"\n{len(meta)} Crops geschrieben nach {CROPS}")
    if problems:
        print("WARNUNG, Farbton passt nicht zum Label:")
        for p in problems:
            print("  " + p)
    else:
        print("Farbton-Prüfung: alle 81 Labels konsistent")


if __name__ == "__main__":
    main()
