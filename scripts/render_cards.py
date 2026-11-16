"""Schritt 3: Alle 81 Karten als SVG und PNG rendern.

Aufruf:
  python scripts/render_cards.py                      # SVG + PNG mit 20 px/mm (1200 x 1840)
  python scripts/render_cards.py --px-per-mm 5        # kleine PNGs
  python scripts/render_cards.py --height 640 --png-only
Ausgabe: data/templates/svg/<name>.svg, data/templates/png/<name>.png (RGBA, sRGB)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from setcards.common import CARD_H, TEMPLATES  # noqa: E402
from setcards.render import all_cards, render_raster, render_svg, to_srgba8  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--px-per-mm", type=float, default=20.0)
    ap.add_argument("--height", type=int, help="Kartenhöhe in px (überschreibt --px-per-mm)")
    ap.add_argument("--out", type=Path, default=TEMPLATES)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--svg-only", action="store_true")
    g.add_argument("--png-only", action="store_true")
    args = ap.parse_args()
    ppmm = args.height / CARD_H if args.height else args.px_per_mm

    cards = all_cards()
    if not args.png_only:
        d = args.out / "svg"
        d.mkdir(parents=True, exist_ok=True)
        for c in cards:
            (d / f"{c.name}.svg").write_text(render_svg(c), encoding="utf-8")
        print(f"{len(cards)} SVG -> {d}")
    if not args.svg_only:
        d = args.out / "png"
        d.mkdir(parents=True, exist_ok=True)
        thumbs = []
        for c in cards:
            rgb, a = render_raster(c, ppmm)
            img = to_srgba8(rgb, a)
            cv2.imwrite(str(d / f"{c.name}.png"), img[..., [2, 1, 0, 3]])
            thumbs.append(cv2.resize(img, (120, 184), interpolation=cv2.INTER_AREA))
        h, w = img.shape[:2]
        print(f"{len(cards)} PNG ({w}x{h}, {ppmm:.2f} px/mm) -> {d}")
        # Übersicht 9 x 9 auf grauem Grund
        bg = np.full((184 + 8, 120 + 8, 3), 60, np.uint8)
        tiles = []
        for t in thumbs:
            tile = bg.copy()
            al = t[..., 3:4] / 255.0
            tile[4:-4, 4:-4] = (t[..., :3] * al + 60 * (1 - al)).astype(np.uint8)
            tiles.append(tile)
        sheet = np.vstack([np.hstack(tiles[i:i + 9]) for i in range(0, 81, 9)])
        cv2.imwrite(str(args.out / "overview.png"), sheet[..., ::-1])
        print(f"Übersicht -> {args.out / 'overview.png'}")


if __name__ == "__main__":
    main()
