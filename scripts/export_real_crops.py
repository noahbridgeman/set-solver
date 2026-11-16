"""Echte Karten-Crops (data/crops) im Klassifikator-Format exportieren, als Sim-to-Real-Validierung.

Die Crops sind weißabgeglichen; zusätzlich wird eine Variante ohne Weißabgleich aus den
Rohfotos erzeugt (Kunstlicht-Farbstich, wie die Kamera es sieht). Beide Varianten dienen
NUR zur Validierung, nicht zum Training.

Ausgabe: data/real/classify/{wb,raw}/*.jpg + data/real/classify/{wb,raw}.csv
Aufruf:  python scripts/export_real_crops.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from setcards.common import CROPS, RAW, ROOT  # noqa: E402
from setcards.synth import CROP_H, CROP_W  # noqa: E402

OUT = ROOT / "data" / "real" / "classify"


def main() -> None:
    labels = json.loads((CROPS / "labels.json").read_text())
    raw_cache: dict[str, np.ndarray] = {}
    for variant in ("wb", "raw"):
        (OUT / variant).mkdir(parents=True, exist_ok=True)
        rows = []
        for key, m in labels.items():
            stem = key.replace("/", "_")
            if variant == "wb":
                im = cv2.imread(str(CROPS / f"{key}.png"))
                im = cv2.resize(im, (CROP_W, CROP_H), interpolation=cv2.INTER_AREA)
            else:
                src = m["source"]
                if src not in raw_cache:
                    raw_cache[src] = cv2.imread(str(RAW / src))
                c = np.array(m["corners"], np.float64)
                # wie crop_cards.warp_card: kurze Kante oben, Hochformat
                if np.linalg.norm(c[1] - c[0]) > np.linalg.norm(c[2] - c[1]):
                    c = np.roll(c, -1, axis=0)
                w2, h2 = CROP_W * 2, CROP_H * 2
                dst = np.array([[0, 0], [w2, 0], [w2, h2], [0, h2]], np.float32) - 0.5
                M = cv2.getPerspectiveTransform(c.astype(np.float32), dst)
                big = cv2.warpPerspective(raw_cache[src], M, (w2, h2), flags=cv2.INTER_AREA)
                im = cv2.resize(big, (CROP_W, CROP_H), interpolation=cv2.INTER_AREA)
            fn = f"{stem}.jpg"
            cv2.imwrite(str(OUT / variant / fn), im, [cv2.IMWRITE_JPEG_QUALITY, 95])
            rows.append([fn, m["color"], m["shape"], m["fill"], m["count"]])
        with open(OUT / f"{variant}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["file", "color", "shape", "fill", "count"])
            w.writerows(rows)
        print(f"{variant}: {len(rows)} Crops -> {OUT / variant}")


if __name__ == "__main__":
    main()
