"""Labels eines synthetischen Datensatzes sichtbar machen (zur Kontrolle).

- preview/detect_*.jpg: Bilder mit Box, Ecken (k0 rot, k1 grün, k2 blau, k3 gelb; hohl = verdeckt)
- preview/crops.jpg:    zufällige Klassifikator-Crops mit Label, daneben echte Crops zum Vergleich

Aufruf: python scripts/preview_synth.py [--dir data/synth/v1] [--n 12]
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from setcards.common import CROPS, ROOT  # noqa: E402

COL = [(0, 0, 255), (0, 200, 0), (255, 0, 0), (0, 220, 255)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "data" / "synth" / "v1"))
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    d = Path(a.dir)
    pv = d / "preview"
    pv.mkdir(exist_ok=True)
    rng = np.random.default_rng(a.seed)

    imgs = sorted((d / "detect" / "images" / "train").glob("*.jpg"))
    for p in rng.choice(imgs, min(a.n, len(imgs)), replace=False):
        im = cv2.imread(str(p))
        H, W = im.shape[:2]
        lab = d / "detect" / "labels" / "train" / (p.stem + ".txt")
        for line in lab.read_text().split("\n"):
            if not line.strip():
                continue
            v = np.array(line.split(), float)
            cx, cy, bw, bh = v[1:5] * [W, H, W, H]
            cv2.rectangle(im, (int(cx - bw / 2), int(cy - bh / 2)), (int(cx + bw / 2), int(cy + bh / 2)),
                          (255, 255, 255), 1)
            k = v[5:].reshape(4, 3)
            for j, (x, y, vis) in enumerate(k):
                if vis > 0:
                    cv2.circle(im, (int(x * W), int(y * H)), 5, COL[j], -1 if vis == 2 else 2, cv2.LINE_AA)
        cv2.imwrite(str(pv / f"detect_{p.stem}.jpg"), im)

    rows = list(csv.DictReader(open(d / "classify" / "train.csv")))
    pick = [rows[i] for i in rng.choice(len(rows), min(40, len(rows)), replace=False)]
    tiles = []
    for r in pick:
        im = cv2.imread(str(d / "classify" / "train" / r["file"]))
        im = cv2.copyMakeBorder(im, 0, 26, 2, 2, cv2.BORDER_CONSTANT, value=(30, 30, 30))
        txt = f"{r['color'][:2]} {r['shape'][:2]} {r['fill'][:2]} {r['count']}"
        cv2.putText(im, txt, (4, im.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(im)
    real = sorted(CROPS.glob("*.png"))
    for p in rng.choice(real, 10, replace=False):
        im = cv2.resize(cv2.imread(str(p)), (128, 192), interpolation=cv2.INTER_AREA)
        im = cv2.copyMakeBorder(im, 0, 26, 2, 2, cv2.BORDER_CONSTANT, value=(0, 90, 0))
        cv2.putText(im, "ECHT", (4, im.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(im)
    sheet = np.vstack([np.hstack(tiles[i:i + 10]) for i in range(0, len(tiles) - len(tiles) % 10, 10)])
    cv2.imwrite(str(pv / "crops.jpg"), sheet)
    print(f"Vorschau -> {pv}")


if __name__ == "__main__":
    main()
