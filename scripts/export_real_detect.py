"""Echtes Mini-Testset für den Detektor aus den Rohfotos (Ecken aus crop_cards / labels.json).

17 Fotos, 99 Karten, Draufsicht auf dunklem Grund: nur ein grober Realitätscheck, kein
Ersatz für das Video-Testset. Format wie data/synth/*/detect (YOLO-pose, gleiche Eck-Reihenfolge).

Ausgabe: data/real/detect/{images,labels}/val + data.yaml
Aufruf:  python scripts/export_real_detect.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from setcards.common import CROPS, RAW, ROOT  # noqa: E402
from setcards.synth import order_keypoints  # noqa: E402

OUT = ROOT / "data" / "real" / "detect"
SIZE = 960


def to_card_order(c: np.ndarray) -> np.ndarray:
    """Beliebig startende Ecken im Umlauf -> Startpunkt so, dass (0,1) eine kurze Kante ist."""
    if np.linalg.norm(c[1] - c[0]) > np.linalg.norm(c[2] - c[1]):
        c = np.roll(c, -1, axis=0)
    return c


def main() -> None:
    labels = json.loads((CROPS / "labels.json").read_text())
    by_src = defaultdict(list)
    for m in labels.values():
        by_src[m["source"]].append(np.array(m["corners"], np.float64))
    (OUT / "images" / "val").mkdir(parents=True, exist_ok=True)
    (OUT / "labels" / "val").mkdir(parents=True, exist_ok=True)
    n = 0
    for src, cs in sorted(by_src.items()):
        img = cv2.imread(str(RAW / src))
        H, W = img.shape[:2]
        s = SIZE / max(H, W)
        cv2.imwrite(str(OUT / "images" / "val" / f"{Path(src).stem}.jpg"),
                    cv2.resize(img, (round(W * s), round(H * s)), interpolation=cv2.INTER_AREA),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        lines = []
        for c in cs:
            k = order_keypoints(to_card_order(c))
            x0, y0 = k.min(0)
            x1, y1 = k.max(0)
            vals = [0, (x0 + x1) / 2 / W, (y0 + y1) / 2 / H, (x1 - x0) / W, (y1 - y0) / H]
            for x, y in k:
                vals += [x / W, y / H, 2]
            lines.append(" ".join(str(v) if isinstance(v, int) else f"{v:.6f}" for v in vals))
            n += 1
        (OUT / "labels" / "val" / f"{Path(src).stem}.txt").write_text("\n".join(lines) + "\n")
    (OUT / "data.yaml").write_text(f"path: {OUT.resolve()}\ntrain: images/val\nval: images/val\n"
                                   "kpt_shape: [4, 3]\nflip_idx: [1, 0, 3, 2]\nnames:\n  0: card\n")
    print(f"{len(by_src)} Fotos, {n} Karten -> {OUT}")


if __name__ == "__main__":
    main()
