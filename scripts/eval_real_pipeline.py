"""End-to-End-Test auf den echten Rohfotos: Detektor -> Entzerren (volle Auflösung) -> Klassifikator.

Misst pro Karte: gefunden? Eckfehler (in % der kurzen Kartenkante), Attribute richtig?
Die echten Karten wurden nie zum Training oder zur Modellauswahl benutzt.

Aufruf:  python scripts/eval_real_pipeline.py [--det models/detector_v1.pt]
                                              [--cls models/classifier_v1.pt]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from setcards.classifier import ATTRS, CardNet, to_tensor  # noqa: E402
from setcards.common import CROPS, RAW, ROOT  # noqa: E402
from setcards.synth import order_keypoints, warp_crop  # noqa: E402


def to_card_order(c: np.ndarray) -> np.ndarray:
    if np.linalg.norm(c[1] - c[0]) > np.linalg.norm(c[2] - c[1]):
        c = np.roll(c, -1, axis=0)
    return c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", default=str(ROOT / "models/detector_v1.pt"))
    ap.add_argument("--cls", default=str(ROOT / "models/classifier_v1.pt"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.6)
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()

    from ultralytics import YOLO
    det = YOLO(a.det)
    ck = torch.load(a.cls, map_location="cpu", weights_only=False)
    net = CardNet(pretrained=False)
    net.load_state_dict(ck["model"])
    net.eval().to(a.device)

    labels = json.loads((CROPS / "labels.json").read_text())
    by_src = defaultdict(list)
    for name, m in labels.items():
        by_src[m["source"]].append((name, m))

    out_dir = ROOT / "runs" / "pipeline_real"
    out_dir.mkdir(parents=True, exist_ok=True)
    n_gt = n_found = n_fp = n_ok = 0
    errs, wrong = [], []
    for src, cards in sorted(by_src.items()):
        img = cv2.imread(str(RAW / src))
        r = det.predict(img, imgsz=a.imgsz, conf=a.conf, device=a.device, verbose=False)[0]
        kps = r.keypoints.xy.cpu().numpy() if r.keypoints is not None else np.zeros((0, 4, 2))
        used = set()
        vis = img.copy()
        for name, m in cards:
            n_gt += 1
            gt = order_keypoints(to_card_order(np.array(m["corners"], np.float64)))
            short = np.linalg.norm(gt[1] - gt[0])
            d = [np.linalg.norm(k.mean(0) - gt.mean(0)) for k in kps]
            j = int(np.argmin(d)) if d else -1
            if j < 0 or d[j] > 0.3 * short or j in used:
                wrong.append((src, name, "nicht gefunden"))
                continue
            used.add(j)
            n_found += 1
            k = kps[j].astype(np.float64)
            # Eckfehler; Reihenfolge darf um 180° gedreht sein (Klassifikator nutzt TTA 0°/180°)
            e = min(np.linalg.norm(np.roll(k, s, 0) - gt, axis=1).max() for s in (0, 2))
            errs.append(e / short * 100)
            crop = warp_crop(img, k)[..., ::-1]
            x = torch.stack([to_tensor(np.ascontiguousarray(crop))]).to(a.device)
            with torch.no_grad():
                o1, o2 = net(x), net(torch.flip(x, dims=(2, 3)))
            pred = {kk: ATTRS[kk][int((o1[kk].softmax(1) + o2[kk].softmax(1)).argmax(1))] for kk in ATTRS}
            truth = {kk: str(m[kk]) for kk in ATTRS}
            ok = pred == truth
            n_ok += ok
            if not ok:
                wrong.append((src, name, f"als {pred}"))
            col = (0, 200, 0) if ok else (0, 0, 255)
            cv2.polylines(vis, [k.astype(np.int32)], True, col, 4)
        n_fp += len(kps) - len(used)
        cv2.imwrite(str(out_dir / Path(src).with_suffix(".jpg").name), vis, [cv2.IMWRITE_JPEG_QUALITY, 80])

    errs = np.array(errs)
    print(f"Karten {n_gt}: gefunden {n_found}, Fehlalarme {n_fp}, komplett richtig erkannt {n_ok}")
    print(f"Eckfehler (% kurze Kante): Median {np.median(errs):.2f}, 95% {np.percentile(errs, 95):.2f}, "
          f"max {errs.max():.2f}")
    for w in wrong:
        print("  FEHLER", *w)
    print(f"Bilder -> {out_dir}")


if __name__ == "__main__":
    main()
