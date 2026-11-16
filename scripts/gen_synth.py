"""Phase 2: Synthetischen Datensatz erzeugen.

Ausgabe (Standard data/synth/v1/):
  detect/images/{train,val}/*.jpg, detect/labels/{train,val}/*.txt, detect/data.yaml
      YOLO-pose: Klasse 0 (Karte), Box, 4 Ecken (x, y, v), Reihenfolge siehe setcards/synth.py
  classify/{train,val}/*.jpg + classify/{train,val}.csv
      entzerrte Karten 128x192 (wie in der Inferenz aus dem Kamerabild geschnitten),
      Spalten: file, color, shape, fill, count

Aufruf:
  python scripts/gen_synth.py --prepare-backgrounds        # einmalig: Hintergrund-Ausschnitte aus data/raw
  python scripts/gen_synth.py --n 200 --out data/synth/test
  python scripts/gen_synth.py --n 20000                     # voller Lauf
  python scripts/gen_synth.py --rebuild-csv                 # Labels nach Abbruch rekonstruieren
Eigene Hintergrundfotos (ohne SET-Karten) einfach in data/backgrounds/ legen.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from setcards.common import RAW, ROOT, find_cards, read_lin  # noqa: E402
from setcards.render import load_geometry  # noqa: E402
from setcards.synth import Backgrounds, render_scene, warp_crop  # noqa: E402

BG_DIR = ROOT / "data" / "backgrounds"

_G = _BGS = _ARGS = None


def _init(args):
    global _G, _BGS, _ARGS
    cv2.setNumThreads(1)
    _ARGS = args
    _G = load_geometry()
    _BGS = Backgrounds(Path(args.backgrounds))


def _scene(idx: int):
    a = _ARGS
    rng = np.random.default_rng([a.seed, idx])
    split = "val" if idx % round(1 / a.val_frac) == 0 else "train"
    res = render_scene(rng, _BGS, _G)
    H, W = res.image.shape[:2]
    name = f"{a.seed:03d}_{idx:06d}"

    # Detektor: Bild auf die lange Seite det_size verkleinern (Labels sind normiert)
    s = a.det_size / max(H, W)
    det = cv2.resize(res.image, (round(W * s), round(H * s)), interpolation=cv2.INTER_AREA)
    out = Path(a.out)
    cv2.imwrite(str(out / "detect" / "images" / split / f"{name}.jpg"), det[..., ::-1],
                [cv2.IMWRITE_JPEG_QUALITY, 92])
    lines = []
    for kp, box, vis in zip(res.keypoints, res.boxes, res.visible_frac):
        x0, y0, x1, y1 = np.clip(box, 0, [W, H, W, H])
        if vis < a.min_visible or x1 - x0 < 4 or y1 - y0 < 4:
            continue
        vals = [0, (x0 + x1) / 2 / W, (y0 + y1) / 2 / H, (x1 - x0) / W, (y1 - y0) / H]
        for x, y, v in kp:
            vals += [x / W, y / H, int(v)] if v > 0 else [0, 0, 0]
        lines.append(" ".join(f"{v:.6f}" if isinstance(v, float) else str(v) for v in vals))
    (out / "detect" / "labels" / split / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))

    # Klassifikator: gut sichtbare Karten wie in der Inferenz entzerren
    rows = []
    for j, (kp, card, vis, inside) in enumerate(zip(res.keypoints, res.cards, res.visible_frac, res.inside)):
        height_px = np.linalg.norm(kp[3, :2] - kp[0, :2])
        if not inside or vis < 0.9 or height_px < a.min_crop_px:
            continue
        crop = warp_crop(res.image, kp, rng, jitter=abs(rng.normal(0, a.crop_jitter)))
        fn = f"{name}_{j:02d}.jpg"
        cv2.imwrite(str(out / "classify" / split / fn), crop[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 95])
        rows.append((split, fn, card.color, card.shape, card.fill, card.count))
    return split, len(lines), rows


def prepare_backgrounds(n_per_image: int = 25, seed: int = 0) -> None:
    """Hintergrund-Ausschnitte (Tisch, Filz, Tablet) aus den Rohfotos, Karten ausgespart."""
    out = BG_DIR / "raw_patches"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    expected = {"farben": 3}
    total = 0
    for p in sorted(RAW.glob("*.jpeg")):
        n = expected.get(p.stem, 1 if p.stem.startswith("einzel") else 9)
        img = read_lin(p)
        corners = find_cards(img, expected=n)
        H, W = img.shape[:2]
        mask = np.zeros((H, W), np.uint8)
        for c in corners:
            cv2.fillPoly(mask, [np.round(c).astype(np.int32)], 1)
        mask = cv2.dilate(mask, np.ones((81, 81), np.uint8))
        integ = cv2.integral(mask)
        bgr = cv2.imread(str(p))
        got = 0
        for _ in range(400):
            if got >= n_per_image:
                break
            sz = int(rng.uniform(500, 1600))
            if sz >= min(H, W):
                continue
            x, y = int(rng.integers(0, W - sz)), int(rng.integers(0, H - sz))
            if integ[y + sz, x + sz] - integ[y, x + sz] - integ[y + sz, x] + integ[y, x] > 0:
                continue
            cv2.imwrite(str(out / f"{p.stem}_{got:02d}.jpg"), bgr[y:y + sz, x:x + sz], [cv2.IMWRITE_JPEG_QUALITY, 92])
            got += 1
        total += got
        print(f"{p.name}: {got} Ausschnitte")
    print(f"{total} Hintergründe -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "data" / "synth" / "v1"))
    ap.add_argument("--n", type=int, default=2000, help="Anzahl Szenen")
    ap.add_argument("--start", type=int, default=0, help="erster Szenenindex (zum Erweitern)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--det-size", type=int, default=960, help="lange Seite der Detektorbilder")
    ap.add_argument("--min-visible", type=float, default=0.15, help="Mindestanteil sichtbar für ein Label")
    ap.add_argument("--min-crop-px", type=float, default=60, help="Mindesthöhe einer Karte für Klassifikator-Crops")
    ap.add_argument("--crop-jitter", type=float, default=0.012, help="Eckrauschen relativ zur kurzen Kante")
    ap.add_argument("--backgrounds", default=str(BG_DIR))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--prepare-backgrounds", action="store_true")
    ap.add_argument("--rebuild-csv", action="store_true", help="CSVs aus vorhandenen Crops rekonstruieren")
    args = ap.parse_args()

    if args.prepare_backgrounds:
        prepare_backgrounds()
        return
    if args.rebuild_csv:
        rebuild_csv(Path(args.out))
        return

    out = Path(args.out)
    for split in ("train", "val"):
        (out / "detect" / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "detect" / "labels" / split).mkdir(parents=True, exist_ok=True)
        (out / "classify" / split).mkdir(parents=True, exist_ok=True)
    (out / "detect" / "data.yaml").write_text(
        f"path: {(out / 'detect').resolve()}\ntrain: images/train\nval: images/val\n"
        "kpt_shape: [4, 3]\nflip_idx: [1, 0, 3, 2]\nnames:\n  0: card\n")

    # CSV-Zeilen sofort anhängen (absturzsicher); bei --start 0 neu beginnen
    fhs, writers = {}, {}
    for split in ("train", "val"):
        f = out / "classify" / f"{split}.csv"
        new = not f.exists() or args.start == 0
        fhs[split] = open(f, "w" if new else "a", newline="")
        writers[split] = csv.writer(fhs[split])
        if new:
            writers[split].writerow(["file", "color", "shape", "fill", "count"])

    t0 = time.time()
    n_rows, n_lab, idx = 0, 0, range(args.start, args.start + args.n)
    with Pool(args.workers, initializer=_init, initargs=(args,), maxtasksperchild=40) as pool:
        for k, (split, nl, r) in enumerate(pool.imap_unordered(_scene, idx, chunksize=4), 1):
            for row in r:
                writers[row[0]].writerow(row[1:])
            n_rows += len(r)
            n_lab += nl
            if k % 100 == 0 or k == args.n:
                for fh in fhs.values():
                    fh.flush()
                el = time.time() - t0
                print(f"{k}/{args.n} Szenen, {n_lab} Kartenlabels, {n_rows} Crops, "
                      f"{el:.0f}s ({el / k * args.workers:.2f}s/Szene/Kern)", flush=True)
    for fh in fhs.values():
        fh.close()
    print(f"fertig -> {out}")


def rebuild_csv(out: Path) -> None:
    """Klassifikator-CSVs aus den Crop-Dateinamen rekonstruieren (nach einem Abbruch).

    Dateiname <seed>_<szene>_<karte>.jpg; das Kartenlayout einer Szene hängt nur vom Seed ab
    und wird ohne Rendern nachgerechnet (synth.scene_setup).
    """
    from setcards.synth import scene_setup
    cache: dict = {}
    for split in ("train", "val"):
        rows = []
        for p in sorted((out / "classify" / split).glob("*.jpg")):
            seed, idx, j = (int(v) for v in p.stem.split("_"))
            if (seed, idx) not in cache:
                cache = {(seed, idx): scene_setup(np.random.default_rng([seed, idx]))[2]}
            c = cache[(seed, idx)][j].card
            rows.append([p.name, c.color, c.shape, c.fill, c.count])
        with open(out / "classify" / f"{split}.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["file", "color", "shape", "fill", "count"])
            w.writerows(rows)
        print(f"{split}: {len(rows)} Labels rekonstruiert")


if __name__ == "__main__":
    main()
