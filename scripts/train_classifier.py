"""Phase 4: Attribut-Klassifikator auf synthetischen Crops trainieren.

Validierung auf
- synth:    data/synth/<v>/classify/val (gleiche Verteilung wie Training, für die Modellauswahl)
- real_wb:  die 81(+10) echten Karten, weißabgeglichen
- real_raw: dieselben ohne Weißabgleich (Kunstlicht-Farbstich)
Die echten Karten werden nie zum Training oder zur Modellauswahl benutzt.

Zusätzlich lernt der Kopf "card", Fehlalarme des Detektors abzulehnen (Negativbeispiele: Untergrund,
leeres Kartenpapier). Ausgewertet als neg_val (Anteil abgelehnt) und card-Recall der echten Karten.

Ausgabe: runs/classifier/<name>/{best.pt,last.pt,metrics.jsonl,real_errors.jpg}
Aufruf:
  python scripts/train_classifier.py --data data/synth/v1 --epochs 15
  python scripts/train_classifier.py --data data/synth/v1 data/synth/v2 --init models/classifier_v1.pt --epochs 6
  python scripts/train_classifier.py --data data/synth/smoke --epochs 1 --limit 2000   # Rauchtest
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from setcards.classifier import ATTRS, CardDataset, CardNet, NegativeDataset, WithCard  # noqa: E402
from setcards.common import ROOT  # noqa: E402

REAL = ROOT / "data" / "real" / "classify"


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def evaluate(model, loader, device, tta: bool = True):
    """Genauigkeit je Attribut und für die ganze Karte. TTA: Mittel über 0°/180°."""
    model.eval()
    correct = {k: 0 for k in ATTRS}
    card_ok, n, is_card = 0, 0, 0
    preds_all, ys_all = [], []
    for batch in loader:
        x, y = batch[0].to(device), batch[1].to(device)
        out = model(x)
        if tta:
            out2 = model(torch.flip(x, dims=(2, 3)))
            out = {k: out[k].softmax(1) + out2[k].softmax(1) for k in out}
        is_card += int((out["card"].argmax(1) == 1).sum())
        if (y < 0).all():  # Negativbeispiele: nur "card" zählt
            n += len(y)
            continue
        pred = torch.stack([out[k].argmax(1) for k in ATTRS], 1)
        ok = pred == y
        for i, k in enumerate(ATTRS):
            correct[k] += int(ok[:, i].sum())
        card_ok += int(ok.all(1).sum())
        n += len(y)
        preds_all.append(pred.cpu())
        ys_all.append(y.cpu())
    if not preds_all:
        return {"rejected": 1 - is_card / n}, None, None
    res = {k: correct[k] / n for k in ATTRS}
    res["card"] = card_ok / n
    res["is_card"] = is_card / n
    return res, torch.cat(preds_all).numpy(), torch.cat(ys_all).numpy()


def error_sheet(ds: CardDataset, preds, ys, path: Path, max_n: int = 40) -> None:
    tiles = []
    for i in np.flatnonzero((preds != ys).any(1))[:max_n]:
        fn, _ = ds.rows[i]
        im = cv2.imread(str(ds.img_dir / fn))
        im = cv2.copyMakeBorder(im, 0, 44, 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30))
        for j, (k, names) in enumerate(ATTRS.items()):
            if preds[i, j] != ys[i, j]:
                txt = f"{names[ys[i, j]][:5]}>{names[preds[i, j]][:5]}"
                cv2.putText(im, txt, (2, 205 + 12 * (j % 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 80, 255), 1,
                            cv2.LINE_AA)
        tiles.append(im)
    if not tiles:
        return
    while len(tiles) % 10:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(str(path), np.vstack([np.hstack(tiles[i:i + 10]) for i in range(0, len(tiles), 10)]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", nargs="+", default=[str(ROOT / "data" / "synth" / "v1")])
    ap.add_argument("--init", default=None, help="Startgewichte (Checkpoint); fehlende Köpfe werden neu angelegt")
    ap.add_argument("--neg-frac", type=float, default=0.12, help="Anteil Negativbeispiele im Training")
    ap.add_argument("--name", default=None)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=0.02)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None, help="nur die ersten N Trainingsbilder (Rauchtest)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-pretrained", action="store_true")
    a = ap.parse_args()

    datas = [Path(d) for d in a.data]
    data = datas[-1]
    name = a.name or f"{data.name}_{time.strftime('%Y%m%d_%H%M')}"
    out = ROOT / "runs" / "classifier" / name
    out.mkdir(parents=True, exist_ok=True)
    dev = pick_device(a.device)
    torch.manual_seed(0)

    trs = [CardDataset(d / "classify" / "train", d / "classify" / "train.csv", augment=True, limit=a.limit)
           for d in datas]
    n_pos = sum(len(t) for t in trs)
    neg_dirs = [d / "detect" / "images" / "train" for d in datas] + [ROOT / "data" / "backgrounds" / "raw_patches"]
    neg_lbls = [d / "detect" / "labels" / "train" for d in datas] + [None]
    neg = NegativeDataset(neg_dirs, neg_lbls, n=int(n_pos * a.neg_frac / (1 - a.neg_frac)), paper_from=trs[0])
    tr = ConcatDataset([WithCard(t) for t in trs] + [neg])
    evals = {}
    for d in datas:
        evals[f"synth_{d.name}"] = CardDataset(d / "classify" / "val", d / "classify" / "val.csv",
                                               limit=a.limit // 5 if a.limit else None)
    val_pos = evals[f"synth_{data.name}"]
    evals["neg_val"] = NegativeDataset([data / "detect" / "images" / "val"], [data / "detect" / "labels" / "val"],
                                       n=3000, seed=123, paper_from=val_pos)
    for v in ("wb", "raw"):
        if (REAL / f"{v}.csv").exists():
            evals[f"real_{v}"] = CardDataset(REAL / v, REAL / f"{v}.csv")
    print(f"Gerät {dev}; Training {len(tr)}, " + ", ".join(f"{k} {len(d)}" for k, d in evals.items()))

    kw = dict(num_workers=a.workers, persistent_workers=a.workers > 0, pin_memory=False)
    dl_tr = DataLoader(tr, batch_size=a.bs, shuffle=True, drop_last=True, **kw)
    dl_ev = {k: DataLoader(d, batch_size=256, shuffle=False, num_workers=min(a.workers, 2)) for k, d in evals.items()}

    model = CardNet(pretrained=not a.no_pretrained and not a.init)
    if a.init:
        missing, _ = model.load_state_dict(torch.load(a.init, map_location="cpu", weights_only=False)["model"],
                                           strict=False)
        print("Startgewichte", a.init, "| neu angelegt:", missing)
    model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    steps = a.epochs * len(dl_tr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05, ignore_index=-1)

    best = -1.0
    log = open(out / "metrics.jsonl", "a")
    for ep in range(1, a.epochs + 1):
        model.train()
        t0, tot, nb = time.time(), 0.0, 0
        for x, y, c in dl_tr:
            x, y, c = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True), c.to(dev, non_blocking=True)
            out_ = model(x)
            pos = y[:, 0] >= 0
            loss = crit(out_["card"], c)
            if pos.any():
                loss = loss + sum(crit(out_[k][pos], y[pos, i]) for i, k in enumerate(ATTRS))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
            nb += 1
        rec = {"epoch": ep, "loss": tot / max(nb, 1), "time_s": round(time.time() - t0, 1)}
        for k, dl in dl_ev.items():
            res, preds, ys = evaluate(model, dl, dev)
            rec[k] = {m: round(v, 4) for m, v in res.items()}
            if k.startswith("real") and preds is not None:
                error_sheet(evals[k], preds, ys, out / f"errors_{k}.jpg")
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(f"Epoche {ep}/{a.epochs}  Loss {rec['loss']:.4f}  ({rec['time_s']}s)  "
              + "  ".join(f"{k}: abgelehnt {rec[k]['rejected']:.4f}" if "rejected" in rec[k] else
                          f"{k}: Karte {rec[k]['card']:.4f} (als Karte {rec[k]['is_card']:.4f}) "
                          f"[F {rec[k]['color']:.3f} Fo {rec[k]['shape']:.3f} Fü {rec[k]['fill']:.3f} "
                          f"A {rec[k]['count']:.3f}]" for k in evals), flush=True)
        ckpt = {"model": model.state_dict(), "attrs": ATTRS, "epoch": ep, "metrics": rec}
        torch.save(ckpt, out / "last.pt")
        # Auswahl NUR auf synthetischer Validierung: Mittel der Karten-Genauigkeiten und der Ablehnung
        score = np.mean([rec[k]["card"] for k in rec if k.startswith("synth_")] + [rec["neg_val"]["rejected"]])
        if score > best:
            best = score
            torch.save(ckpt, out / "best.pt")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
