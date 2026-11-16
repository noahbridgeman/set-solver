"""Modelle für die iPhone-App nach CoreML umwandeln und gegen PyTorch prüfen.

Läuft in der eigenen Umgebung .venv-coreml (Python 3.12, coremltools gibt es nicht für 3.14):
  /opt/homebrew/bin/python3.12 -m venv .venv-coreml
  .venv-coreml/bin/pip install coremltools torch==2.7.0 torchvision==0.22.0 ultralytics opencv-python-headless scipy

Aufruf:
  .venv-coreml/bin/python scripts/export_coreml.py                  # Standardmodelle v2
  .venv-coreml/bin/python scripts/export_coreml.py --cls models/classifier_v3.pt

Ergebnis: ios/Models/SetDetector.mlpackage und ios/Models/SetClassifier.mlpackage. Danach in Xcode
einfach neu bauen (Run) – die App liest die Modelle unter diesen festen Namen.

Detektor: Eingabe Bild 384x640 (Hochformat, RGB, letterboxed mit Grau 114), Ausgabe (1,17,5040):
  je Kandidat Mitte x, Mitte y, Breite, Höhe, Sicherheit, dann 4 Ecken als (x, y, sichtbar) – in
  Eingabe-Pixeln, Ecke 0-1 = obere kurze Kante. Doppelte Treffer filtert die App per NMS (IoU 0.7),
  genau wie Ultralytics in Python. Bewusst nicht der NMS-freie YOLO26-Kopf: Der ist ein anderer
  Ausgabezweig, dessen Ecken im Median 3 % von denen in Python abweichen.
Klassifikator: Eingabe entzerrte Karte 128x192 RGB, Ausgabe "probs" (1,14) = Softmax je Kopf
  [Farbe 3 | Form 3 | Füllung 3 | Anzahl 3 | keine Karte, Karte], gemittelt über 0° und 180°.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from setcards.classifier import ATTRS, IN_H, IN_W, MEAN, STD, CardNet  # noqa: E402
from setcards.synth import warp_crop  # noqa: E402

DET_H, DET_W = 640, 384  # Hochformat, wie das iPhone-Kamerabild in der App
OUT = ROOT / "ios" / "Models"
HEADS = list(ATTRS) + ["card"]


class ClassifierExport(nn.Module):
    """Normierung, 180°-Mittelung und Softmax direkt im Modell -> die App muss nichts nachrechnen."""

    def __init__(self, net: CardNet):
        super().__init__()
        self.net = net
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1))

    def forward(self, x):  # x: RGB 0..1
        x = (x - self.mean) / self.std
        o1, o2 = self.net(x), self.net(torch.flip(x, dims=(2, 3)))
        return torch.cat([(o1[k].softmax(1) + o2[k].softmax(1)) / 2 for k in HEADS], 1)


def export_detector(pt: Path, dst: Path) -> None:
    from ultralytics import YOLO
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "det.pt"
        shutil.copy(pt, src)
        out = YOLO(str(src)).export(format="coreml", imgsz=(DET_H, DET_W), half=False, nms=None)
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(out, dst)


def export_classifier(pt: Path, dst: Path) -> None:
    import coremltools as ct
    net = CardNet(pretrained=False)
    ck = torch.load(str(pt), map_location="cpu", weights_only=False)
    net.load_state_dict(ck["model"])
    m = ClassifierExport(net.eval()).eval()
    ts = torch.jit.trace(m, torch.rand(1, 3, IN_H, IN_W))
    ml = ct.convert(ts, inputs=[ct.ImageType(name="image", shape=(1, 3, IN_H, IN_W), scale=1 / 255.0,
                                             color_layout=ct.colorlayout.RGB)],
                    outputs=[ct.TensorType(name="probs")], convert_to="mlprogram",
                    minimum_deployment_target=ct.target.iOS16)
    ml.short_description = "SET-Karte: Farbe, Form, Füllung, Anzahl, ist Karte"
    ml.user_defined_metadata["heads"] = json.dumps({k: [str(v) for v in ATTRS[k]] for k in ATTRS} |
                                                   {"card": ["nein", "ja"]})
    ml.user_defined_metadata["source"] = pt.name
    shutil.rmtree(dst, ignore_errors=True)
    ml.save(str(dst))


# ------------------------------------------------------------------ Gegenprobe
def letterbox(img: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    h, w = img.shape[:2]
    r = min(DET_H / h, DET_W / w)
    nh, nw = round(h * r), round(w * r)
    top, left = (DET_H - nh) // 2, (DET_W - nw) // 2
    out = np.full((DET_H, DET_W, 3), 114, np.uint8)
    out[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return out, r, left, top


def coreml_detect(det, img_bgr: np.ndarray, conf: float = 0.6) -> np.ndarray:
    from PIL import Image
    lb, r, dx, dy = letterbox(img_bgr)
    o = next(iter(det.predict({"image": Image.fromarray(lb[..., ::-1].copy())}).values()))[0].T  # (5040,17)
    o = o[o[:, 4] >= conf]
    o = o[nms(o[:, :4], o[:, 4])]
    k = (o[:, 5:].reshape(-1, 4, 3)[..., :2] - [dx, dy]) / r
    # wie Ultralytics: Ecken auf das Bild begrenzen (die App macht es genauso)
    h, w = img_bgr.shape[:2]
    return np.stack([k[..., 0].clip(0, w), k[..., 1].clip(0, h)], -1)


def nms(xywh: np.ndarray, score: np.ndarray, iou: float = 0.7) -> list[int]:
    """Gleiche einfache NMS wie in der App (Swift): höchste Sicherheit zuerst, Überlappende verwerfen."""
    b = np.concatenate([xywh[:, :2] - xywh[:, 2:] / 2, xywh[:, :2] + xywh[:, 2:] / 2], 1)
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    keep: list[int] = []
    for i in np.argsort(-score):
        ok = True
        for j in keep:
            w = max(0.0, min(b[i, 2], b[j, 2]) - max(b[i, 0], b[j, 0]))
            h = max(0.0, min(b[i, 3], b[j, 3]) - max(b[i, 1], b[j, 1]))
            if w * h / (area[i] + area[j] - w * h + 1e-9) > iou:
                ok = False
                break
        if ok:
            keep.append(i)
    return keep


def check(det_pt: Path, cls_pt: Path) -> None:
    import coremltools as ct
    from PIL import Image
    from scipy.optimize import linear_sum_assignment
    from ultralytics import YOLO
    det_ml = ct.models.MLModel(str(OUT / "SetDetector.mlpackage"))
    cls_ml = ct.models.MLModel(str(OUT / "SetClassifier.mlpackage"))
    det_t = YOLO(str(det_pt))
    net = CardNet(pretrained=False)
    net.load_state_dict(torch.load(str(cls_pt), map_location="cpu", weights_only=False)["model"])
    cls_t = ClassifierExport(net.eval()).eval()

    files = sorted([*(ROOT / "data/raw").glob("*.jp*g"), *(ROOT / "data/real/live_snaps").glob("*_roh.jpg")])
    n_t = n_ml = n_match = same_label = 0
    kp_err, p_diff = [], []
    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            continue
        if img.shape[1] > img.shape[0]:  # App arbeitet im Hochformat
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        r = det_t.predict(img, imgsz=(DET_H, DET_W), conf=0.6, device="cpu", verbose=False)[0]
        kt = r.keypoints.xy.numpy() if r.keypoints is not None else np.zeros((0, 4, 2))
        km = coreml_detect(det_ml, img)
        n_t, n_ml = n_t + len(kt), n_ml + len(km)
        if len(kt) and len(km):
            cost = np.linalg.norm(kt.mean(1)[:, None] - km.mean(1)[None], axis=2)
            for i, j in zip(*linear_sum_assignment(cost)):
                short = np.linalg.norm(kt[i, 1] - kt[i, 0])
                kp_err.append(min(np.linalg.norm(kt[i] - np.roll(km[j], s, 0), axis=1).max() for s in (0, 2)) / short)
                n_match += 1
        for k in km:
            crop = np.ascontiguousarray(warp_crop(img, k)[..., ::-1])
            with torch.no_grad():
                x = torch.from_numpy(crop.transpose(2, 0, 1).astype(np.float32) / 255.0)[None]
                pt_ = cls_t(x).numpy()[0]
            pm = cls_ml.predict({"image": Image.fromarray(crop)})["probs"][0]
            p_diff.append(np.abs(pt_ - pm).max())
            lt = [pt_[3 * i:3 * i + 3].argmax() for i in range(4)] + [pt_[12:].argmax()]
            lm = [pm[3 * i:3 * i + 3].argmax() for i in range(4)] + [pm[12:].argmax()]
            same_label += lt == lm
    kp_err, p_diff = np.array(kp_err), np.array(p_diff)
    print(f"\nGegenprobe auf {len(files)} echten Bildern (Hochformat):")
    print(f"  Detektor: PyTorch {n_t} Karten, CoreML {n_ml}, zugeordnet {n_match}; "
          f"Eckabweichung max {100 * kp_err.max():.2f} % der kurzen Kante (Median {100 * np.median(kp_err):.3f} %)")
    print(f"  Klassifikator: {same_label}/{len(p_diff)} gleiche Labels, "
          f"Wahrscheinlichkeiten max. Abweichung {p_diff.max():.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--det", default=str(ROOT / "models/detector_v2.pt"))
    ap.add_argument("--cls", default=str(ROOT / "models/classifier_v2.pt"))
    ap.add_argument("--no-check", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    export_detector(Path(a.det), OUT / "SetDetector.mlpackage")
    export_classifier(Path(a.cls), OUT / "SetClassifier.mlpackage")
    (OUT / "models.json").write_text(json.dumps({"detector": Path(a.det).name, "classifier": Path(a.cls).name,
                                                 "det_input": [DET_W, DET_H]}, indent=2))
    print(f"gespeichert in {OUT}")
    if not a.no_check:
        check(Path(a.det), Path(a.cls))


if __name__ == "__main__":
    main()
