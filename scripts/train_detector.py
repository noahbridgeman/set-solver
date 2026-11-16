"""Phase 3: Karten-Detektor (YOLO26-pose, 1 Klasse, 4 Ecken) auf synthetischen Szenen trainieren.

Wichtig zur Augmentierung: Die Eck-Reihenfolge hängt von der Bildlage ab (k0-k1 = obere kurze
Kante, im Uhrzeigersinn). Drehung (degrees) und vertikales Spiegeln (flipud) würden diese Regel
verletzen und bleiben deshalb aus; Drehungen und Perspektive erzeugt bereits der Generator.
Horizontales Spiegeln ist erlaubt (flip_idx [1,0,3,2] in data.yaml).

Nach dem Training: Auswertung auf dem echten Mini-Testset (data/real/detect, aus den Rohfotos).

Aufruf:
  python scripts/train_detector.py                       # yolo26n-pose, 640 px, 40 Epochen
  python scripts/train_detector.py --epochs 1 --fraction 0.02 --name smoke   # Rauchtest
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from setcards.common import ROOT  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data" / "synth" / "v1" / "detect" / "data.yaml"))
    ap.add_argument("--model", default="yolo26n-pose.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--fraction", type=float, default=1.0, help="Anteil der Trainingsbilder")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--name", default="v1_det")
    a = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(a.model)
    model.train(
        data=a.data, imgsz=a.imgsz, epochs=a.epochs, batch=a.batch, workers=a.workers, device=a.device,
        fraction=a.fraction, project=str(ROOT / "runs" / "detector"), name=a.name, exist_ok=True,
        degrees=0.0, flipud=0.0, fliplr=0.5, perspective=0.0,  # siehe Docstring
        hsv_h=0.01, mosaic=1.0, close_mosaic=5, patience=15, plots=True,
    )
    real = ROOT / "data" / "real" / "detect" / "data.yaml"
    if real.exists():
        best = ROOT / "runs" / "detector" / a.name / "weights" / "best.pt"
        print("\n=== Echtes Mini-Testset (Rohfotos) ===")
        YOLO(str(best)).val(data=str(real), imgsz=a.imgsz, device=a.device, project=str(ROOT / "runs" / "detector"),
                            name=f"{a.name}_real", exist_ok=True, plots=True)


if __name__ == "__main__":
    main()
