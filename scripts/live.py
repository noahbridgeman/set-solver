"""Live-Erkennung am Mac: Kamera (z. B. iPhone über „Integrierte Kamera“), Video oder Fotos.

Aufruf:
  python scripts/live.py --list-cameras            # verfügbare Kameras anzeigen
  python scripts/live.py --source 1                # Kamera Nr. 1
  python scripts/live.py --source data/real/video.mov
  python scripts/live.py --source data/raw          # Ordner mit Fotos (Taste n = nächstes)
  python scripts/live.py --source video.mov --out runs/live/test.mp4 --no-show   # ohne Fenster speichern

Tasten im Fenster: q/Esc beenden, Leertaste Pause, s Bild speichern, l Beschriftung an/aus,
r Tracking zurücksetzen, a alle Sets / einzeln, Tab nächstes Set, n nächstes Foto.
Beim ersten Start fragt macOS nach dem Kamerazugriff für das Terminal bzw. PyCharm.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from setcards.common import ROOT  # noqa: E402
from setcards.live import DEFAULT_CLS, DEFAULT_DET, LivePipeline, Recognizer, draw, hud  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png", ".heic", ".bmp"}
WIN = "SET live"


def list_cameras() -> None:
    try:
        names = subprocess.run(["system_profiler", "SPCameraDataType"], capture_output=True, text=True).stdout
        print("macOS meldet:\n" + "\n".join(l.strip() for l in names.splitlines()
                                            if l.startswith("    ") and not l.startswith("      ")))
    except OSError:
        pass
    print("\nOpenCV-Indizes (Reihenfolge kann abweichen):")
    for i in range(6):
        cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():  # Indizes sind lückenlos -> beim ersten Fehlschlag aufhören
            break
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            ok, f = cap.read()
            print(f"  {i}: {'%dx%d' % (f.shape[1], f.shape[0]) if ok else 'kein Bild'}")
        cap.release()


def image_frames(src: Path):
    files = sorted(p for p in ([src] if src.is_file() else src.iterdir()) if p.suffix.lower() in IMG_EXT)
    for p in files:
        img = cv2.imread(str(p))
        if img is not None:
            yield p.name, img


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="0", help="Kamera-Index, Videodatei, Foto oder Ordner")
    ap.add_argument("--list-cameras", action="store_true")
    ap.add_argument("--det", default=str(DEFAULT_DET))
    ap.add_argument("--cls", default=str(DEFAULT_CLS))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--conf", type=float, default=0.6, help="Mindest-Konfidenz des Detektors")
    ap.add_argument("--min-conf", type=float, default=0.8, help="Mindest-Sicherheit je Attribut für Sets")
    ap.add_argument("--width", type=int, default=1920, help="gewünschte Kamera-Auflösung (Breite)")
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--display", type=int, default=1280, help="Fensterbreite in Pixeln")
    ap.add_argument("--out", default=None, help="Ergebnis speichern (.mp4 für Video, sonst Ordner)")
    ap.add_argument("--no-show", action="store_true")
    a = ap.parse_args()

    if a.list_cameras:
        list_cameras()
        return

    pipe = LivePipeline(Recognizer(Path(a.det), Path(a.cls), a.device, conf=a.conf), min_conf=a.min_conf)
    src = Path(a.source)
    is_images = not a.source.isdigit() and (src.is_dir() or src.suffix.lower() in IMG_EXT)
    snap_dir = ROOT / "runs" / "live"
    snap_dir.mkdir(parents=True, exist_ok=True)

    writer = None
    out = Path(a.out) if a.out else None
    if out and not is_images:
        out.parent.mkdir(parents=True, exist_ok=True)
    elif out:
        out.mkdir(parents=True, exist_ok=True)

    def show(img: np.ndarray, wait: int) -> int:
        if a.no_show:
            return -1
        s = a.display / img.shape[1]
        cv2.imshow(WIN, cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img)
        return cv2.waitKey(wait) & 0xFF

    show_labels = True
    only_set: int | None = None
    if is_images:
        for name, img in image_frames(src):
            pipe.tracker.reset()
            for _ in range(pipe.min_hits):  # Standbild: mehrfach zeigen, damit das Tracking die Karte bestätigt
                res = pipe.process(img)
            vis = draw(img, res, a.min_conf, show_labels, only_set)
            hud(vis, res, 0.0, only_set)
            print(f"{name}: {len(res.cards)} Karten, {len(res.sets)} Sets  "
                  + "; ".join(" + ".join(res.cards[i].text() for i in s) for s in res.sets))
            if out:
                cv2.imwrite(str(out / (Path(name).stem + ".jpg")), vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
            k = show(vis, 0)
            if k in (ord("q"), 27):
                break
        cv2.destroyAllWindows()
        return

    cap = cv2.VideoCapture(int(a.source), cv2.CAP_AVFOUNDATION) if a.source.isdigit() else cv2.VideoCapture(a.source)
    if not cap.isOpened():
        sys.exit(f"Quelle {a.source} lässt sich nicht öffnen (Kamerazugriff erlaubt? --list-cameras)")
    if a.source.isdigit():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
    fps, t_last, paused, vis, n = 0.0, time.perf_counter(), False, None, 0
    while True:
        if not paused or vis is None:
            ok, frame = cap.read()
            if not ok:
                break
            n += 1
            res = pipe.process(frame)
            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 / max(now - t_last, 1e-6) if fps else 1 / max(now - t_last, 1e-6)
            t_last = now
            vis = draw(frame, res, a.min_conf, show_labels, only_set)
            hud(vis, res, fps, only_set)
            if out:
                if writer is None:
                    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
                    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), src_fps,
                                             (vis.shape[1], vis.shape[0]))
                writer.write(vis)
            if a.no_show and n % 30 == 0:
                print(f"Bild {n}: {len(res.cards)} Karten, {len(res.sets)} Sets, {fps:.1f} fps", flush=True)
        k = show(vis, 1)
        if k in (ord("q"), 27):
            break
        if k == ord(" "):
            paused = not paused
        elif k == ord("s"):
            p = snap_dir / time.strftime("snap_%Y%m%d_%H%M%S.jpg")
            cv2.imwrite(str(p), vis)
            cv2.imwrite(str(p.with_name(p.stem + "_roh.jpg")), frame)
            print("gespeichert:", p)
        elif k == ord("l"):
            show_labels = not show_labels
        elif k == ord("r"):
            pipe.tracker.reset()
        elif k == ord("a"):
            only_set = None if only_set is not None else 0
        elif k == 9 and only_set is not None:  # Tab
            only_set += 1
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
