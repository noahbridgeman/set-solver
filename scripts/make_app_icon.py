"""App-Symbol (1024x1024) aus den gemessenen Kartenformen: drei aufgefächerte Karten, die ein Set bilden.

Aufruf: .venv/bin/python scripts/make_app_icon.py  ->  ios/SetSolver/Assets.xcassets/AppIcon.appiconset/icon.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from setcards.shapes import sdf_rounded_rect, shape_sdf  # noqa: E402

S = 1024
SS = 3  # Supersampling gegen Treppenkanten
OUT = ROOT / "ios/SetSolver/Assets.xcassets/AppIcon.appiconset"


def hex_rgb(h: str) -> np.ndarray:
    return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], np.float32)


def render_card(g: dict, shape: str, color: np.ndarray, px_per_mm: float) -> Image.Image:
    cw, ch, cr = g["card"]["width"], g["card"]["height"], g["card"]["corner_radius"]
    W, H = int(cw * px_per_mm * SS), int(ch * px_per_mm * SS)
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    x = (xs + 0.5) / (px_per_mm * SS) - cw / 2
    y = (ys + 0.5) / (px_per_mm * SS) - ch / 2
    px = 1 / (px_per_mm * SS)
    card = np.clip(0.5 - sdf_rounded_rect(x, y, cw / 2, ch / 2, cr) / px, 0, 1)
    sym = np.clip(0.5 - shape_sdf(g["shapes"][shape], x, y) / px, 0, 1)
    rgb = np.full((H, W, 3), 246, np.float32) * (1 - sym[..., None]) + color * sym[..., None]
    a = card
    img = np.dstack([rgb, a * 255]).clip(0, 255).astype(np.uint8)
    return Image.fromarray(img, "RGBA").resize((W // SS, H // SS), Image.LANCZOS)


def main() -> None:
    g = json.loads((ROOT / "data/geometry.json").read_text())
    # Hintergrund: dunkler Verlauf
    yy = np.linspace(0, 1, S)[:, None, None]
    top, bot = np.array([46, 38, 82], np.float32), np.array([18, 16, 36], np.float32)
    bg = (top * (1 - yy) + bot * yy) * np.ones((1, S, 1), np.float32)
    icon = Image.fromarray(bg.astype(np.uint8), "RGB").convert("RGBA")

    px_per_mm = 4.5
    # (Form, Farbe, Drehung gegen den Uhrzeigersinn, Versatz x, Versatz y); mittlere Karte zuletzt = oben
    cards = [("raute", "rot", -13, -265, 40), ("welle", "lila", 13, 265, 40), ("oval", "gruen", 0, 0, 0)]
    for shape, color, angle, dx, dy in cards:
        c = render_card(g, shape, hex_rgb(g["colors"][color]["srgb_hex"]), px_per_mm).rotate(
            angle, resample=Image.BICUBIC, expand=True)
        cx, cy = S // 2 + dx - c.width // 2, S // 2 + dy - c.height // 2 + 10
        pad = 60  # Platz für den weichen Schatten
        alpha = Image.new("L", (c.width + 2 * pad, c.height + 2 * pad), 0)
        alpha.paste(c.getchannel("A").point(lambda v: int(v * 0.6)), (pad, pad))
        shadow = Image.new("RGBA", alpha.size, (0, 0, 0, 255))
        shadow.putalpha(alpha.filter(ImageFilter.GaussianBlur(20)))
        icon.alpha_composite(shadow, (cx - pad + 6, cy - pad + 16))
        icon.alpha_composite(c, (cx, cy))
    OUT.mkdir(parents=True, exist_ok=True)
    icon.convert("RGB").save(OUT / "icon.png")  # App-Symbole dürfen keine Transparenz haben
    (OUT / "Contents.json").write_text(json.dumps({
        "images": [{"filename": "icon.png", "idiom": "universal", "platform": "ios", "size": "1024x1024"}],
        "info": {"author": "xcode", "version": 1}}, indent=2))
    (OUT.parent / "Contents.json").write_text(json.dumps({"info": {"author": "xcode", "version": 1}}, indent=2))
    print("gespeichert:", OUT / "icon.png")


if __name__ == "__main__":
    main()
