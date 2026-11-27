"""Build web-sized dashboard images from the original dataset photos.

Reads only from ``dataset/raw`` (never writes to the dataset) and writes
optimised JPEGs to ``static/images/dataset/`` plus a ``manifest.json`` that
records the exact source file of every image.

    .venv\\Scripts\\python.exe app\\build_static_assets.py

The leaf bounding box used for cropping comes from the verified
preprocessing pipeline (``core.preprocessing``), so crops follow the leaf.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.preprocessing import load_image, preprocess_leaf_image  # noqa: E402

OUT_DIR = ROOT / "static" / "images" / "dataset"

# One representative front-view photo per class (chosen visually: sharp,
# high resolution, whole leaf visible).
CLASS_SOURCES = {
    "Mangkok": "dataset/raw/daun_mangkok/foto depan/mangkok 7.png",
    "Salam": "dataset/raw/daun_salam/foto depan/daun salam 9.png",
    "Sambang Darah": "dataset/raw/daun_sambang_darah/foto depan/sambang darah 6.png",
    "Sirih": "dataset/raw/daun_sirih/foto depan/sirih 5.jpeg",
    "Tapak Kuda": "dataset/raw/daun_tapak_kuda/foto depan/tapak kuda 12.png",
}

# Close-up of the Mangkok leaf surface for the hero banner (x1, y1, x2, y2).
HERO_SOURCE = "dataset/raw/daun_mangkok/foto depan/mangkok 7.png"
HERO_BOX = (300, 500, 1000, 780)

CARD_SIZE = (600, 750)  # 4:5 portrait
TEXTURE_SIZE = (700, 900)
THUMB_SIZE = 160
JPEG_QUALITY = 84


def slugify(name: str) -> str:
    return name.lower().replace(" ", "-")


def fit_box(box, aspect, width, height):
    """Grow box (x1, y1, x2, y2) to the target aspect ratio, clamped to the image."""
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    if bw / bh < aspect:
        bw = bh * aspect
    else:
        bh = bw / aspect
    bw, bh = min(bw, width), min(bh, height)
    x1 = min(max(0, cx - bw / 2), width - bw)
    y1 = min(max(0, cy - bh / 2), height - bh)
    return int(round(x1)), int(round(y1)), int(round(x1 + bw)), int(round(y1 + bh))


def leaf_bbox(image_array):
    """Leaf ROI on the original image from the verified preprocessing pipeline."""
    _, _, _, info, inter = preprocess_leaf_image(image_array)
    if inter["roi_bbox"] is None:
        h, w = image_array.shape[:2]
        return 0, 0, w, h
    return inter["roi_bbox"]


def save_jpeg(img: Image.Image, path: Path) -> None:
    img.save(path, format="JPEG", quality=JPEG_QUALITY, optimize=True, progressive=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {"classes": {}, "hero": {}}

    for class_name, rel in CLASS_SOURCES.items():
        array = load_image(ROOT / rel)
        image = Image.fromarray(array)
        width, height = image.size
        x1, y1, x2, y2 = leaf_bbox(array)
        slug = slugify(class_name)

        # Card: whole leaf, padded to 4:5.
        card_box = fit_box((x1, y1, x2, y2), CARD_SIZE[0] / CARD_SIZE[1], width, height)
        card = image.crop(card_box).resize(CARD_SIZE, Image.LANCZOS)
        save_jpeg(card, OUT_DIR / f"{slug}_card.jpg")

        # Texture: central part of the leaf only (no background).
        bw, bh = x2 - x1, y2 - y1
        inner = (x1 + bw * 0.30, y1 + bh * 0.30, x2 - bw * 0.30, y2 - bh * 0.30)
        tex_box = fit_box(inner, TEXTURE_SIZE[0] / TEXTURE_SIZE[1], width, height)
        texture = image.crop(tex_box).resize(TEXTURE_SIZE, Image.LANCZOS)
        save_jpeg(texture, OUT_DIR / f"{slug}_texture.jpg")

        # Thumbnail: square around the leaf.
        thumb_box = fit_box((x1, y1, x2, y2), 1.0, width, height)
        thumb = image.crop(thumb_box).resize((THUMB_SIZE, THUMB_SIZE), Image.LANCZOS)
        save_jpeg(thumb, OUT_DIR / f"{slug}_thumb.jpg")

        manifest["classes"][class_name] = {
            "source": rel,
            "card": f"images/dataset/{slug}_card.jpg",
            "texture": f"images/dataset/{slug}_texture.jpg",
            "thumb": f"images/dataset/{slug}_thumb.jpg",
            "card_box": card_box,
            "texture_box": tex_box,
        }
        print(f"[OK] {class_name:14s} <- {rel}")

    hero = Image.fromarray(load_image(ROOT / HERO_SOURCE)).crop(HERO_BOX)
    hero = hero.resize((1400, int(1400 * hero.height / hero.width)), Image.LANCZOS)
    save_jpeg(hero, OUT_DIR / "hero.jpg")
    manifest["hero"] = {"source": HERO_SOURCE, "box": HERO_BOX, "file": "images/dataset/hero.jpg"}
    print(f"[OK] hero           <- {HERO_SOURCE}")

    with open(OUT_DIR / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"Manifest: {OUT_DIR / 'manifest.json'}")


if __name__ == "__main__":
    main()
