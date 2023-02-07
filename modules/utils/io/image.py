from collections.abc import Iterable
from pathlib import Path

from PIL import Image as Image

# Hardcode an extension set, due to PIL.Image.registered_extensions() returns too many weird stuffs.
SUPPORTED_EXTENSIONS = {'.jpe', '.jpg', '.jpeg', '.gif', '.apng', '.jfif', '.tif', '.tiff', '.bmp', '.png', '.webp'}


def is_image_file(path: Path):
    return path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS


def list_images(*paths: Path) -> Iterable[Path]:
    from itertools import chain
    return chain(*((
        x for x in path.iterdir() if is_image_file(x)
    ) for path in paths))


def read_image(filepath: Path) -> Image.Image:
    img = Image.open(filepath)

    if not img.mode == "RGB":
        img = img.convert("RGB")
    return img
