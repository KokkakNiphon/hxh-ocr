"""Thumbnail contact sheet of a folder of images (for checking downloaded data by eye).
Run isolated when the folder holds downloaded files:

    python -I scripts/contact_sheet.py data/real/raw previews/real_raw_sheet.jpg
"""
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    files = sorted(p for p in src.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".gif"})
    cw, ch, cols = 300, 200, 6
    rows = (len(files) + cols - 1) // cols
    sheet = Image.new("RGB", (cw * cols, (ch + 18) * rows), "white")
    d = ImageDraw.Draw(sheet)
    for i, p in enumerate(files):
        try:
            im = Image.open(p)
            im.draft("RGB", (cw * 2, ch * 2))
            im = ImageOps.contain(im.convert("RGB"), (cw - 6, ch - 6))
        except Exception as e:  # unreadable file: show the error instead
            im = Image.new("RGB", (cw - 6, ch - 6), "pink")
            ImageDraw.Draw(im).text((4, 4), str(e)[:60], fill="black")
        x, y = (i % cols) * cw, (i // cols) * (ch + 18)
        sheet.paste(im, (x + 3, y + 3))
        d.text((x + 3, y + ch), f"{i:02d} {p.name[:44]}", fill=(180, 0, 0))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=85)
    print(out, len(files), "images")


if __name__ == "__main__":
    main()
