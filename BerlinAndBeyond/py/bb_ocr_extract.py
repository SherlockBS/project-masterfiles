# -*- coding: utf-8 -*-
# Berlin&Beyond - OCR extractor for exhibition labels in images (works on GitHub Actions)
# Requires: tesseract-ocr + deu, pip: pillow opencv-python pytesseract pandas

import os, re, sys, hashlib, datetime
from pathlib import Path
from typing import List
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageFilter
import pytesseract

# ---------- Config ----------
IMAGE_EXTS = {".jpg",".jpeg",".png",".tif",".tiff",".webp"}
OUTPUT_CSV = Path("data/kunstforum_ocr.csv")      # Ziel-CSV im Repo
SAVE_ROI_DIR = Path("data/roi")                   # optional: ausgeschnittene Label-Bereiche
LANG = "deu"                                      # OCR-Sprache
BATCH_LIMIT = None                                # None = alles; sonst int für Tests
MATERIAL_WORDS = (
    "öl","öl","acryl","mischtechnik","hartfaser","leinwand","papier",
    "holzschnitt","radierung","lithographie","fotografie","tempera",
    "bronze","skulptur","installation","gelatine","c-print","aquarell","tusche"
)
ARTIST_LINE = re.compile(r'[A-ZÄÖÜ].*\(\s*(?:\*|geb\.|born)?\s*\d{4}(?:\s*[-–/]\s*\d{2,4})?\s*\)$', re.IGNORECASE)
YEAR_STANDALONE = re.compile(r'^(?:ca\.\s*)?((?:19|20)\d{2}(?:[/-]\d{2,4})?)$')

# ---------- Utils ----------
def sha256_of_file(p: Path, buf_size=1024*1024) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(buf_size), b""):
            h.update(chunk)
    return h.hexdigest()

def detect_label_roi(pil_img: Image.Image) -> Image.Image:
    """Finde größten Textblock im unteren Bildbereich; liefere kontrastverstärkte Graustufen-ROI."""
    rgb = np.array(pil_img.convert("RGB"))
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    g = cv2.equalizeHist(g)
    _, bin_inv = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    h, w = bin_inv.shape
    mask = bin_inv.copy()
    mask[: int(h*0.35), :] = 0  # oberes Drittel ignorieren (meist Artwork)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
    morph = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    cnts, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    roi = None
    if cnts:
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        for c in cnts:
            x, y, w2, h2 = cv2.boundingRect(c)
            if h2 > 30 and w2 > 200:
                pad = 12
                x0 = max(0, x - pad); y0 = max(0, y - pad)
                x1 = min(w, x + w2 + pad); y1 = min(h, y + h2 + pad)
                roi = pil_img.crop((x0, y0, x1, y1)).convert("L")
                break
    if roi is None:
        roi = pil_img.convert("L").crop((0, int(h*0.45), w, h))
    roi = ImageOps.autocontrast(roi).filter(ImageFilter.SHARPEN)
    return roi

def ocr_lines(pil_img: Image.Image, lang=LANG) -> List[str]:
    config = r"--oem 1 --psm 4 -l " + lang  # Single column
    txt = pytesseract.image_to_string(pil_img, config=config)
    return [l.strip() for l in txt.splitlines() if l.strip()]

def parse_fields(lines: List[str]):
    artist = title = year = None
    artist_idx = None
    for i, line in enumerate(lines):
        if ARTIST_LINE.search(line):
            artist = line; artist_idx = i; break
    if artist_idx is not None:
        tparts = []
        j = artist_idx - 1
        while j >= 0 and len(tparts) < 4:
            low = lines[j].lower()
            if any(w in low for w in MATERIAL_WORDS):  # Material/Technik nicht in Titel ziehen
                break
            tparts.append(lines[j])
            j -= 1
        if tparts:
            title = " ".join(reversed(tparts)).strip()
        for k in range(artist_idx + 1, min(artist_idx + 6, len(lines))):
            m = YEAR_STANDALONE.match(lines[k])
            if m:
                year = m.group(1); break
    else:
        if lines: title = lines[0]
        if len(lines) > 1: artist = lines[1]
    return artist or "", title or "", year or ""

def main():
    # Alle Bildpfade im Repo finden
    repo = Path(".").resolve()
    images = []
    for p in repo.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS and ".git" not in p.parts and not str(p).startswith(str(OUTPUT_CSV.parent)):
            images.append(p)
    images.sort(key=lambda x: x.as_posix())

    if BATCH_LIMIT:
        images = images[:BATCH_LIMIT]

    rows = []
    SAVE_ROI_DIR.mkdir(parents=True, exist_ok=True)

    for p in images:
        try:
            img = Image.open(p)
            roi = detect_label_roi(img)
            lines = ocr_lines(roi)
            artist, title, year = parse_fields(lines)
            notes = "" if (artist and title) else "check manually"
            rows.append({
                "filename": p.as_posix(),
                "sha256": sha256_of_file(p),
                "artist": artist,
                "title": title,
                "year": year,
                "notes": notes
            })
            # optional: ROI ablegen für Audit
            roi_name = sha256_of_file(p)[:12] + ".png"
            roi.convert("RGB").save(SAVE_ROI_DIR / roi_name)
        except Exception as e:
            rows.append({
                "filename": p.as_posix(),
                "sha256": sha256_of_file(p) if p.exists() else "",
                "artist": "",
                "title": "",
                "year": "",
                "notes": f"error: {e}"
            })

    df = pd.DataFrame(rows, columns=["filename","sha256","artist","title","year","notes"])
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    # Wenn es schon eine CSV gibt: nur updaten, wenn Änderungen
    if OUTPUT_CSV.exists():
        old = pd.read_csv(OUTPUT_CSV)
        # Merge auf filename (Quelle der Wahrheit ist NEU → überschreibt Felder)
        merged = (old.merge(df, on="filename", how="outer", suffixes=("_old",""))
                    .assign(
                        sha256=lambda x: x["sha256"].fillna(x["sha256_old"]),
                        artist=lambda x: x["artist"].fillna(x["artist_old"]),
                        title=lambda x: x["title"].fillna(x["title_old"]),
                        year=lambda x: x["year"].fillna(x["year_old"]),
                        notes=lambda x: x["notes"].fillna(x["notes_old"])
                    )[["filename","sha256","artist","title","year","notes"]])
        df_out = merged.sort_values("filename")
    else:
        df_out = df.sort_values("filename")

    df_out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"[OK] Wrote {OUTPUT_CSV} with {len(df_out)} rows.")

if __name__ == "__main__":
    sys.exit(main())
