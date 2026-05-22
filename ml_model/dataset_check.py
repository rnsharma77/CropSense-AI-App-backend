#!/usr/bin/env python3
import os
import csv
# imghdr removed in newer Pythons; we'll use simple header checks as fallback
from pathlib import Path

try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

ROOT = Path(__file__).resolve().parent
TRAIN_DIR = ROOT / 'dataset' / 'train'
OUT_SUMMARY = ROOT / 'dataset_check_output.txt'
OUT_MANIFEST = ROOT / 'manifest.csv'

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

summary_lines = []
manifest_rows = []

def _quick_header_check(p: Path):
    try:
        with open(p, 'rb') as f:
            h = f.read(8)
        if h.startswith(b"\xff\xd8\xff"):
            return True  # JPEG
        if h.startswith(b"\x89PNG\r\n\x1a\n"):
            return True  # PNG
        if h[:2] == b"BM":
            return True  # BMP
        return False
    except Exception:
        return False


def is_image_file(p: Path):
    if p.suffix.lower() in IMAGE_EXTS:
        return True
    return _quick_header_check(p)

if not TRAIN_DIR.exists():
    print(f"Train directory not found: {TRAIN_DIR}")
    raise SystemExit(1)

class_counts = {}
corrupted = []

for class_dir in sorted(TRAIN_DIR.iterdir()):
    if not class_dir.is_dir():
        continue
    files = [p for p in class_dir.iterdir() if p.is_file() and is_image_file(p)]
    class_name = class_dir.name
    class_counts[class_name] = len(files)
    sample = files[:5]
    summary_lines.append(f"Class: {class_name} → {len(files)} files; sample: {[p.name for p in sample]}")

    for p in files:
        ok = True
        if PIL_AVAILABLE:
            try:
                with Image.open(p) as im:
                    im.verify()
            except Exception:
                ok = False
        else:
            try:
                if not _quick_header_check(p):
                    ok = False
            except Exception:
                ok = False
        if not ok:
            corrupted.append(str(p.relative_to(ROOT)))
        manifest_rows.append((str(p.relative_to(Path(__file__).resolve().parents[2])), class_name))

# Write manifest.csv (path relative to repo root)
with open(OUT_MANIFEST, 'w', newline='', encoding='utf-8') as mf:
    writer = csv.writer(mf)
    writer.writerow(['filepath','label'])
    for row in manifest_rows:
        writer.writerow(row)

summary_lines.insert(0, f"Total classes: {len(class_counts)}")
summary_lines.insert(1, f"Total images: {sum(class_counts.values())}")
summary_lines.append(f"Corrupted files: {len(corrupted)}")
if corrupted:
    summary_lines.append('Examples of corrupted (relative to script):')
    summary_lines.extend(corrupted[:20])

# Save summary
with open(OUT_SUMMARY, 'w', encoding='utf-8') as out:
    out.write('\n'.join(summary_lines))

# Also print summary
print('\n'.join(summary_lines))
print()
print(f"Manifest written to: {OUT_MANIFEST}")
print(f"Summary written to: {OUT_SUMMARY}")
