"""
Prepare the Stage B dataset from the pre-converted Fusion PNG dataset.

Source: Kaggle `marcesalas/fusion-dataset-59-malware-families-in-png-format`
(2.24 GB, 224x224 PNGs, 59 families across train/valid/test, combines
BIG 2015 + Malimg + MaleVis).

This script pulls out only the 7 BIG 2015 families MalVision targets, merges the
two Kelihos versions into one class, and lays them out as

    data/processed/<family>/*.png

so `src/train.py` can build its own stratified 70/15/15 split (random_state=42)
from there. `src/image_gen.py` is not needed — the images are already 224x224 PNG.

Usage:
    # after: kaggle datasets download -d marcesalas/fusion-dataset-59-malware-families-in-png-format
    #        unzip the archive somewhere, e.g. ~/Downloads/fusion
    python src/prepare_big2015_fusion.py ~/Downloads/fusion
    python src/prepare_big2015_fusion.py ~/Downloads/fusion --processed_dir data/processed
"""

import argparse
import shutil
from collections import defaultdict
from pathlib import Path

# Fusion source folder name  ->  MalVision class name.
# Kelihos_ver1 + Kelihos_ver3 collapse into a single "Kelihos" class.
FAMILY_MAP = {
    'Ramnit':          'Ramnit',
    'Lollipop':        'Lollipop',
    'Kelihos_ver1':    'Kelihos',
    'Kelihos_ver3':    'Kelihos',
    'Vundo':           'Vundo',
    'Tracur':          'Tracur',
    'Obfuscator.ACY':  'Obfuscator.ACY',
    'Gatak':           'Gatak',
}


def find_family_dirs(src_root):
    """Return {source_family_name: [dir, ...]} for every target family folder
    found anywhere under src_root (handles all_malware/{train,valid,test}/<fam>)."""
    src_root = Path(src_root)
    found = defaultdict(list)
    for path in src_root.rglob('*'):
        if path.is_dir() and path.name in FAMILY_MAP:
            found[path.name].append(path)
    return found


def unique_dest(dest_dir, src_file, tag):
    """Collision-safe destination path. Fusion filenames are unique within a
    family, but merging Kelihos versions and flattening train/valid/test could
    still collide, so we prefix with a short tag (split name) and fall back to a
    counter if needed."""
    stem, suffix = src_file.stem, src_file.suffix
    candidate = dest_dir / f'{tag}_{stem}{suffix}'
    if not candidate.exists():
        return candidate
    i = 1
    while True:
        candidate = dest_dir / f'{tag}_{stem}_{i}{suffix}'
        if not candidate.exists():
            return candidate
        i += 1


def prepare(src_root, processed_dir):
    processed_dir = Path(processed_dir)
    found = find_family_dirs(src_root)

    missing = sorted(set(FAMILY_MAP) - set(found))
    if missing:
        print(f"WARNING: these source families were not found under {src_root}:")
        for m in missing:
            print(f"  - {m}")
        print("Check that the archive is fully extracted and the path is correct.\n")

    counts = defaultdict(int)
    for src_name, dirs in sorted(found.items()):
        dest_name = FAMILY_MAP[src_name]
        dest_dir = processed_dir / dest_name
        dest_dir.mkdir(parents=True, exist_ok=True)

        for d in dirs:
            # tag = split name (train/valid/test) when present, else parent dir name
            tag = d.parent.name or 'x'
            for png in d.glob('*.png'):
                shutil.copy2(png, unique_dest(dest_dir, png, tag))
                counts[dest_name] += 1
        print(f"  {src_name:16s} -> {dest_name:16s}  (class total now: {counts[dest_name]})")

    print("\n=== Stage B dataset ready ===")
    total = 0
    for fam in sorted(set(FAMILY_MAP.values())):
        print(f"  {fam:16s}: {counts[fam]:6d} images")
        total += counts[fam]
    print(f"  {'TOTAL':16s}: {total:6d} images")
    print(f"\nWritten to: {processed_dir}/<family>/")
    print("Next: python src/train.py   (it builds the stratified split from data/processed/)")
    return counts


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('src_root',
                        help='Path to the extracted Fusion dataset (contains the family folders)')
    parser.add_argument('--processed_dir', default='data/processed',
                        help='Output dir (default: data/processed)')
    args = parser.parse_args()

    prepare(args.src_root, args.processed_dir)
