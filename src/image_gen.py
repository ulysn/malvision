import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def binary_to_image(filepath, size=224):
    with open(filepath, 'rb') as f:
        bytes_data = np.frombuffer(f.read(), dtype=np.uint8)

    if len(bytes_data) == 0:
        return np.zeros((size, size), dtype=np.uint8)

    side = int(np.sqrt(len(bytes_data))) + 1
    padded = np.pad(bytes_data, (0, side * side - len(bytes_data)))
    img = padded.reshape(side, side)
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def process_dataset(raw_dir, processed_dir, size=224, extensions=None):
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)

    if extensions is None:
        extensions = {'.exe', '.dll', '.bytes', '.bin', ''}

    stats = {'processed': 0, 'skipped': 0, 'errors': 0}

    family_dirs = sorted(d for d in raw_dir.iterdir() if d.is_dir())
    if not family_dirs:
        print(f"No family subdirectories found in {raw_dir}")
        return stats

    for family_dir in family_dirs:
        out_dir = processed_dir / family_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)

        files = [f for f in family_dir.iterdir() if f.is_file()]
        print(f"  {family_dir.name}: {len(files)} files")

        for fp in tqdm(files, desc=family_dir.name, leave=False):
            if extensions and fp.suffix.lower() not in extensions:
                stats['skipped'] += 1
                continue

            out_path = out_dir / (fp.stem + '.png')
            if out_path.exists():
                stats['skipped'] += 1
                continue

            try:
                img = binary_to_image(str(fp), size=size)
                cv2.imwrite(str(out_path), img)
                stats['processed'] += 1
            except Exception as e:
                print(f"    Error on {fp.name}: {e}")
                stats['errors'] += 1

    return stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert malware binaries to grayscale images')
    parser.add_argument('raw_dir', help='data/raw — one subdir per family')
    parser.add_argument('processed_dir', help='data/processed — output PNGs')
    parser.add_argument('--size', type=int, default=224)
    args = parser.parse_args()

    stats = process_dataset(args.raw_dir, args.processed_dir, size=args.size)
    print(f"\nDone — processed: {stats['processed']}  skipped: {stats['skipped']}  errors: {stats['errors']}")
