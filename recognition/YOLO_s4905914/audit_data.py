"""Read-only input audit before dataset splitting; no labels or split are created.

Only uniquely paired images/masks are decoded. CSV headers are inventoried,
not patient records. Finding a CSV does not verify patient independence.
"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from dataset import mask_to_bbox


def index_files(directory, suffixes, mask=False):
    index = defaultdict(list)
    for path in sorted(directory.rglob('*')):
        if path.is_file() and path.suffix.lower() in suffixes:
            identifier = path.stem.removesuffix('_segmentation') if mask else path.stem
            index[identifier].append(path)
    return index


def metadata_headers(root):
    """Record candidate CSV names/headers only; do not guess patient or class IDs."""
    candidates = []
    for path in sorted(root.rglob('*.csv')):
        record = {'file': str(path.relative_to(root))}
        try:
            with path.open(encoding='utf-8-sig', newline='') as handle:
                line = handle.readline(8193)
            if len(line) > 8192:
                raise ValueError('Header exceeds 8192 characters; manual review needed')
            record['header'] = next(csv.reader([line]), [])
        except (OSError, UnicodeError, csv.Error, ValueError) as error:
            record['error'] = str(error)
        candidates.append(record)
    return candidates


def audit_pairs(images_dir, masks_dir, metadata_root=None):
    images_dir, masks_dir = Path(images_dir), Path(masks_dir)
    for directory in (images_dir, masks_dir):
        if not directory.is_dir():
            raise ValueError(f'Input directory does not exist: {directory}')
    if metadata_root is not None and not Path(metadata_root).is_dir():
        raise ValueError('Metadata root must be an existing directory')
    images = index_files(images_dir, {'.jpg', '.jpeg', '.png'})
    masks = index_files(masks_dir, {'.png'}, mask=True)
    missing = sorted(images.keys() - masks.keys())
    orphan = sorted(masks.keys() - images.keys())
    duplicates_images = sorted(key for key, paths in images.items() if len(paths) != 1)
    duplicates_masks = sorted(key for key, paths in masks.items() if len(paths) != 1)
    unique_pairs = sorted(key for key in images.keys() & masks.keys()
                          if len(images[key]) == len(masks[key]) == 1)
    image_sizes, mask_sizes = Counter(), Counter()
    issues, areas = [], []
    for identifier in unique_pairs:
        try:
            with Image.open(images[identifier][0]) as image, Image.open(masks[identifier][0]) as mask:
                # Decode both files, not merely their headers, to detect corrupt data.
                image.load()
                values = np.asarray(mask)
                image_sizes[f'{image.width}x{image.height}'] += 1
                mask_sizes[f'{mask.width}x{mask.height}'] += 1
                if image.size != mask.size:
                    raise ValueError('Image/mask size mismatch')
                left, top, right, bottom = mask_to_bbox(values)
                areas.append((right - left) * (bottom - top) / (image.width * image.height))
        except (OSError, ValueError) as error:
            issues.append({'image_id': identifier, 'error': str(error)})
    passed = bool(images and masks and unique_pairs) and not any(
        (missing, orphan, duplicates_images, duplicates_masks, issues))
    return {
        'purpose': 'Input integrity audit only; not model evaluation or dataset splitting',
        'images_dir': str(images_dir), 'masks_dir': str(masks_dir),
        'image_files': sum(map(len, images.values())),
        'mask_files': sum(map(len, masks.values())),
        'unique_pairs_checked': len(unique_pairs), 'valid_pairs': len(areas),
        'missing_mask_ids': missing, 'orphan_mask_ids': orphan,
        'duplicate_image_ids': duplicates_images, 'duplicate_mask_ids': duplicates_masks,
        'paired_image_dimensions': dict(sorted(image_sizes.items())),
        'paired_mask_dimensions': dict(sorted(mask_sizes.items())), 'pair_issues': issues,
        'min_box_area_fraction': min(areas) if areas else None,
        'max_box_area_fraction': max(areas) if areas else None,
        'pair_checks_passed': passed,
        'metadata_csv_candidates': metadata_headers(Path(metadata_root)) if metadata_root else [],
        'diagnosis_mapping_verified': False, 'patient_independence_verified': False,
        'split_created': False,
        'limitations': [
            'Matching filenames do not establish patient independence.',
            'CSV headers alone do not establish diagnosis labels or patient identity.',
            'Dimensions alone do not establish what a directory suffix such as _x2 means.',
            'No duplicate-content check or cross-split overlap check has been performed.',
        ],
    }


def save_report(report, destination, protected_dirs):
    """Prevent writing the audit report into any protected input tree."""
    destination = Path(destination).resolve()
    if any(destination == Path(root).resolve()
           or Path(root).resolve() in destination.parents for root in protected_dirs):
        raise ValueError('Report destination must be outside all original input trees')
    if destination.exists():
        raise ValueError('Report already exists; choose a new filename to preserve earlier evidence')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', type=Path, required=True)
    parser.add_argument('--masks', type=Path, required=True)
    parser.add_argument('--metadata-root', type=Path)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    result = audit_pairs(args.images, args.masks, args.metadata_root)
    if args.report:
        protected = [args.images, args.masks]
        if args.metadata_root:
            protected.append(args.metadata_root)
        save_report(result, args.report, protected)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['pair_checks_passed'] else 1)
