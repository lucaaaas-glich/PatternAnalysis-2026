"""Prepare localisation labels from original binary masks, not school grey masks.

Verify archive provenance, pair IDs and the expected 2x source dimensions.
Resize masks with NEAREST, recheck binary/nonempty content, then derive boxes.
No source is changed, image is copied, split is made, or diagnosis inferred.
"""

import argparse
import csv
import hashlib
import io
import json
import os
import random
import tempfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

from audit_data import index_files
from dataset import bbox_to_yolo, mask_to_bbox, preview

REFERENCE_URL = 'https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1_Training_GroundTruth.zip'
REFERENCE_SHA256 = '99f8b2bb3c4d6af483362010715f7e7d5d122d9f6c02cac0e0d15bef77c7604c'
REFERENCE_PATH = Path('data/reference/ISIC2018_Task1_Training_GroundTruth.zip')
SCHOOL_IMAGES = Path('/home/groups/comp3710/ISIC2018/ISIC2018_Task1-2_Training_Input_x2')


def file_sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def download_reference(destination, expected_hash=REFERENCE_SHA256):
    """Use a verified cache or download the official small mask ZIP atomically."""
    destination = Path(destination)
    if destination.exists():
        if file_sha256(destination) != expected_hash:
            raise ValueError('Existing reference checksum differs; do not overwrite it')
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix='isic-mask-download-',
                                         suffix='.part', delete=False) as handle:
            temporary = Path(handle.name)
            with urllib.request.urlopen(REFERENCE_URL, timeout=60) as response:
                total = 0
                while chunk := response.read(1024 * 1024):
                    total += len(chunk)
                    if total > 40 * 1024 * 1024:
                        raise ValueError('Reference exceeds expected small-mask download size')
                    handle.write(chunk)
        if file_sha256(temporary) != expected_hash:
            raise ValueError('Downloaded reference checksum mismatch; not accepted')
        # Same-directory hard link publishes the complete verified file atomically,
        # and fails instead of replacing a cache created by another process.
        os.link(temporary, destination)
        return destination
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()  # Only this invocation's own incomplete/generated temp file.


def nearest_binary_mask(original, image_size, expected_scale=2):
    """Guard the observed _x2 geometry and preserve the official 0/255 encoding."""
    values = np.asarray(original)
    if original.mode != 'L' or values.dtype != np.uint8 or values.ndim != 2:
        raise ValueError('Official reference must be an L-mode uint8 mask')
    if not set(np.unique(values).tolist()).issubset({0, 255}) or not np.any(values):
        raise ValueError('Official mask must be nonempty and strictly 0/255')
    width, height = image_size
    if width <= 0 or height <= 0 or expected_scale <= 0:
        raise ValueError('Image dimensions and expected scale must be positive')
    # Allow one original pixel for odd dimensions rounded during 2x downsampling.
    if abs(original.width - expected_scale * width) > 1 or abs(original.height - expected_scale * height) > 1:
        raise ValueError('Source/image dimensions do not match expected 2x geometry')
    resized = original.resize(image_size, Image.Resampling.NEAREST)
    box = mask_to_bbox(np.asarray(resized))  # Also catches small regions erased by downsampling.
    return resized, box


def preparation_preview_ids(records, samples=6, seed=3710):
    """Include smallest/largest boxes, then seeded random samples; deduplicate."""
    samples = min(max(samples, 0), len(records))
    if not samples:
        return []
    ordered = sorted(records, key=lambda record: (record['box_area_fraction'], record['image_id']))
    chosen = [ordered[0]['image_id']]
    if samples > 1 and ordered[-1]['image_id'] != chosen[0]:
        chosen.append(ordered[-1]['image_id'])
    remaining = sorted(record['image_id'] for record in records if record['image_id'] not in chosen)
    chosen.extend(random.Random(seed).sample(remaining, samples - len(chosen)))
    return chosen


def prepare_labels(images_dir, reference_zip, output_dir, expected_pairs=2594,
                   expected_hash=REFERENCE_SHA256, samples=6, seed=3710):
    images_dir, reference_zip, output_dir = map(Path, (images_dir, reference_zip, output_dir))
    if not images_dir.is_dir() or not reference_zip.is_file():
        raise ValueError('School images and verified reference ZIP must exist')
    if output_dir.exists():
        raise ValueError('Output already exists; preserve prior evidence and use a new directory')
    resolved = output_dir.resolve()
    if resolved == images_dir.resolve() or images_dir.resolve() in resolved.parents:
        raise ValueError('Output must be outside original image directory')
    digest = file_sha256(reference_zip)
    if digest != expected_hash:
        raise ValueError('Official reference checksum mismatch')
    images = index_files(images_dir, {'.jpg', '.jpeg'})
    if len(images) != expected_pairs or any(len(paths) != 1 for paths in images.values()):
        raise ValueError(f'Expected {expected_pairs} unique school image IDs; found {len(images)}')
    records, issues = [], []
    with zipfile.ZipFile(reference_zip) as archive:
        masks = {}
        for member in archive.namelist():
            filename = PurePosixPath(member).name
            if filename.endswith('_segmentation.png'):
                identifier = filename.removesuffix('_segmentation.png')
                if identifier in masks:
                    raise ValueError(f'Duplicate official mask ID: {identifier}')
                masks[identifier] = member
        if set(images) != set(masks):
            raise ValueError('School image IDs and official mask IDs must match exactly')
        for identifier in sorted(images):
            try:
                with Image.open(images[identifier][0]) as image, Image.open(io.BytesIO(archive.read(masks[identifier]))) as original:
                    image.load()  # Detect corrupt JPEGs rather than reading headers only.
                    resized, box = nearest_binary_mask(original, image.size)
                    yolo = bbox_to_yolo(box, *image.size)
                    records.append(dict(image_id=identifier, image_path=str(images[identifier][0].resolve()),
                                        width=image.width, height=image.height,
                                        original_mask_width=original.width, original_mask_height=original.height,
                                        left=box[0], top=box[1], right=box[2], bottom=box[3],
                                        center_x=yolo[0], center_y=yolo[1], box_width=yolo[2], box_height=yolo[3],
                                        box_area_fraction=yolo[2] * yolo[3]))
            except (OSError, ValueError) as error:
                issues.append({'image_id': identifier, 'error': str(error)})
        # All input pairs must pass before ANY training label is exported.
        output_dir.mkdir(parents=True)
        selected = preparation_preview_ids(records, samples, seed) if not issues else []
        summary = {
            'purpose': 'Official-mask localisation label preparation, NOT training/evaluation',
            'reference_url': REFERENCE_URL, 'reference_archive_sha256': digest,
            'source_mask_count': len(masks), 'image_count': len(images), 'valid_pairs': len(records),
            'all_pairs_validated': not issues and len(records) == expected_pairs,
            'issues_count': len(issues), 'issue_examples': issues[:5],
            'mask_resize_rule': 'NEAREST to school photo dimensions; original 0/255 only',
            'expected_downsample_factor': 2, 'original_dimension_rounding_tolerance_pixels': 1,
            'class_names': {'0': 'lesion'}, 'class_scope': 'Localisation only; not benign/malignant',
            'source_images_modified': False, 'source_masks_modified': False,
            'school_preprocessing_provenance_verified': False,
            'patient_independence_verified': False, 'diagnosis_mapping_verified': False,
            'split_created': False, 'labels_exported': 0, 'seed': seed,
            'preview_ids': selected, 'preview_selection': 'Area extremes plus seeded random IDs',
            'min_box_area_fraction': min((r['box_area_fraction'] for r in records), default=None),
            'max_box_area_fraction': max((r['box_area_fraction'] for r in records), default=None),
        }
        if issues:
            (output_dir / 'issues.json').write_text(json.dumps(issues, indent=2) + '\n', encoding='utf-8')
        else:
            labels_dir, previews_dir = output_dir / 'labels', output_dir / 'previews'
            labels_dir.mkdir()
            previews_dir.mkdir()
            for record in records:
                values = [record[k] for k in ('center_x', 'center_y', 'box_width', 'box_height')]
                (labels_dir / (record['image_id'] + '.txt')).write_text(
                    '0 ' + ' '.join(f'{value:.8f}' for value in values) + '\n', encoding='utf-8')
            with (output_dir / 'annotations.csv').open('w', encoding='utf-8', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(records[0]))
                writer.writeheader()
                writer.writerows(records)
            record_map = {record['image_id']: record for record in records}
            for identifier in selected:
                record = record_map[identifier]
                with Image.open(images[identifier][0]) as image, Image.open(io.BytesIO(archive.read(masks[identifier]))) as original:
                    resized, box = nearest_binary_mask(original, image.size)
                    preview(image.convert('RGB'), resized, box, identifier, previews_dir / (identifier + '.png'),
                            mask_title='2. Official mask (NEAREST)')
            summary['labels_exported'] = len(records)
        (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    # Compact log; do not print thousands of repeated errors.
    print(f"Validated {summary['valid_pairs']}/{summary['image_count']} pairs; "
          f"labels={summary['labels_exported']}; issues={summary['issues_count']}")
    print(f'Summary: {output_dir / "summary.json"}')
    for issue in summary['issue_examples']:
        print(issue['image_id'], issue['error'])
    print('No model, diagnosis mapping, or patient-independent split has been created.')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', type=Path, default=SCHOOL_IMAGES)
    parser.add_argument('--reference-zip', type=Path, default=REFERENCE_PATH)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--download-reference-only', action='store_true')
    args = parser.parse_args()
    if args.download_reference_only:
        print('Checking/downloading the small official mask reference (not photos)...', flush=True)
        download_reference(args.reference_zip)
        print(f'Official reference checksum verified: {args.reference_zip}')
    else:
        if args.output is None:
            parser.error('--output is required for full preparation')
        report = prepare_labels(args.images, args.reference_zip, args.output)
        raise SystemExit(0 if report['all_pairs_validated'] else 1)
