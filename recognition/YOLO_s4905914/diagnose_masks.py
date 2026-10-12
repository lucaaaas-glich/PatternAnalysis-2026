"""Create small-sample grey-mask diagnostics, NOT approved training labels.

Compare >=1, >=128 (candidate), and >=255. Preserve all original files;
the candidate threshold remains provisional until visual/source review.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from audit_data import index_files
from dataset import mask_to_bbox

SCHOOL_ROOT = Path('/home/groups/comp3710/ISIC2018')


def threshold_diagnostics(values, candidate=128):
    """Inspect uint8 greyscale annotations; do not guess other encodings."""
    values = np.asarray(values)
    if values.ndim != 2 or not values.size or values.dtype != np.uint8:
        raise ValueError('Diagnosis requires a nonempty 2D uint8 greyscale mask')
    if isinstance(candidate, bool) or not isinstance(candidate, int) or not 1 <= candidate <= 255:
        raise ValueError('Candidate threshold must be an integer from 1 to 255')
    if values.max() <= 1:
        raise ValueError('0/1 mask encoding: do not apply an 8-bit grey threshold')
    unique = np.unique(values)
    boxes = {}
    for threshold in sorted({1, candidate, 255}):
        binary = (values >= threshold).astype(np.uint8)
        try:
            boxes[str(threshold)] = list(mask_to_bbox(binary))
        except ValueError:
            boxes[str(threshold)] = None  # Threshold erased all positives; record, do not hide.
    return {
        'dtype': str(values.dtype), 'unique_count': len(unique),
        'min_value': int(unique.min()), 'max_value': int(unique.max()),
        'first_values': unique[:12].tolist(),
        'intermediate_fraction': float(np.mean((values > 0) & (values < 255))),
        'candidate_threshold': candidate, 'candidate_rule': 'pixel >= threshold',
        'candidate_has_foreground': boxes[str(candidate)] is not None,
        'boxes_by_threshold': boxes,
    }


def preview_row(image, raw_mask, record):
    """Scientific comparison figure: source, grey mask, candidate, box sensitivity."""
    threshold = record['candidate_threshold']
    binary = Image.fromarray((np.asarray(raw_mask) >= threshold).astype(np.uint8) * 255)
    boxed = image.copy()
    colours = {'1': '#ff9900', str(threshold): '#00ff88', '255': '#e05bff'}
    for key, box in record['boxes_by_threshold'].items():
        if box:
            left, top, right, bottom = box
            ImageDraw.Draw(boxed).rectangle((left, top, right - 1, bottom - 1),
                                            outline=colours[key], width=max(3, image.width // 200))
    row = Image.new('RGB', (1600, 390), '#f6f7fa')
    draw = ImageDraw.Draw(row)
    font = ImageFont.load_default(size=18)
    panels = [(image, '1. Photo', Image.Resampling.LANCZOS),
              (raw_mask, '2. Source grey mask', Image.Resampling.NEAREST),
              (binary, f'3. Candidate: pixel >= {threshold}', Image.Resampling.NEAREST),
              (boxed, '4. Derived boxes, NOT predictions', Image.Resampling.LANCZOS)]
    for column, (panel, title, method) in enumerate(panels):
        fitted = ImageOps.contain(panel.convert('RGB'), (390, 285), method=method)
        row.paste(fitted, (column * 400 + (400 - fitted.width) // 2,
                           45 + (285 - fitted.height) // 2))
        draw.text((column * 400 + 8, 12), title, font=font, fill='#182438')
    draw.text((10, 338), f"{record['image_id']} | intermediate pixels: "
              f"{record['intermediate_fraction']:.2%} | candidate NOT approved", font=font, fill='#182438')
    draw.text((10, 363), f'Boxes: orange >=1; green >={threshold}; magenta >=255. '
              'Overlapping boxes may hide one another.', font=font, fill='#182438')
    return row


def diagnose_samples(images_dir, masks_dir, output_dir, samples=3, candidate=128):
    images_dir, masks_dir, output_dir = map(Path, (images_dir, masks_dir, output_dir))
    if not images_dir.is_dir() or not masks_dir.is_dir():
        raise ValueError('Both original input directories must exist')
    if not 1 <= samples <= 6:
        raise ValueError('Use 1 to 6 samples; this is a small diagnostic, not a full audit')
    destination = output_dir.resolve()
    if any(destination == root.resolve() or root.resolve() in destination.parents
           for root in (images_dir, masks_dir)):
        raise ValueError('Output must be outside original inputs')
    if output_dir.exists():
        raise ValueError('Output already exists; choose a new directory to preserve evidence')
    images = index_files(images_dir, {'.jpg', '.jpeg', '.png'})
    masks = index_files(masks_dir, {'.png'}, mask=True)
    identifiers = sorted(masks)[:samples]
    if not identifiers:
        raise ValueError('No masks found')
    records, rows = [], []
    for identifier in identifiers:
        if len(masks[identifier]) != 1 or len(images.get(identifier, [])) != 1:
            raise ValueError(f'{identifier}: missing/ambiguous image-mask pair')
        with Image.open(images[identifier][0]) as image, Image.open(masks[identifier][0]) as mask:
            if image.size != mask.size or mask.mode != 'L':
                raise ValueError(f'{identifier}: needs size-matched L-mode source mask')
            record = threshold_diagnostics(np.asarray(mask), candidate)
            record.update(image_id=identifier, mode=mask.mode, width=mask.width, height=mask.height,
                          source_mask_sha256=hashlib.sha256(masks[identifier][0].read_bytes()).hexdigest())
            rows.append(preview_row(image.convert('RGB'), mask, record))
            records.append(record)
    # Validate selected samples before writing any outputs.
    output_dir.mkdir(parents=True)
    overview = Image.new('RGB', (1600, 390 * len(rows)))
    for position, row in enumerate(rows):
        overview.paste(row, (0, position * 390))
    overview.save(output_dir / 'overview.png')
    report = {'purpose': 'Candidate label-preprocessing diagnosis, not model evaluation',
              'sampling': 'First sorted mask IDs; not a representative dataset-wide sample',
              'candidate_threshold': candidate, 'threshold_approved': False,
              'source_preprocessing_provenance_verified': False,
              'original_files_modified': False, 'training_labels_exported': False,
              'records': records}
    (output_dir / 'report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    for record in records:
        print(f"{record['image_id']}: unique={record['unique_count']}, "
              f"intermediate={record['intermediate_fraction']:.2%}, "
              f"candidate_box={record['boxes_by_threshold'][str(candidate)]}")
    print(f'Preview: {output_dir / "overview.png"}\nReport: {output_dir / "report.json"}')
    print('Candidate threshold is NOT approved; no original file or training label was changed.')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', type=Path, default=SCHOOL_ROOT / 'ISIC2018_Task1-2_Training_Input_x2')
    parser.add_argument('--masks', type=Path, default=SCHOOL_ROOT / 'ISIC2018_Task1_Training_GroundTruth_x2')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=3)
    parser.add_argument('--candidate-threshold', type=int, default=128)
    args = parser.parse_args()
    diagnose_samples(args.images, args.masks, args.output, args.samples, args.candidate_threshold)
