"""Audit paired ISIC images/masks and derive single-lesion YOLO annotations.

This is annotation preparation, not model inference. No split is created here.
Bounding boxes use pixel edges: left/top inclusive, right/bottom exclusive.
"""

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


def mask_to_bbox(mask):
    """Return (left, top, right, bottom), enclosing ALL nonzero mask pixels.

    A disconnected mask still describes one annotated lesion in this task.
    Reject empty/nonbinary masks rather than silently making a negative label.
    """
    values = np.asarray(mask)
    if values.ndim != 2 or values.size == 0:
        raise ValueError('Mask must be a nonempty two-dimensional array')
    if not set(np.unique(values).tolist()).issubset({0, 1, 255}):
        raise ValueError('Expected a binary mask with values 0/1 or 0/255')
    rows, cols = np.nonzero(values)
    if not len(rows):
        raise ValueError('Empty lesion mask')
    return int(cols.min()), int(rows.min()), int(cols.max()) + 1, int(rows.max()) + 1


def bbox_to_yolo(box, width, height):
    """Convert pixel edges to normalised centre-x, centre-y, width, height."""
    left, top, right, bottom = box
    if not (width > 0 and height > 0 and 0 <= left < right <= width
            and 0 <= top < bottom <= height):
        raise ValueError('Box must have positive area within the image')
    return ((left + right) / (2 * width), (top + bottom) / (2 * height),
            (right - left) / width, (bottom - top) / height)


def preview(image, mask, box, name, destination):
    """Three panels: input, human mask, derived ground-truth box."""
    boxed = image.copy()
    left, top, right, bottom = box
    ImageDraw.Draw(boxed).rectangle((left, top, right - 1, bottom - 1),
                                    outline='#00ff88', width=max(3, image.width // 200))
    canvas = Image.new('RGB', (1200, 370), '#f6f7fa')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=19)
    for i, (panel, title) in enumerate(zip(
            (image, mask.convert('RGB'), boxed),
            ('1. Original image', '2. Human mask (ground truth)', '3. Derived box (NOT prediction)'))):
        fitted = ImageOps.contain(panel, (390, 290))
        canvas.paste(fitted, (400 * i + (400 - fitted.width) // 2,
                             45 + (290 - fitted.height) // 2))
        draw.text((400 * i + 8, 12), title, font=font, fill='#182438')
    draw.text((12, 340), name + ' | class 0 = lesion', font=font, fill='#182438')
    canvas.save(destination)


def inspect_pairs(images_dir, masks_dir, output_dir, samples=6, seed=3710):
    images_dir, masks_dir, output_dir = map(Path, (images_dir, masks_dir, output_dir))
    if not images_dir.is_dir() or not masks_dir.is_dir():
        raise ValueError('Both input directories must exist')
    images = sorted(p for p in images_dir.rglob('*')
                    if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg'})
    masks = sorted(masks_dir.rglob('*_segmentation.png'))
    if not images or not masks:
        raise ValueError('No ISIC image/mask pairs found')
    image_ids = [p.stem for p in images]
    mask_ids = [p.stem.removesuffix('_segmentation') for p in masks]
    if len(set(image_ids)) != len(image_ids) or len(set(mask_ids)) != len(mask_ids):
        raise ValueError('Duplicate image or mask identifiers')
    mask_map = dict(zip(mask_ids, masks))
    missing = sorted(set(image_ids) - set(mask_ids))
    orphan = sorted(set(mask_ids) - set(image_ids))
    if missing or orphan:
        raise ValueError(f'Unpaired files: missing masks={missing}, orphan masks={orphan}')
    # Validate the entire input before writing any labels.
    records = []
    for path in images:
        with Image.open(path) as src, Image.open(mask_map[path.stem]) as raw_mask:
            image, mask = src.convert('RGB'), raw_mask.convert('L')
            if image.size != mask.size:
                raise ValueError(f'{path.stem}: image/mask size mismatch')
            box = mask_to_bbox(mask)
            yolo = bbox_to_yolo(box, *image.size)
            records.append(dict(image_id=path.stem, width=image.width, height=image.height,
                                left=box[0], top=box[1], right=box[2], bottom=box[3],
                                center_x=yolo[0], center_y=yolo[1], box_width=yolo[2],
                                box_height=yolo[3], box_area_fraction=yolo[2] * yolo[3]))
    # Generated files belong in an ignored data/outputs directory.
    labels_dir, previews_dir = output_dir / 'labels', output_dir / 'previews'
    labels_dir.mkdir(parents=True, exist_ok=True)
    previews_dir.mkdir(parents=True, exist_ok=True)
    for record in records:
        numbers = [record[k] for k in ('center_x', 'center_y', 'box_width', 'box_height')]
        (labels_dir / f"{record['image_id']}.txt").write_text(
            '0 ' + ' '.join(f'{v:.8f}' for v in numbers) + '\n', encoding='utf-8')
    with (output_dir / 'annotations.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summary = dict(images=len(images), masks=len(masks), valid_pairs=len(records),
                   empty_masks=0, missing_masks=0, orphan_masks=0, class_names={'0': 'lesion'},
                   min_box_area_fraction=min(r['box_area_fraction'] for r in records),
                   max_box_area_fraction=max(r['box_area_fraction'] for r in records),
                   split_created=False, patient_independence_verified=False,
                   purpose='ISIC2018 validation annotation inspection; no training or test score',
                   seed=seed)
    (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    image_map = dict(zip(image_ids, images))
    for record in random.Random(seed).sample(records, min(max(samples, 0), len(records))):
        name = record['image_id']
        with Image.open(image_map[name]) as src, Image.open(mask_map[name]) as raw_mask:
            preview(src.convert('RGB'), raw_mask.convert('L'),
                    tuple(record[k] for k in ('left', 'top', 'right', 'bottom')),
                    name, previews_dir / f'{name}.png')
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', type=Path, required=True)
    parser.add_argument('--masks', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=6)
    parser.add_argument('--seed', type=int, default=3710)
    args = parser.parse_args()
    inspect_pairs(args.images, args.masks, args.output, args.samples, args.seed)

