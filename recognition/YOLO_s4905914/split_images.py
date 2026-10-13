"""Reproducible IMAGE-level split, explicitly not a lesion/patient split.

Keeps all images. The singleton Indeterminate class is training-only and
cannot be evaluated on held-out data. Sources are never modified.
"""

import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

CLASSES = {'Benign': 0, 'Malignant': 1, 'Indeterminate': 2}


def keyed(rows, field):
    result = {}
    for row in rows:
        key = row[field].strip()
        if not key or key in result:
            raise ValueError('Missing or duplicate image ID')
        result[key] = row
    return result


def assign_images(annotations, metadata, seed=3710):
    images, meta = keyed(annotations, 'image_id'), keyed(metadata, 'isic_id')
    if not images or images.keys() != meta.keys():
        raise ValueError('Annotation and metadata image IDs must match exactly')
    groups = {name: [] for name in CLASSES}
    for identifier in sorted(images):
        category = meta[identifier]['diagnosis_1'].strip()
        if category not in CLASSES:
            raise ValueError('Unknown or missing diagnosis; no automatic remapping')
        coordinates = [float(images[identifier][key]) for key in
                       ('center_x', 'center_y', 'box_width', 'box_height')]
        x, y, w, h = coordinates
        if not all(0 <= value <= 1 for value in coordinates) or w <= 0 or h <= 0:
            raise ValueError('Invalid normalised box')
        if min(x-w/2, y-h/2) < -1e-7 or max(x+w/2, y+h/2) > 1+1e-7:
            raise ValueError('Box outside image')
        groups[category].append(identifier)
    if len(groups['Indeterminate']) != 1:
        raise ValueError('This protocol requires exactly one Indeterminate image')
    splits = {name: [] for name in ('train', 'val', 'test')}
    rng = random.Random(seed)
    for category in ('Benign', 'Malignant'):
        ids = groups[category]
        if len(ids) < 7:
            raise ValueError('Too few images for this stratified protocol')
        rng.shuffle(ids)
        heldout = int(len(ids) * 0.15)
        splits['val'].extend(ids[:heldout])
        splits['test'].extend(ids[heldout:2*heldout])
        splits['train'].extend(ids[2*heldout:])
    splits['train'].extend(groups['Indeterminate'])
    return {name: sorted(ids) for name, ids in splits.items()}, images, meta


def prepare(annotations_path, metadata_path, output, seed=3710):
    annotations_path, metadata_path, output = map(Path, (annotations_path, metadata_path, output))
    if output.exists():
        raise ValueError('Preserve existing split; choose a new output directory')
    if any(output.resolve() == p.parent.resolve() or p.parent.resolve() in output.resolve().parents
           for p in (annotations_path, metadata_path)):
        raise ValueError('Output must be outside input directories')
    with annotations_path.open(encoding='utf-8-sig', newline='') as handle:
        annotations = list(csv.DictReader(handle))
    with metadata_path.open(encoding='utf-8-sig', newline='') as handle:
        metadata = list(csv.DictReader(handle))
    splits, images, meta = assign_images(annotations, metadata, seed)
    image_sets = [set(ids) for ids in splits.values()]
    assert len(set.union(*image_sets)) == len(images)
    assert all(not a & b for i, a in enumerate(image_sets) for b in image_sets[i+1:])
    counts = {name: dict(Counter(meta[i]['diagnosis_1'].strip() for i in ids))
              for name, ids in splits.items()}
    lesion_sets = {name: {meta[i].get('lesion_id', '').strip() for i in ids
                          if meta[i].get('lesion_id', '').strip()} for name, ids in splits.items()}
    overlaps = {f'{a}_{b}': len(lesion_sets[a] & lesion_sets[b])
                for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]}
    hashes = {}
    for name, path in [('annotations', annotations_path), ('metadata', metadata_path)]:
        with path.open('rb') as handle:
            hashes[name] = hashlib.file_digest(handle, 'sha256').hexdigest()
    summary = {'split_unit': 'image', 'seed': seed, 'source_sha256': hashes,
               'image_count': len(images), 'split_counts': {k: len(v) for k, v in splits.items()},
               'class_counts': counts, 'classes': CLASSES,
               'image_id_overlap': 0, 'known_lesion_overlap_counts': overlaps,
               'missing_lesion_id': sum(not row.get('lesion_id', '').strip() for row in meta.values()),
               'patient_independence_verified': False, 'lesion_independence_verified': False,
               'policy': 'User-selected image-level split, departing from instructor lesion-group guidance',
               'limitations': ['Known and unknown lesion leakage can inflate held-out metrics.',
                               'Indeterminate singleton retained in train only; no held-out class assessment.',
                               'No image content duplicate check or image-path existence check performed.',
                               'Original localisation labels unchanged; new labels use three diagnosis classes.']}
    output.mkdir(parents=True)
    for name, ids in splits.items():
        labels = output / name / 'labels'
        labels.mkdir(parents=True)
        with (output / f'{name}.csv').open('x', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=['image_id', 'image_path', 'class_id', 'diagnosis'])
            writer.writeheader()
            for identifier in ids:
                row, category = images[identifier], meta[identifier]['diagnosis_1'].strip()
                writer.writerow({'image_id': identifier, 'image_path': row['image_path'],
                                 'class_id': CLASSES[category], 'diagnosis': category})
                values = ' '.join(row[k] for k in ('center_x', 'center_y', 'box_width', 'box_height'))
                (labels / f'{identifier}.txt').write_text(f'{CLASSES[category]} {values}\n', encoding='utf-8')
    (output / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=3710)
    args = parser.parse_args()
    prepare(args.annotations, args.metadata, args.output, args.seed)
