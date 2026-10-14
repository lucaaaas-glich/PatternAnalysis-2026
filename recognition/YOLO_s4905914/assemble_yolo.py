"""Validate split manifests, then link school images into a YOLO runtime tree.

No training, network downloads or changes to original school images.
Requires image-level split policy acknowledgement; never claims group independence.
"""

import argparse
import csv
import json
import math
import shutil
from pathlib import Path

SPLITS = ('train', 'val', 'test')
CLASS_NAMES = ['Benign', 'Malignant', 'Indeterminate']


def validate_split(split_root):
    split_root = Path(split_root)
    summary = json.loads((split_root / 'summary.json').read_text(encoding='utf-8'))
    if summary['split_unit'] != 'image' or summary['classes'] != dict(zip(CLASS_NAMES, range(3))):
        raise ValueError('Unexpected split protocol or class mapping')
    records, seen = {}, set()
    for name in SPLITS:
        with (split_root / f'{name}.csv').open(encoding='utf-8', newline='') as handle:
            rows = list(csv.DictReader(handle))
        if not rows or len(rows) != summary['split_counts'][name]:
            raise ValueError('Manifest count does not match summary')
        for row in rows:
            identifier, class_id = row['image_id'], int(row['class_id'])
            if not identifier.startswith('ISIC_') or not identifier[5:].isdigit() or len(identifier) != 12:
                raise ValueError('Invalid image identifier')
            if identifier in seen:
                raise ValueError('Image ID overlap or duplicate')
            seen.add(identifier)
            if class_id not in range(3) or CLASS_NAMES[class_id] != row['diagnosis']:
                raise ValueError('Class mismatch')
            image = Path(row['image_path'])
            if not image.is_absolute() or not image.is_file() or image.stem != identifier or image.suffix.lower() not in {'.jpg', '.jpeg'}:
                raise ValueError(f'Missing or mismatched school image: {identifier}')
            label = split_root / name / 'labels' / f'{identifier}.txt'
            lines = label.read_text(encoding='utf-8').strip().splitlines()
            tokens = lines[0].split() if len(lines) == 1 else []
            if len(tokens) != 5 or tokens[0] != str(class_id):
                raise ValueError('Label must contain one box with matching class')
            x, y, w, h = map(float, tokens[1:])
            if not all(math.isfinite(v) and 0 <= v <= 1 for v in (x, y, w, h)) or min(w, h) <= 0:
                raise ValueError('Invalid normalised coordinates')
            if min(x-w/2, y-h/2) < -1e-7 or max(x+w/2, y+h/2) > 1+1e-7:
                raise ValueError('Box outside image')
        if len(list((split_root / name / 'labels').glob('*.txt'))) != len(rows):
            raise ValueError('Extra or missing label files')
        records[name] = rows
    if len(seen) != summary['image_count']:
        raise ValueError('Total image count mismatch')
    return records, summary


def assemble(split_root, output, acknowledge_image_split=False):
    split_root, output = Path(split_root).resolve(), Path(output).resolve()
    if not acknowledge_image_split:
        raise ValueError('Explicit image-level leakage acknowledgement required')
    if output.exists():
        raise ValueError('Output exists; preserve previous evidence')
    if output == split_root or split_root in output.parents or output in split_root.parents:
        raise ValueError('Runtime tree must not contain or be within split inputs')
    records, summary = validate_split(split_root)  # All checks before writes.
    if any(output == Path(r['image_path']).parent or output in Path(r['image_path']).parents
           or Path(r['image_path']).parent in output.parents for rows in records.values() for r in rows):
        raise ValueError('Runtime output must be separate from source image trees')
    output.mkdir(parents=True)
    for name, rows in records.items():
        images_dir, labels_dir = output / 'images' / name, output / 'labels' / name
        images_dir.mkdir(parents=True)
        labels_dir.mkdir(parents=True)
        for row in rows:
            image = Path(row['image_path'])
            (images_dir / image.name).symlink_to(image)
            shutil.copyfile(split_root / name / 'labels' / f"{row['image_id']}.txt",
                            labels_dir / f"{row['image_id']}.txt")
    # JSON quoting is also valid YAML scalar/list syntax; avoids a YAML dependency.
    yaml_text = 'path: ' + json.dumps(str(output)) + '\ntrain: images/train\nval: images/val\ntest: images/test\n'
    yaml_text += 'names: ' + json.dumps(CLASS_NAMES) + '\n'
    (output / 'data.yaml').write_text(yaml_text, encoding='utf-8')
    report = dict(summary, runtime_path=str(output), sources_modified=False,
                  model_trained=False, images_linked=sum(map(len, records.values())))
    (output / 'assembly.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(f"Linked {report['images_linked']} images; YOLO config: {output / 'data.yaml'}")
    print('IMAGE-level split: known/unknown lesion leakage remains. No training performed.')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--acknowledge-image-level-split', action='store_true')
    args = parser.parse_args()
    assemble(args.split_root, args.output, args.acknowledge_image_level_split)
