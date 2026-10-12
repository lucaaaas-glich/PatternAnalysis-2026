"""Compare sample candidate boxes with official masks resized by nearest neighbour.

This is annotation consistency, not model evaluation, and does not approve a
threshold or prove the school's preprocessing algorithm or patient separation.
"""

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

from audit_data import save_report
from dataset import mask_to_bbox

REFERENCE_URL = 'https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1_Training_GroundTruth.zip'


def box_iou(first, second):
    """Pixel-edge boxes use the same exclusive right/bottom convention."""
    if first is None or second is None:
        return None
    for box in (first, second):
        if len(box) != 4 or box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError('Expected a positive-area four-coordinate box')
    width = max(0, min(first[2], second[2]) - max(first[0], second[0]))
    height = max(0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = width * height
    area_a = (first[2] - first[0]) * (first[3] - first[1])
    area_b = (second[2] - second[0]) * (second[3] - second[1])
    return intersection / (area_a + area_b - intersection)


def compare_reference(sample_report, reference_zip):
    sample_report, reference_zip = Path(sample_report), Path(reference_zip)
    candidates = json.loads(sample_report.read_text(encoding='utf-8'))
    records = []
    with reference_zip.open('rb') as handle:
        archive_hash = hashlib.file_digest(handle, 'sha256').hexdigest()
    with zipfile.ZipFile(reference_zip) as archive:
        for sample in candidates['records']:
            expected_name = sample['image_id'] + '_segmentation.png'
            matches = [name for name in archive.namelist()
                       if PurePosixPath(name).name == expected_name]
            if len(matches) != 1:
                raise ValueError(f'{expected_name}: missing/ambiguous official reference')
            with Image.open(io.BytesIO(archive.read(matches[0]))) as original:
                original_values = np.asarray(original)
                original_box = mask_to_bbox(original_values)  # Strict check, no threshold.
                resized = original.resize((sample['width'], sample['height']), Image.Resampling.NEAREST)
                reference_box = list(mask_to_bbox(np.asarray(resized)))
                candidate_box = sample['boxes_by_threshold'][str(sample['candidate_threshold'])]
                records.append({
                    'image_id': sample['image_id'], 'official_mode': original.mode,
                    'official_size': list(original.size),
                    'official_pixel_values': np.unique(original_values).tolist(),
                    'official_original_box': list(original_box),
                    'school_size': [sample['width'], sample['height']],
                    'official_nearest_box': reference_box,
                    'candidate_threshold': sample['candidate_threshold'],
                    'candidate_box': candidate_box,
                    'bbox_iou_annotation_consistency_only': box_iou(reference_box, candidate_box),
                    'edge_differences_candidate_minus_reference':
                        [a - b for a, b in zip(candidate_box, reference_box)] if candidate_box else None,
                })
    return {
        'purpose': 'Small-sample annotation consistency, NOT model performance',
        'official_reference_url': REFERENCE_URL, 'official_archive_sha256': archive_hash,
        'reference_resize_rule': 'NEAREST to exactly match each school image width/height',
        'threshold_approved': False, 'school_preprocessing_provenance_verified': False,
        'representative_full_dataset_check': False, 'records': records,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample-report', type=Path, required=True)
    parser.add_argument('--reference-zip', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = compare_reference(args.sample_report, args.reference_zip)
    save_report(result, args.output, [args.sample_report, args.reference_zip])
    for record in result['records']:
        print(record['image_id'], 'official size=', record['official_size'],
              'values=', record['official_pixel_values'],
              'reference box=', record['official_nearest_box'],
              'candidate box=', record['candidate_box'],
              'annotation-only bbox IoU=', round(record['bbox_iou_annotation_consistency_only'], 6)
              if record['bbox_iou_annotation_consistency_only'] is not None else None)
    print('These numbers compare annotations, not YOLO predictions. Threshold remains provisional.')
