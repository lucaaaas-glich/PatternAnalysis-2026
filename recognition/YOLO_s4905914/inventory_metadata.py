"""Bounded read-only metadata discovery; headers only, never patient rows.

Does not decode images, infer diagnoses, export labels or create a split.
Symlinks are not followed. A truncated scan cannot establish absence of data.
"""

import argparse
import csv
import json
import os
from pathlib import Path


def inventory(root, max_depth=3, max_entries=20000):
    root = Path(root)
    if not root.is_dir() or root.is_symlink():
        raise ValueError('Root must be an existing non-symlink directory')
    if max_depth < 0 or max_entries < 1:
        raise ValueError('Depth must be nonnegative and entry limit positive')
    candidates, directories, errors = [], [], []
    visited = 0
    truncated = False
    stack = [(root, 0)]
    while stack and not truncated:
        directory, depth = stack.pop()
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    visited += 1
                    if visited > max_entries:
                        truncated = True
                        break
                    path = Path(entry.path)
                    relative = str(path.relative_to(root))
                    if entry.is_symlink():
                        continue
                    if entry.is_dir(follow_symlinks=False):
                        if depth == 0:
                            directories.append(relative)
                        if depth < max_depth:
                            stack.append((path, depth + 1))
                        continue
                    suffix = path.suffix.lower()
                    if not entry.is_file(follow_symlinks=False) or suffix not in {'.csv', '.tsv', '.json', '.xlsx', '.xls', '.zip'}:
                        continue
                    record = {'file': relative, 'kind': suffix}
                    if suffix in {'.csv', '.tsv'}:
                        try:
                            with path.open(encoding='utf-8-sig', newline='') as handle:
                                header = handle.readline(8193)
                            if len(header) > 8192:
                                raise ValueError('Header exceeds 8192 characters')
                            record['header'] = next(csv.reader([header], delimiter='\t' if suffix == '.tsv' else ','), [])
                        except (OSError, UnicodeError, csv.Error, ValueError) as error:
                            record['error'] = str(error)
                    candidates.append(record)
        except OSError as error:
            errors.append({'directory': str(directory.relative_to(root)), 'error': str(error)})
    return {
        'purpose': 'Metadata discovery only; no patient rows printed or split created',
        'root': str(root), 'max_depth': max_depth, 'max_entries': max_entries,
        'entries_examined': min(visited, max_entries), 'truncated_by_entry_limit': truncated,
        'top_level_directories': sorted(directories),
        'candidates': sorted(candidates, key=lambda item: item['file']), 'directory_errors': errors,
        'patient_independence_verified': False, 'diagnosis_mapping_verified': False,
        'limitations': [
            'CSV/TSV first lines are treated as headers; malformed headerless files need review.',
            'JSON, spreadsheets and ZIPs are listed by name only, not opened.',
            'Symlinks, deeper directories and entries beyond the limit are not scanned.',
            'Absence of candidates in this bounded root does not prove metadata do not exist.',
            'Lesion IDs and patient IDs are different concepts; neither is inferred here.',
        ],
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/home/groups/comp3710'))
    parser.add_argument('--max-depth', type=int, default=3)
    parser.add_argument('--max-entries', type=int, default=20000)
    args = parser.parse_args()
    print(json.dumps(inventory(args.root, args.max_depth, args.max_entries), indent=2))
