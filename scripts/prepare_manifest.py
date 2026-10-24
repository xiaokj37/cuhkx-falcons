"""Derive the training manifest from a locally authorized CUHK-X archive."""
import argparse
import csv
import json
from pathlib import Path, PurePosixPath
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DIRECTORIES = ('Depth_Color', 'IR', 'Thermal', 'Skeleton', 'IMU', 'Radar')


def manifest_rows(names, folds):
    """Keep first occurrence order; a subject always belongs to one held fold."""
    assignment = {int(user): int(fold) for fold, spec in folds.items() for user in spec['subjects']}
    if len(assignment) != sum(len(spec['subjects']) for spec in folds.values()):
        raise ValueError('Subjects occur in multiple folds')
    groups = {}
    for name in names:
        parts = PurePosixPath(name).parts
        if any(p.startswith('.') or p == '__MACOSX' for p in parts) or 'visualizations' in parts:
            continue
        if len(parts) < 7 or parts[:2] != ('HAR', 'data') or parts[2] not in DIRECTORIES:
            continue
        modality, action, subject, trial = parts[2:6]
        if not re.fullmatch(r'user\d+', subject) or int(subject[4:]) not in assignment:
            raise ValueError('Unexpected training subject')
        label_text, separator, _ = action.partition('_')
        if not separator or not label_text.isdigit() or not 0 <= int(label_text) < 40:
            raise ValueError('Invalid action directory')
        key = (action, subject, trial)
        if key not in groups:
            row = dict(action_id=int(label_text), action=action, subject=subject,
                       trial=trial, validation_fold=assignment[int(subject[4:])])
            for directory in DIRECTORIES:
                row[directory + '_path'] = ''
                row[directory + '_file_count'] = 0
            groups[key] = row
        groups[key][modality + '_path'] = '/'.join(parts[:6])
        groups[key][modality + '_file_count'] += 1
    if not groups:
        raise ValueError('No HAR/data/<modality>/<action>/<user>/<trial>/ files found')
    return list(groups.values())


def source_names(source):
    if source.is_dir():
        return [p.relative_to(source).as_posix() for p in source.rglob('*') if p.is_file()]
    with zipfile.ZipFile(source) as archive:
        return [item.filename for item in archive.infolist() if not item.is_dir()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path, help='Merged ZIP or directory containing HAR/')
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--folds', type=Path, default=ROOT / 'configs/subject_folds.json')
    parser.add_argument('--allow-partial', action='store_true', help='Only for metadata/unit checks; not the original training recipe')
    args = parser.parse_args()
    folds = json.loads(args.folds.read_text())['folds']
    rows = manifest_rows(source_names(args.source), folds)
    if not args.allow_partial:
        if len(rows) != 3036 or any(sum(r['validation_fold'] == int(k) for r in rows) != v['clips']
                                  for k, v in folds.items()):
            raise ValueError('Training population differs from the original 3,036-clip threefold protocol')
    if args.out.exists():
        raise ValueError('Refusing to overwrite an existing manifest')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(dict(clips=len(rows), manifest=str(args.out))))


if __name__ == '__main__':
    main()
