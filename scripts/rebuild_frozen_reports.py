"""Rebuild the historical evidence tables in a new directory from aggregate records.

Added for distribution. No models, datasets, per-instance predictions or official
test evaluation are used. The historical source checkout is never written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rebuild_reports(destination: Path) -> dict[str, object]:
    source = SOURCE_ROOT.resolve()
    target = destination.expanduser().resolve()
    if target == source or target.is_relative_to(source) or source.is_relative_to(target):
        raise ValueError('Destination must be outside the source checkout and must not contain it.')
    if target.exists():
        raise FileExistsError('Destination already exists; no files were copied or overwritten.')
    if not target.parent.is_dir():
        raise FileNotFoundError('Destination parent must already exist.')

    selected: set[Path] = set()
    allowed = {'.json', '.csv', '.md', '.tex', '.txt'}
    for directory in ('metadata/brighter_ptbr', 'reports/brighter_ptbr'):
        base = source / directory
        if not base.is_dir():
            raise FileNotFoundError(f'Missing published input directory: {directory}')
        for path in base.rglob('*'):
            if path.is_file() and path.suffix in allowed and 'authorization' not in path.name.lower():
                selected.add(path.relative_to(source))
    records = []
    for relative in sorted(selected):
        original = source / relative
        if original.is_symlink() or not original.resolve().is_relative_to(source):
            raise ValueError(f'Unsafe source file: {relative.as_posix()}')
        records.append({'path': relative.as_posix(), 'sha256': sha256(original),
                        'bytes': original.stat().st_size})
    target.mkdir()
    for item in records:
        relative = Path(str(item['path']))
        output = target / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / relative, output)
        if sha256(output) != item['sha256']:
            raise RuntimeError(f'Copy hash mismatch: {relative.as_posix()}')
    os.environ['MPLCONFIGDIR'] = str(target / '.cache/matplotlib')
    os.environ['MPLBACKEND'] = 'Agg'
    sys.path.insert(0, str(source / 'src'))
    from brighter.paper_evidence import generate_paper_package
    package = generate_paper_package(target)
    changed_originals = [item['path'] for item in records
                         if sha256(source / str(item['path'])) != item['sha256']]
    if changed_originals:
        raise RuntimeError('Historical source changed during report generation.')
    changed_copies = [item['path'] for item in records
                      if sha256(target / str(item['path'])) != item['sha256']]
    receipt = {'purpose': 'FROZEN_AGGREGATE_REPORT_REBUILD', 'source_inputs': records,
               'master_systems': package['master_rows'], 'evidence_claims': package['claims'],
               'copied_files_regenerated_with_different_bytes': changed_copies,
               'historical_sources_unchanged': True, 'raw_dataset_read': False,
               'per_instance_predictions_read': False, 'models_loaded': False,
               'training_executed': False, 'official_test_evaluated': False,
               'thresholds_or_hyperparameters_selected': False,
               'note': 'Regeneration from recorded aggregates is not a new model evaluation. Historical fold means and concatenated-OOF metrics retain their original definitions.'}
    (target / 'report_rebuild_receipt.json').write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, required=True,
                        help='New directory outside this checkout; its parent must exist.')
    args = parser.parse_args()
    result = rebuild_reports(args.destination)
    print(f"Rebuilt {result['master_systems']} system rows and {result['evidence_claims']} claims from frozen aggregates.")
    print('No model inference, training, test evaluation or parameter selection was performed.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
