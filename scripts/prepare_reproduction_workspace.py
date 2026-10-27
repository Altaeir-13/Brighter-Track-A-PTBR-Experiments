"""Prepare an empty, isolated execution tree without copying historical outputs.

Added for distribution. The original scientific modules/configurations are copied
byte-for-byte; this helper never downloads data, imports models, or runs training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[1]
PIN_FILES = (
    Path('metadata/brighter_ptbr/experiments/model_revisions.json'),
    Path('metadata/brighter_ptbr/experiments/cv_folds.json'),
)
REFERENCE_FILES = {
    Path('metadata/brighter_ptbr/dataset_manifest.json'): Path('metadata/publication/reference_dataset_manifest.json'),
    Path('metadata/brighter_ptbr/track_a_manifest.json'): Path('metadata/publication/reference_track_a_manifest.json'),
    Path('metadata/brighter_ptbr/final/final_experiment_manifest.json'): Path('metadata/publication/reference_final_experiment_manifest.json'),
    Path('metadata/brighter_ptbr/experiments/cv_folds.json'): Path('metadata/publication/reference_cv_folds.json'),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_workspace(destination: Path) -> dict[str, object]:
    source = SOURCE_ROOT.resolve()
    target = destination.expanduser().resolve()
    if target == source or target.is_relative_to(source) or source.is_relative_to(target):
        raise ValueError('Destination must be outside the source checkout and must not contain it.')
    if target.exists():
        raise FileExistsError('Destination already exists; no files were copied or overwritten.')
    if not target.parent.is_dir():
        raise FileNotFoundError('Destination parent must already exist.')

    selected: set[Path] = set()
    for directory, suffixes in (
        ('src/brighter', {'.py'}), ('scripts', {'.py'}),
        ('configs', {'.yaml', '.yml'}), ('tests', {'.py'}),
    ):
        base = source / directory
        if not base.is_dir():
            raise FileNotFoundError(f'Missing source directory: {directory}')
        selected.update(path.relative_to(source) for path in base.rglob('*')
                        if path.is_file() and path.suffix in suffixes)
    for name in ('pyproject.toml', 'README.md', *map(str, PIN_FILES)):
        relative = Path(name)
        if not (source / relative).is_file():
            raise FileNotFoundError(f'Missing source file: {relative.as_posix()}')
        selected.add(relative)
    inner = source / 'metadata/brighter_ptbr/experiments/threshold_inner_folds'
    if inner.is_dir():
        selected.update(path.relative_to(source) for path in inner.rglob('*.json') if path.is_file())
    # Reconstructed dependency files, when provided, are copied as documentation
    # for this new execution; they are not represented as historical lockfiles.
    requirements = source / 'requirements'
    if requirements.is_dir():
        selected.update(path.relative_to(source) for path in requirements.rglob('*')
                        if path.is_file() and path.suffix in {'.txt', '.in', '.md', '.json'})
    for path in source.glob('requirements*.txt'):
        if path.is_file():
            selected.add(path.relative_to(source))
    for name in ('LICENSE', 'LICENSE.txt', 'LICENSE.md', 'NOTICE', 'NOTICE.txt'):
        if (source / name).is_file():
            selected.add(Path(name))
    records = []
    for relative in sorted(selected):
        original = source / relative
        if original.is_symlink() or not original.resolve().is_relative_to(source):
            raise ValueError(f'Unsafe source file: {relative.as_posix()}')
        records.append({'path': relative.as_posix(), 'sha256': sha256(original),
                        'bytes': original.stat().st_size})
    for original_relative, output_relative in REFERENCE_FILES.items():
        original = source / original_relative
        if not original.is_file():
            raise FileNotFoundError(f'Missing historical reference: {original_relative.as_posix()}')
        if original.is_symlink() or not original.resolve().is_relative_to(source):
            raise ValueError(f'Unsafe historical reference: {original_relative.as_posix()}')
        records.append({'path': output_relative.as_posix(),
                        'source_path': original_relative.as_posix(),
                        'sha256': sha256(original), 'bytes': original.stat().st_size})
    target.mkdir()
    for item in records:
        relative = Path(str(item['path']))
        output = target / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / str(item.get('source_path', item['path'])), output)
        if sha256(output) != item['sha256']:
            raise RuntimeError(f'Copy hash mismatch: {relative.as_posix()}')
    receipt = {'purpose': 'ISOLATED_NEW_DEVELOPMENT_EXECUTION', 'copied_files': records,
               'dataset_revision': '419566ac8f46f951b51584be36c13e5363008144',
               'raw_data_copied': False, 'historical_models_copied': False,
               'historical_predictions_copied': False, 'historical_reports_copied': False,
               'historical_manifests_copied_as_references': True,
               'final_state_or_authorization_copied': False, 'training_executed': False,
               'official_test_evaluated': False}
    (target / 'reproduction_workspace.json').write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, required=True,
                        help='New directory outside this checkout; its parent must exist.')
    args = parser.parse_args()
    result = prepare_workspace(args.destination)
    print(f"Copied {len(result['copied_files'])} source/configuration/pin files; no training or data access.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
