"""Acquire only the pinned BRIGHTER train/dev shards in an isolated workspace.

Added for distribution; this helper does not download or read the official test
partition and never changes historical reference manifests or fold assignments.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
REVISION = '419566ac8f46f951b51584be36c13e5363008144'
DATASET_ID = 'brighter-dataset/BRIGHTER-emotion-categories'
FILES = {
    'README.md': 'c573b3e4d6dd0a0c4b0ee8daac4e28e354d078b56918e4f992feaf4caa4d382f',
    'ptbr/train-00000-of-00001.parquet': 'dd3e4cb0df63e30d96f9ae8378eaf50208a48c8fb6cc6b60b58d2adda39d32e2',
    'ptbr/dev-00000-of-00001.parquet': 'b33aeba9913bcbd06cb258c4e6f7eaaa6fc758389d92b104ef14d0dda20a9c61',
}
REFERENCE_DIR = Path('metadata/publication')
RECEIPT = REFERENCE_DIR / 'development_acquisition_receipt.json'


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def strip_windows_path_namespace(value: str) -> str:
    """Normalize the Windows long-path namespace only for path comparison."""
    separator = chr(92)
    prefix = separator * 2 + '?' + separator
    if value.startswith(prefix + 'UNC' + separator):
        return separator * 2 + value[len(prefix) + 4:]
    if value.startswith(prefix):
        return value[len(prefix):]
    return value


def comparison_path(path: Path) -> Path:
    # Resolve symlinks first, then remove an optional Windows namespace. Calling
    # resolve again after stripping can reintroduce the namespace for long paths.
    return Path(strip_windows_path_namespace(str(path.resolve())))


def validate_download_path(downloaded: Path, raw_dir: Path, filename: str) -> Path:
    if filename not in FILES:
        raise RuntimeError('The requested file is outside the development allowlist.')
    expected = raw_dir / filename
    downloaded_canonical = comparison_path(downloaded)
    expected_canonical = comparison_path(expected)
    if (downloaded.is_symlink() or expected.is_symlink()
            or not expected_canonical.is_relative_to(comparison_path(raw_dir))
            or downloaded_canonical != expected_canonical):
        raise RuntimeError('Download returned a path outside the exact isolated development file.')
    try:
        same_file = downloaded.samefile(expected)
    except OSError as exc:
        raise RuntimeError('The downloaded development file could not be identified locally.') from exc
    if not same_file:
        raise RuntimeError('Download did not return the expected local development file.')
    # Use the known expected path for downstream readers instead of propagating
    # a platform-specific namespace returned by the Hub.
    return expected


def validate_workspace(root: Path) -> dict[str, Any]:
    marker = root / 'reproduction_workspace.json'
    if not marker.is_file():
        raise RuntimeError('Run prepare_reproduction_workspace.py first; acquisition is allowed only in its isolated tree.')
    payload = read_json(marker)
    if payload.get('purpose') != 'ISOLATED_NEW_DEVELOPMENT_EXECUTION' or payload.get('dataset_revision') != REVISION:
        raise RuntimeError('The isolated-workspace marker does not match the pinned development protocol.')
    if payload.get('raw_data_copied') is not False or payload.get('final_state_or_authorization_copied') is not False:
        raise RuntimeError('The workspace must start without historical raw data or final-test state/authorization.')
    for relative in ('metadata/brighter_ptbr/final/experiment_state.json', 'metadata/brighter_ptbr/final/final_test_authorization.json'):
        if (root / relative).exists():
            raise RuntimeError('Acquisition refuses a tree containing final-test state or authorization.')
    return payload


def choose_cv_manifest(reference: dict[str, Any], current_hashes: dict[str, str],
                       ordered_ids: list[str], accept_reencoded_parquet: bool) -> tuple[dict[str, Any], bool]:
    if reference.get('revision') != REVISION or reference.get('seed') != 42 or reference.get('n_splits') != 5:
        raise RuntimeError('Historical folds do not match the pinned protocol.')
    assignments = reference.get('assignments', [])
    if [row['id'] for row in assignments] != ordered_ids or len(set(ordered_ids)) != len(ordered_ids):
        raise RuntimeError('Reconstructed development IDs/order do not match historical fold assignments.')
    if any(row.get('fold') not in range(5) for row in assignments):
        raise RuntimeError('Historical fold assignment is outside the five-fold protocol.')
    changed = current_hashes != reference.get('source_sha256')
    if changed and not accept_reencoded_parquet:
        raise RuntimeError('Reconstructed Parquet bytes differ from historical hashes. No folds were changed. Prepare a fresh workspace and explicitly use --accept-reencoded-parquet to retain identical IDs/assignments with new, documented serialization fingerprints.')
    result = copy.deepcopy(reference)
    if changed:
        result['source_sha256'] = dict(current_hashes)
        result['publication_reconstruction'] = {
            'historical_manifest': 'metadata/publication/reference_cv_folds.json',
            'historical_source_sha256': copy.deepcopy(reference['source_sha256']),
            'assignments_unchanged': True,
            'reason': 'Pinned upstream bytes and ID-selected source contents verified; Parquet re-serialization differs in reconstructed environment.',
            'not_a_replacement_for_historical_evidence': True,
        }
    return result, changed


def acquire_development(*, accept_reencoded_parquet: bool = False,
                        downloader: Callable[..., str] | None = None) -> dict[str, Any]:
    root = ROOT.resolve()
    validate_workspace(root)
    reference_paths = {
        'dataset': root / REFERENCE_DIR / 'reference_dataset_manifest.json',
        'track_a': root / REFERENCE_DIR / 'reference_track_a_manifest.json',
        'freeze': root / REFERENCE_DIR / 'reference_final_experiment_manifest.json',
        'folds': root / REFERENCE_DIR / 'reference_cv_folds.json',
    }
    references = {key: read_json(path) for key, path in reference_paths.items()}
    reference_hashes = {key: sha256(path) for key, path in reference_paths.items()}
    if references['dataset'].get('huggingface_revision') != REVISION or references['track_a'].get('revision') != REVISION:
        raise RuntimeError('Dataset references do not identify the exact experimental revision.')
    if references['freeze']['dataset']['revision'] != REVISION:
        raise RuntimeError('Freeze reference does not identify the exact experimental revision.')
    for split in ('train', 'dev'):
        item = references['freeze']['dataset']['hashes']['raw'][split]
        if item.get('expected') != FILES[f'ptbr/{split}-00000-of-00001.parquet'] or item.get('observed') != item['expected'] or item.get('match') is not True:
            raise RuntimeError(f'Historical raw hash reference does not validate {split}.')
    raw_dir = root / 'data/raw/brighter_ptbr' / REVISION / 'hf_snapshot'
    interim_dir = root / 'data/interim/brighter_ptbr' / REVISION
    curated_dir = root / 'data/curated/brighter_ptbr' / REVISION / 'track_a'
    outputs = [interim_dir/f'{s}.parquet' for s in ('train', 'dev')]
    outputs += [curated_dir/f'{s}.parquet' for s in ('train', 'dev')]
    outputs += [root/'metadata/brighter_ptbr/dataset_manifest.json',
                root/'metadata/brighter_ptbr/track_a_manifest.json', raw_dir/'.brighter_snapshot.json']
    if any(not comparison_path(path).is_relative_to(comparison_path(root)) for path in (raw_dir, interim_dir, curated_dir)):
        raise RuntimeError('Data directories must resolve inside the isolated workspace.')
    if (root / RECEIPT).exists() or any(path.exists() for path in outputs):
        raise RuntimeError('Acquisition outputs already exist; prepare a new isolated workspace instead of overwriting them.')
    if downloader is None:
        from huggingface_hub import hf_hub_download
        downloader = hf_hub_download
    raw_files: dict[str, Path] = {}
    for filename, expected in FILES.items():
        path = Path(downloader(repo_id=DATASET_ID, repo_type='dataset', filename=filename,
                               revision=REVISION, local_dir=str(raw_dir),
                               cache_dir=str(root/'.cache/huggingface/development-only')))
        path = validate_download_path(path, raw_dir, filename)
        if sha256(path) != expected:
            raise RuntimeError(f'Pinned upstream bytes do not match the historical hash: {filename}')
        raw_files[filename] = path

    # Only these two verified local shards are materialized. No full snapshot
    # or dataset-loading API is invoked that could implicitly read test shards.
    from datasets import Dataset
    sys.path.insert(0, str(root / 'src'))
    from brighter.download import materialize_frames, save_parquet_splits
    from brighter.curation import curate_track_a, verify_curated_fidelity, _write_parquet_atomic
    from brighter.experiments import construct_development_pool
    from brighter.manifest import build_manifest, write_json_atomic
    from brighter.validation import EXPECTED_LABELS
    dataset = {split: Dataset.from_parquet(str(raw_files[f'ptbr/{split}-00000-of-00001.parquet']),
                                          cache_dir=str(root/'.cache/datasets-development'))
               for split in ('train', 'dev')}
    frames, source_columns, text_column, id_column, _ = materialize_frames(dataset)
    if {split: len(frame) for split, frame in frames.items()} != {'train': 2226, 'dev': 400}:
        raise RuntimeError('Pinned train/dev row counts differ from historical acquisition.')
    curated = curate_track_a(frames, REVISION)
    fidelity = verify_curated_fidelity(frames, curated)
    if {split: len(frame) for split, frame in curated.items()} != {'train': 2226, 'dev': 200}:
        raise RuntimeError('Explicit Track-A selection differs from the historical development pool.')
    interim_paths = save_parquet_splits(frames, interim_dir)
    for split, frame in curated.items():
        _write_parquet_atomic(curated_dir/f'{split}.parquet', frame)
    pool = construct_development_pool(root)
    current_hashes = {split: sha256(curated_dir/f'{split}.parquet') for split in ('train', 'dev')}
    receipt: dict[str, Any] = {
        'purpose': 'PINNED_DEVELOPMENT_ONLY_ACQUISITION', 'dataset': DATASET_ID,
        'revision': REVISION, 'downloaded_files': list(FILES),
        'raw_sha256': {name: sha256(path) for name, path in raw_files.items()},
        'raw_hash_provenance': 'Historical hf_snapshot/.brighter_snapshot.json and reference_final_experiment_manifest.json dataset.hashes.raw',
        'interim_sha256': {split: sha256(path) for split, path in interim_paths.items()},
        'curated_sha256': current_hashes,
        'historical_curated_sha256': {s: references['track_a']['curated_files'][s]['sha256'] for s in ('train', 'dev')},
        'source_fidelity': fidelity, 'development_rows': len(pool),
        'reference_manifest_sha256': reference_hashes,
        'test_downloaded': False, 'test_read': False, 'models_loaded': False,
        'training_executed': False, 'official_test_evaluated': False,
        'software_versions': {'python': platform.python_version(), **{name: importlib.metadata.version(name)
            for name in ('datasets', 'huggingface-hub', 'pandas', 'pyarrow')}},
    }
    try:
        cv_payload, reencoded = choose_cv_manifest(references['folds'], current_hashes,
                                                  pool.id.astype(str).tolist(), accept_reencoded_parquet)
    except RuntimeError as exc:
        receipt.update({'status': 'BLOCKED_BEFORE_TRAINING', 'reason': str(exc), 'fold_assignments_modified': False})
        write_json_atomic(root/RECEIPT, receipt)
        raise
    operational_folds = root/'metadata/brighter_ptbr/experiments/cv_folds.json'
    if read_json(operational_folds) != references['folds']:
        raise RuntimeError('Operational folds were changed before acquisition; refusing to replace them.')
    if reencoded:
        write_json_atomic(operational_folds, cv_payload)
    # Original reporting/freeze functions audit only train/dev, but expect these
    # operational paths. Label the new manifests as partial/reconstructed; the
    # complete historical records remain byte-identical reference_* documents.
    snapshot_manifest = {
        'dataset': DATASET_ID, 'configuration': 'ptbr', 'huggingface_revision': REVISION,
        'scope': 'RECONSTRUCTED_DEVELOPMENT_ONLY', 'test_included': False,
        'files': {name: sha256(path) for name, path in raw_files.items()},
    }
    dataset_manifest = build_manifest(
        dataset_id=DATASET_ID, configuration='ptbr', revision=REVISION,
        split_rows={split: len(frame) for split, frame in frames.items()},
        source_columns=source_columns, local_columns=list(frames['train'].columns),
        label_columns=list(EXPECTED_LABELS), parquet_paths=interim_paths,
        snapshot_path=raw_dir.relative_to(root), text_source_column=text_column,
        id_column=id_column, project_root=root,
    )
    dataset_manifest.update({'scope': 'RECONSTRUCTED_DEVELOPMENT_ONLY',
                             'historical_reference': 'metadata/publication/reference_dataset_manifest.json',
                             'test_included': False})
    track_manifest = {
        'dataset': DATASET_ID, 'configuration': 'ptbr', 'revision': REVISION,
        'experimental_track': 'A', 'scope': 'RECONSTRUCTED_DEVELOPMENT_ONLY',
        'historical_reference': 'metadata/publication/reference_track_a_manifest.json',
        'source_rows': {split: len(frame) for split, frame in frames.items()},
        'curated_rows': {split: len(frame) for split, frame in curated.items()},
        'text_processing': 'none', 'test_included': False, 'fidelity': fidelity,
        'curated_files': {split: {'path': (curated_dir/f'{split}.parquet').relative_to(root).as_posix(),
                                   'sha256': current_hashes[split],
                                   'bytes': (curated_dir/f'{split}.parquet').stat().st_size}
                          for split in ('train', 'dev')},
    }
    operational_metadata = {
        raw_dir/'.brighter_snapshot.json': snapshot_manifest,
        root/'metadata/brighter_ptbr/dataset_manifest.json': dataset_manifest,
        root/'metadata/brighter_ptbr/track_a_manifest.json': track_manifest,
    }
    for path, payload in operational_metadata.items():
        write_json_atomic(path, payload)
    if any(sha256(reference_paths[key]) != value for key, value in reference_hashes.items()):
        raise RuntimeError('Historical reference manifests changed during acquisition.')
    receipt.update({'status': 'DEVELOPMENT_READY', 'fold_assignments_modified': False,
                    'operational_fold_fingerprints_reconstructed': reencoded,
                    'operational_development_metadata_sha256': {
                        path.relative_to(root).as_posix(): sha256(path) for path in operational_metadata},
                    'historical_reference_manifests_unchanged': True})
    write_json_atomic(root/RECEIPT, receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--accept-reencoded-parquet', action='store_true',
                        help='Allow a documented new operational fingerprint if pinned data and identical historical fold IDs/order are verified; historical reference manifests remain unchanged.')
    args = parser.parse_args()
    result = acquire_development(accept_reencoded_parquet=args.accept_reencoded_parquet)
    print(f"Development ready: {result['development_rows']} Track-A train/dev rows; no test shard downloaded or read.")
    print(f"Reconstructed operational fold fingerprints: {result['operational_fold_fingerprints_reconstructed']}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
