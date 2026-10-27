"""Safety/identity checks for added distribution helpers; synthetic fixtures only."""
from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def helper(name: str):
    spec = importlib.util.spec_from_file_location(f'public_{name}', ROOT/'scripts'/f'{name}.py')
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding='utf-8')


def marked_workspace(root: Path, acquisition) -> None:
    write_json(root/'reproduction_workspace.json', {
        'purpose': 'ISOLATED_NEW_DEVELOPMENT_EXECUTION',
        'dataset_revision': acquisition.REVISION,
        'raw_data_copied': False, 'final_state_or_authorization_copied': False,
    })


def test_workspace_copy_preserves_pins_and_license_without_copying_historical_outputs(tmp_path: Path, monkeypatch) -> None:
    module = helper('prepare_reproduction_workspace')
    source = tmp_path/'source'
    for directory in ('src/brighter', 'scripts', 'configs', 'tests'):
        (source/directory).mkdir(parents=True)
    for relative in ('src/brighter/example.py', 'scripts/example.py', 'tests/example.py',
                     'configs/example.yaml', 'pyproject.toml', 'README.md', 'LICENSE',
                     'requirements/reconstructed-cpu.in'):
        path = source/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f'fixture:{relative}\n', encoding='utf-8')
    for relative in set(module.PIN_FILES)|set(module.REFERENCE_FILES):
        write_json(source/relative, {'source': relative.as_posix()})
    for relative in ('data/private.txt', 'artifacts/models/model.safetensors',
                     'artifacts/predictions/output.parquet', 'reports/history.json',
                     'metadata/brighter_ptbr/final/experiment_state.json',
                     'metadata/brighter_ptbr/final/final_test_authorization.json'):
        path = source/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'must not be copied')
    monkeypatch.setattr(module, 'SOURCE_ROOT', source)
    target = tmp_path/'execution'
    receipt = module.prepare_workspace(target)
    assert (target/'LICENSE').read_bytes() == (source/'LICENSE').read_bytes()
    assert (target/'requirements/reconstructed-cpu.in').read_bytes() == (source/'requirements/reconstructed-cpu.in').read_bytes()
    assert not (target/'data').exists() and not (target/'artifacts').exists() and not (target/'reports').exists()
    assert not (target/'metadata/brighter_ptbr/final').exists()
    for original, reference in module.REFERENCE_FILES.items():
        assert (target/reference).read_bytes() == (source/original).read_bytes()
    assert receipt['final_state_or_authorization_copied'] is False
    with pytest.raises(FileExistsError):
        module.prepare_workspace(target)


def test_workspace_refuses_nested_destination_before_copying(tmp_path: Path, monkeypatch) -> None:
    module = helper('prepare_reproduction_workspace')
    monkeypatch.setattr(module, 'SOURCE_ROOT', tmp_path)
    with pytest.raises(ValueError, match='outside'):
        module.prepare_workspace(tmp_path/'nested')
    assert not (tmp_path/'nested').exists()


@pytest.mark.parametrize('condition', ('no_marker', 'wrong_purpose', 'raw_data_copied', 'final_state'))
def test_acquisition_refuses_unisolated_or_finalized_tree_before_download(tmp_path: Path, monkeypatch, condition: str) -> None:
    module = helper('acquire_development')
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    if condition != 'no_marker':
        marked_workspace(tmp_path, module)
        marker = tmp_path/'reproduction_workspace.json'
        payload = json.loads(marker.read_text(encoding='utf-8'))
        if condition == 'wrong_purpose':
            payload['purpose'] = 'HISTORICAL_CHECKOUT'
        if condition == 'raw_data_copied':
            payload['raw_data_copied'] = True
        marker.write_text(json.dumps(payload), encoding='utf-8')
        if condition == 'final_state':
            write_json(tmp_path/'metadata/brighter_ptbr/final/experiment_state.json', {'state': 'TEST_EVALUATED'})
    calls = []
    with pytest.raises(RuntimeError):
        module.acquire_development(downloader=lambda **kwargs: calls.append(kwargs))
    assert calls == []
    assert not (tmp_path/'data').exists()


def fold_reference(acquisition) -> dict:
    return {'revision': acquisition.REVISION, 'seed': 42, 'n_splits': 5,
            'source_sha256': {'train': 'a'*64, 'dev': 'b'*64},
            'assignments': [{'id': f'train_{i}', 'fold': i % 5} for i in range(10)]}


def test_parquet_reencoding_is_explicit_and_preserves_exact_fold_assignments() -> None:
    module = helper('acquire_development')
    reference = fold_reference(module)
    original = copy.deepcopy(reference)
    ids = [row['id'] for row in reference['assignments']]
    new_hashes = {'train': 'c'*64, 'dev': 'd'*64}
    with pytest.raises(RuntimeError, match='Parquet bytes differ'):
        module.choose_cv_manifest(reference, new_hashes, ids, False)
    derived, changed = module.choose_cv_manifest(reference, new_hashes, ids, True)
    assert changed and derived['source_sha256'] == new_hashes
    assert derived['assignments'] == original['assignments']
    assert derived['publication_reconstruction']['historical_source_sha256'] == original['source_sha256']
    assert reference == original


def test_reencoded_acceptance_never_allows_changed_id_order() -> None:
    module = helper('acquire_development')
    reference = fold_reference(module)
    ids = [row['id'] for row in reference['assignments']]
    with pytest.raises(RuntimeError, match='IDs/order'):
        module.choose_cv_manifest(reference, {'train': 'c'*64, 'dev': 'd'*64}, list(reversed(ids)), True)


def test_upstream_hash_mismatch_stops_before_materialization(tmp_path: Path, monkeypatch) -> None:
    module = helper('acquire_development')
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    marked_workspace(tmp_path, module)
    raw = {split: {'expected': module.FILES[f'ptbr/{split}-00000-of-00001.parquet'],
                   'observed': module.FILES[f'ptbr/{split}-00000-of-00001.parquet'], 'match': True}
           for split in ('train', 'dev')}
    refs = tmp_path/module.REFERENCE_DIR
    write_json(refs/'reference_dataset_manifest.json', {'huggingface_revision': module.REVISION})
    write_json(refs/'reference_track_a_manifest.json', {'revision': module.REVISION})
    write_json(refs/'reference_final_experiment_manifest.json', {'dataset': {'revision': module.REVISION, 'hashes': {'raw': raw}}})
    write_json(refs/'reference_cv_folds.json', fold_reference(module))
    calls = []
    def invalid_download(**kwargs):
        calls.append(kwargs)
        path = Path(kwargs['local_dir'])/kwargs['filename']
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'wrong bytes')
        return str(path)
    with pytest.raises(RuntimeError, match='upstream bytes'):
        module.acquire_development(downloader=invalid_download)
    assert [call['filename'] for call in calls] == ['README.md']
    assert all(call['revision'] == module.REVISION for call in calls)
    assert not (tmp_path/'data/interim').exists()
    assert not (tmp_path/'data/curated').exists()


def test_download_guard_requires_exact_allowlisted_path_even_for_equal_bytes(tmp_path: Path) -> None:
    module = helper('acquire_development')
    raw = tmp_path/'raw'
    raw.mkdir()
    expected = raw/'README.md'
    expected.write_bytes(b'fixture')
    other = tmp_path/'README.md'
    other.write_bytes(b'fixture')
    assert module.validate_download_path(expected, raw, 'README.md') == expected
    with pytest.raises(RuntimeError, match='exact isolated'):
        module.validate_download_path(other, raw, 'README.md')
    with pytest.raises(RuntimeError, match='allowlist'):
        module.validate_download_path(expected, raw, '../README.md')


def test_windows_namespace_normalization_preserves_drive_and_unc_identity() -> None:
    module = helper('acquire_development')
    separator = chr(92)
    prefix = separator * 2 + '?' + separator
    normal = 'C:' + separator + 'fixture' + separator + 'README.md'
    unc = separator * 2 + 'server' + separator + 'share' + separator + 'README.md'
    assert module.strip_windows_path_namespace(prefix + normal) == normal
    assert module.strip_windows_path_namespace(prefix + 'UNC' + separator + unc[2:]) == unc
    assert module.strip_windows_path_namespace(normal) == normal


@pytest.mark.skipif(os.name != 'nt', reason='Windows namespace regression requires Windows paths')
def test_windows_hub_extended_path_is_the_exact_expected_local_file(tmp_path: Path) -> None:
    module = helper('acquire_development')
    raw = tmp_path/'raw'
    raw.mkdir()
    expected = raw/'README.md'
    expected.write_bytes(b'fixture')
    prefix = chr(92) * 2 + '?' + chr(92)
    extended = Path(prefix + str(expected.resolve()))
    assert extended.samefile(expected)
    assert module.validate_download_path(extended, raw, 'README.md') == expected
