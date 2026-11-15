"""Export the selected 75% Joint-history / 25% item32 log-space blend."""

import argparse
import importlib.util
import json
import tarfile
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from spend_baselines.data import read_objects
from spend_baselines.suite_data import atomic_json, digest

_spec = importlib.util.spec_from_file_location(
    'joint_history_export', Path(__file__).with_name('export_joint_history_submission.py'))
baseline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(baseline)

EMBEDDING = {'window_days': 180, 'half_life_days': 90, 'max_items': 128,
             'min_customers': 3, 'weight_decay': .001, 'initial_std': .05}
SETTINGS = {**baseline.SETTINGS, 'embedding_dim': 32, 'embedding': EMBEDDING}
DATE = '2018-11-01'


def validate_manifest(manifest, baseline_manifest):
    if manifest.get('smoke') or manifest.get('device') != 'cuda':
        raise ValueError('Submission requires full CUDA item32, not smoke/CPU')
    cfg = manifest['config']
    if (cfg['base_settings'] != baseline.SETTINGS or cfg['seeds'] != baseline.SEEDS
            or cfg['embedding'] != EMBEDDING
            or cfg['profiles'] != [{'name': 'item16', 'overrides': {'embedding_dim': 16}},
                                   {'name': 'item32', 'overrides': {'embedding_dim': 32}}]):
        raise ValueError('Expected the verified item32 configuration')
    for key in ['reference_manifest_sha256', 'cache', 'reference_prediction_hashes']:
        if manifest[key] != baseline_manifest[key]:
            raise ValueError(f'Baseline/item32 reference differs: {key}')
    source_root = Path(__file__).resolve().parents[1] / 'src'
    expected = {f'{p.parent.name}/{p.name}' for package in
                ['spend_baselines', 'spend_behavior', 'spend_tuning', 'spend_embeddings']
                for p in (source_root / package).glob('*.py')}
    if set(manifest['source_sha256']) != expected:
        raise ValueError('Incomplete model source manifest')
    for name, sha in manifest['source_sha256'].items():
        if digest(source_root / name) != sha:
            raise ValueError(f'Original item32 implementation differs: {name}')


def validate_item32(run_dir, test):
    """Validate all three test refits before using their predictions."""
    directory = run_dir / 'refit/item32'
    marker = json.loads((directory / 'result.json').read_text())
    if marker['profile'] != 'item32' or marker['cutoff'] != DATE:
        raise ValueError('Expected item32 test refit')
    if digest(directory / 'predictions.npz') != marker['predictions_sha256']:
        raise ValueError('Corrupt item32 mean3 forecasts')
    predictions, provenance = [], []
    for seed in baseline.SEEDS:
        jobdir = run_dir / 'refit_jobs/item32' / DATE / str(seed)
        result = json.loads((jobdir / 'result.json').read_text())
        expected = {'profile': 'item32', 'date': DATE, 'seed': seed,
                    'testing': True, 'settings': SETTINGS}
        if result['job'] != expected or result['n_validation'] != len(test):
            raise ValueError('Unexpected item32 refit job/model/population')
        if (pd.Timestamp(result['max_target_end_train']) > pd.Timestamp(DATE)
                or result['inner_date'] != '2018-10-01'
                or pd.Timestamp(result['max_target_end_inner']) > pd.Timestamp('2018-10-01')):
            raise ValueError('Item32 refit crosses a temporal cutoff')
        if result['vocabulary_min_customers'] != 3:
            raise ValueError('Item32 vocabulary support differs')
        required = {'predictions.npz', 'selection.csv', 'refit.csv', 'model.pt'}
        if not required.issubset(result['sha256']):
            raise ValueError('Incomplete item32 refit artifacts')
        for name, sha in result['sha256'].items():
            if Path(name).name != name or digest(jobdir / name) != sha:
                raise ValueError(f'Corrupt item32 refit artifact: {name}')
        selection = pd.read_csv(jobdir / 'selection.csv')
        refit = pd.read_csv(jobdir / 'refit.csv')
        best, epoch = np.inf, 0
        if (not np.isfinite(selection.inner_metric).all()
                or selection.epoch.tolist() != list(range(1, len(selection) + 1))):
            raise ValueError('Invalid item32 selection curve')
        for row in selection.itertuples():
            if row.inner_metric < best - SETTINGS['min_delta']:
                best, epoch = row.inner_metric, row.epoch
        if (epoch < 1 or epoch != result['selected_epoch']
                or refit.epoch.tolist() != list(range(1, epoch + 1))):
            raise ValueError('Item32 refit epoch differs from inner selection')
        predictions.append(baseline.load_npz(jobdir / 'predictions.npz'))
        provenance.append({'seed': seed, 'selected_epoch': epoch, 'n_train': result['n_train'],
                           'vocabulary_items': result['vocabulary_items'],
                           'result_sha256': digest(jobdir / 'result.json')})
    saved = baseline.load_npz(directory / 'predictions.npz')
    sample = pd.DataFrame({'Id': test.Id, 'spend_30d': 0.})
    baseline.assemble(test, sample, predictions, saved)
    return saved, provenance


def blend(test, sample, history, item32):
    h = baseline.prediction_frame(history, test)
    i = baseline.prediction_frame(item32, test)
    z = .75 * h.z.to_numpy() + .25 * i.z.to_numpy()
    values = np.maximum(np.expm1(z), 0)
    if not np.isfinite(values).all():
        raise ValueError('Non-finite blended forecasts')
    if (list(sample.columns) != ['Id', 'spend_30d'] or not sample.Id.is_unique
            or set(sample.Id) != set(test.Id)):
        raise ValueError('Sample submission schema/Ids differ')
    return (pd.DataFrame({'Id': test.Id.to_numpy(str), 'spend_30d': values})
            .set_index('Id').loc[sample.Id].reset_index())


def audit_archive(path, run_dir, baseline_dir, output):
    files = [(output, output.name), (output.with_suffix('.json'), output.with_suffix('.json').name)]
    for label, directory, profile in [('item32', run_dir, 'item32'),
                                      ('baseline', baseline_dir, 'baseline')]:
        files.append((directory / 'manifest.json', f'{label}/manifest.json'))
        for name in ['predictions.npz', 'result.json']:
            relative = f'refit/{profile}/{name}'
            files.append((directory / relative, f'{label}/{relative}'))
        for seed in baseline.SEEDS:
            for name in ['predictions.npz', 'result.json', 'selection.csv', 'refit.csv']:
                relative = f'refit_jobs/{profile}/{DATE}/{seed}/{name}'
                files.append((directory / relative, f'{label}/{relative}'))
    temporary = path.with_name(path.name + '.tmp')
    with tarfile.open(temporary, 'w:gz') as archive:
        for source, name in files:
            archive.add(source, arcname=name, recursive=False)
    temporary.replace(path)


def export(run_dir, baseline_dir, reference, data_dir, output):
    run_dir, baseline_dir = Path(run_dir), Path(baseline_dir)
    data_dir, output = Path(data_dir), Path(output)
    baseline_manifest = json.loads((baseline_dir / 'manifest.json').read_text())
    manifest = json.loads((run_dir / 'manifest.json').read_text())
    validate_manifest(manifest, baseline_manifest)
    # Reuse the existing full provenance checks without altering its CSV/receipt.
    with tempfile.TemporaryDirectory() as tmp:
        baseline_receipt = baseline.export(baseline_dir, reference, data_dir,
                                           Path(tmp) / 'baseline.csv')
    test = read_objects(data_dir / 'test.csv')
    if test.as_of_date.nunique() != 1 or not test.as_of_date.eq(DATE).all():
        raise ValueError('Expected the official November test')
    sample = pd.read_csv(data_dir / 'sample_submission.csv', dtype={'Id': str})
    item32, provenance = validate_item32(run_dir, test)
    history = baseline.load_npz(baseline_dir / 'refit/baseline/predictions.npz')
    out = blend(test, sample, history, item32)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + '.tmp')
    out.to_csv(temporary, index=False)
    checked = pd.read_csv(temporary, dtype={'Id': str})
    if (list(checked.columns) != ['Id', 'spend_30d'] or not checked.Id.equals(sample.Id)
            or not np.isfinite(checked.spend_30d).all() or checked.spend_30d.lt(0).any()):
        raise ValueError('Written submission check failed')
    temporary.replace(output)
    receipt = {'candidate': '75% Joint+history mean3 + 25% item32 mean3',
               'space': 'log1p', 'weights': {'joint_history': .75, 'item32': .25},
               'cutoff': DATE, 'rows': len(out), 'seeds_each': baseline.SEEDS,
               'item32_settings': SETTINGS, 'item32_seed_refits': provenance,
               'baseline_receipt': baseline_receipt,
               'item32_manifest_sha256': digest(run_dir / 'manifest.json'),
               'test_sha256': digest(data_dir / 'test.csv'),
               'sample_sha256': digest(data_dir / 'sample_submission.csv'),
               'submission_sha256': digest(output), 'kaggle_uploaded': False}
    atomic_json(output.with_suffix('.json'), receipt)
    audit_archive(output.with_name(output.stem + '_audit.tar.gz'), run_dir, baseline_dir, output)
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, default=Path('outputs/item-embeddings-v1'))
    p.add_argument('--baseline-dir', type=Path, default=Path('outputs/history-tuning-v1'))
    p.add_argument('--reference', type=Path, default=Path('outputs/behavior-v1'))
    p.add_argument('--data-dir', type=Path, default=Path('data/raw'))
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    output = args.output or args.run_dir / 'submission_joint_history_item32_75_25.csv'
    receipt = export(args.run_dir, args.baseline_dir, args.reference, args.data_dir, output)
    print(f'Created {output}: {receipt["rows"]} rows; SHA256 {receipt["submission_sha256"]}')
    print(f'Audit archive: {output.with_name(output.stem + "_audit.tar.gz")}')
    print('CSV validated. Kaggle upload has not been performed.')


if __name__ == '__main__':
    raise SystemExit('Archived item32 blend exporter. Active scope is seminars1-10; use scripts/make_course10_submission.sh')
