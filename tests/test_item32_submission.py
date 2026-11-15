"""Check the selected blend, test-refit provenance and its portable audit."""

import copy
import hashlib
import json
import runpy
import tarfile
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

helpers = runpy.run_path('scripts/export_item32_submission.py', run_name='item32_helpers')
baseline = helpers['baseline']


def prediction(z, reverse=False):
    out = {'Id': np.array(['a', 'b']), 'customer_id': np.array(['ca', 'cb']),
           'z': np.array(z, dtype=float), 'spend_30d': np.expm1(z)}
    return {k: v[::-1] if reverse else v for k, v in out.items()}


class ItemSubmissionTests(unittest.TestCase):
    def setUp(self):
        self.test = pd.DataFrame({'Id': ['a', 'b'], 'customer_id': ['ca', 'cb']})
        self.sample = pd.DataFrame({'Id': ['b', 'a'], 'spend_30d': [0., 0.]})

    def fixture(self, root):
        data, history, item = [root / x for x in ['data', 'history', 'item']]
        for directory in [data, history, item]:
            directory.mkdir()
        test = self.test.assign(as_of_date='2018-11-01', target_end='2018-12-01')
        test.to_csv(data / 'test.csv', index=False)
        self.sample.to_csv(data / 'sample_submission.csv', index=False)
        common = {'device': 'cuda', 'smoke': False, 'cache': {},
                  'reference_prediction_hashes': {},
                  'reference_manifest_sha256': hashlib.sha256(b'fixture').hexdigest()}
        base_cfg = {'base_settings': baseline.SETTINGS, 'seeds': baseline.SEEDS,
                    'profiles': [{'name': 'baseline', 'overrides': {}}]}
        embed_cfg = {'base_settings': baseline.SETTINGS, 'seeds': baseline.SEEDS,
                     'embedding': helpers['EMBEDDING'],
                     'profiles': [{'name': 'item16', 'overrides': {'embedding_dim': 16}},
                                  {'name': 'item32', 'overrides': {'embedding_dim': 32}}]}
        source = {f'{p.parent.name}/{p.name}': baseline.digest(p) for package in
                  ['spend_baselines', 'spend_behavior', 'spend_tuning', 'spend_embeddings']
                  for p in (Path('src') / package).glob('*.py')}
        for directory, profile, settings, cfg, shift in [
                (history, 'baseline', baseline.SETTINGS, base_cfg, 0.),
                (item, 'item32', helpers['SETTINGS'], embed_cfg, 2.)]:
            manifest = {**common, 'config': cfg, 'source_sha256': source if shift else {}}
            (directory / 'manifest.json').write_text(json.dumps(manifest))
            for number, seed in enumerate(baseline.SEEDS):
                job = directory / 'refit_jobs' / profile / helpers['DATE'] / str(seed)
                job.mkdir(parents=True)
                np.savez_compressed(job / 'predictions.npz',
                                    **prediction([1. + shift + number, 2. + shift + number],
                                                 reverse=(number == 1)))
                pd.DataFrame({'epoch': [1, 2], 'inner_metric': [1., 2.]}).to_csv(
                    job / 'selection.csv', index=False)
                pd.DataFrame({'epoch': [1], 'train_loss': [1.]}).to_csv(job / 'refit.csv', index=False)
                (job / 'model.pt').write_bytes(b'fixture weights, never deserialised')
                result = {'job': {'profile': profile, 'date': helpers['DATE'], 'seed': seed,
                                  'testing': True, 'settings': settings},
                          'selected_epoch': 1, 'inner_date': '2018-10-01',
                          'max_target_end_train': '2018-10-31', 'max_target_end_inner': '2018-10-01',
                          'n_validation': 2, 'n_train': 30, 'vocabulary_min_customers': 3,
                          'vocabulary_items': 5,
                          'sha256': {p.name: baseline.digest(p) for p in job.iterdir()}}
                (job / 'result.json').write_text(json.dumps(result))
            mean = directory / 'refit' / profile
            mean.mkdir(parents=True)
            np.savez_compressed(mean / 'predictions.npz', **prediction([2. + shift, 3. + shift]))
            (mean / 'result.json').write_text(json.dumps({
                'profile': profile, 'cutoff': helpers['DATE'],
                'predictions_sha256': baseline.digest(mean / 'predictions.npz')}))
        return data, history, item

    def export(self, root, data, history, item):
        fake = lambda *args: SimpleNamespace(manifest_bytes=b'fixture')
        with patch.object(baseline, 'FrozenRun', fake):
            return helpers['export'](item, history, root / 'reference', data, root / 'submission.csv')

    def test_blend_aligns_ids_and_blends_logs_before_expm1(self):
        out = helpers['blend'](self.test, self.sample, prediction([2., 3.]), prediction([4., 5.], True))
        self.assertEqual(out.Id.tolist(), ['b', 'a'])
        np.testing.assert_allclose(out.spend_30d, np.expm1([3.5, 2.5]))
        money_average = .75 * np.expm1([3., 2.]) + .25 * np.expm1([5., 4.])
        self.assertFalse(np.allclose(out.spend_30d, money_average))
        bad = prediction([4., 5.]); bad['customer_id'][0] = 'xx'
        with self.assertRaisesRegex(ValueError, 'customer'):
            helpers['blend'](self.test, self.sample, prediction([2., 3.]), bad)

    def test_end_to_end_export_and_weight_free_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); data, history, item = self.fixture(root)
            before = {p: p.read_bytes() for p in history.rglob('*') if p.is_file()}
            receipt = self.export(root, data, history, item)
            out = pd.read_csv(root / 'submission.csv')
            self.assertEqual(out.Id.tolist(), ['b', 'a'])
            np.testing.assert_allclose(out.spend_30d, np.expm1([3.5, 2.5]))
            self.assertEqual(receipt['weights'], {'joint_history': .75, 'item32': .25})
            self.assertFalse(receipt['kaggle_uploaded'])
            self.assertEqual(receipt['submission_sha256'], baseline.digest(root / 'submission.csv'))
            for path, content in before.items():
                self.assertEqual(path.read_bytes(), content)
            with tarfile.open(root / 'submission_audit.tar.gz') as archive:
                names = archive.getnames()
                self.assertFalse(any(n.endswith('.pt') or 'cache' in n for n in names))
                self.assertEqual(sum(n.endswith('predictions.npz') for n in names), 8)
                self.assertEqual(archive.extractfile('submission.csv').read(),
                                 (root / 'submission.csv').read_bytes())

    def test_cpu_smoke_wrong_config_and_reference_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, history, item = self.fixture(Path(tmp))
            manifest = json.loads((item / 'manifest.json').read_text())
            base = json.loads((history / 'manifest.json').read_text())
            for field, value in [('device', 'cpu'), ('smoke', True),
                                 ('reference_manifest_sha256', 'wrong'), ('cache', {'rows': 2})]:
                bad = copy.deepcopy(manifest); bad[field] = value
                with self.assertRaises(ValueError):
                    helpers['validate_manifest'](bad, base)
            bad = copy.deepcopy(manifest); bad['config']['embedding']['min_customers'] = 1
            with self.assertRaisesRegex(ValueError, 'configuration'):
                helpers['validate_manifest'](bad, base)

    def test_corrupt_artifacts_and_wrong_epoch_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); data, history, item = self.fixture(root)
            job = item / 'refit_jobs/item32/2018-11-01/42'
            (job / 'refit.csv').write_text('epoch,train_loss\n1,1\n2,1\n')
            with self.assertRaisesRegex(ValueError, 'Corrupt'):
                self.export(root, data, history, item)
            result = json.loads((job / 'result.json').read_text())
            result['sha256']['refit.csv'] = baseline.digest(job / 'refit.csv')
            (job / 'result.json').write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError, 'epoch'):
                self.export(root, data, history, item)
            self.assertFalse((root / 'submission.csv').exists())

    def test_late_labels_and_wrong_mean_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); data, history, item = self.fixture(root)
            job = item / 'refit_jobs/item32/2018-11-01/42/result.json'
            result = json.loads(job.read_text()); result['max_target_end_train'] = '2018-11-02'
            job.write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError, 'temporal'):
                self.export(root, data, history, item)
            result['max_target_end_train'] = '2018-10-31'; job.write_text(json.dumps(result))
            mean = item / 'refit/item32'
            np.savez_compressed(mean / 'predictions.npz', **prediction([4.1, 5.]))
            marker = json.loads((mean / 'result.json').read_text())
            marker['predictions_sha256'] = baseline.digest(mean / 'predictions.npz')
            (mean / 'result.json').write_text(json.dumps(marker))
            with self.assertRaises(AssertionError):
                self.export(root, data, history, item)


if __name__ == '__main__':
    unittest.main()
