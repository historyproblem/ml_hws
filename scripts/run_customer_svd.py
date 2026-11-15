"""Offline trial: a fixed SVD of customer purchase histories as MLP input."""

import argparse
import json
import platform
import time
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

from spend_baselines.data import feature_matrix, load_dataset, read_objects
from spend_baselines.suite_data import atomic_json, digest
from spend_baselines.suite_models import Neural
from spend_behavior.data import positive_lines

SETTINGS = {'hidden_sizes': [128, 64], 'dropout': 0., 'weight_decay': .0001,
            'learning_rate': .001, 'batch_size': 1024, 'max_epochs': 200,
            'patience': 20, 'min_delta': .0001}
DATES = ['2018-09-01', '2018-10-01']


def build_matrix(objects, lines, window=180, half_life=90):
    """Log decayed purchase-day counts, with each row's own strict cutoff."""
    events = (lines.assign(day=lines.event_ts.dt.normalize())
              .drop_duplicates(['customer_id', 'day', 'item_id']))
    names = np.array(sorted(events.item_id.unique()), dtype=str)
    codes = {name: i for i, name in enumerate(names)}
    groups = {str(cid): g.sort_values('day') for cid, g in events.groupby('customer_id', sort=False)}
    rows, columns, values = [], [], []
    for cid, snapshots in objects.groupby('customer_id', sort=False):
        events = groups.get(str(cid))
        if events is None:
            continue
        days = events.day.to_numpy(dtype='datetime64[D]').astype(np.int64)
        items = np.array([codes[name] for name in events.item_id])
        for row in snapshots.itertuples():
            t = int(np.datetime64(row.as_of_date, 'D').astype(np.int64))
            lo, hi = np.searchsorted(days, [t-window, t])
            if lo == hi:
                continue
            unique, inverse = np.unique(items[lo:hi], return_inverse=True)
            mass = np.bincount(inverse, weights=np.exp2((days[lo:hi]-t)/half_life))
            rows.extend([row.Index]*len(unique))
            columns.extend(unique.tolist())
            values.extend(np.log1p(mass).tolist())
    matrix = sparse.csr_matrix((values, (rows, columns)), shape=(len(objects), len(names)), dtype=np.float32)
    return matrix, names


class HistorySVD:
    """Vocabulary and factorisation fit only on the supplied training rows."""
    def fit(self, matrix, indices, names, components=32):
        self.columns = np.flatnonzero(matrix[indices].getnnz(axis=0))
        self.names = names[self.columns]
        x = normalize(matrix[indices][:, self.columns], copy=True)
        k = min(components, len(self.columns)-1, len(indices)-1)
        if k < 1:
            raise ValueError('Not enough training histories/items for SVD')
        self.model = TruncatedSVD(n_components=k, n_iter=7, random_state=42).fit(x)
        return self

    def transform(self, matrix, indices):
        x = normalize(matrix[indices][:, self.columns], copy=True)
        values = self.model.transform(x).astype(np.float32)
        if not np.isfinite(values).all():
            raise ValueError('Non-finite customer vectors')
        return values


def indices(objects, cutoff):
    return np.flatnonzero(objects.target_end.le(pd.Timestamp(cutoff)).to_numpy())


def paired_scores(frame, repeats=2000):
    rows = []
    for label, g in [*frame.groupby('date'), ('pooled_SepOct', frame)]:
        a = (g.z_control-g.target_z)**2
        b = (g.z_svd-g.target_z)**2
        grouped = pd.DataFrame({'customer_id': g.customer_id, 'a': a, 'b': b})
        grouped = grouped.groupby('customer_id', sort=True).agg(a=('a', 'sum'), b=('b', 'sum'), n=('a', 'size'))
        sums = grouped.to_numpy(float)
        rng = np.random.default_rng(17000)
        draws = []
        for _ in range(repeats):
            total = sums[rng.integers(0, len(sums), size=len(sums))].sum(axis=0)
            draws.append(np.sqrt(total[1]/total[2])-np.sqrt(total[0]/total[2]))
        lo, hi = np.quantile(draws, [.025, .975])
        rows.append({'date': label, 'n': len(g), 'control_RMSLE': float(np.sqrt(a.mean())),
                     'svd32_RMSLE': float(np.sqrt(b.mean())),
                     'delta': float(np.sqrt(b.mean())-np.sqrt(a.mean())),
                     'conditional_ci_low': float(lo), 'conditional_ci_high': float(hi)})
    return pd.DataFrame(rows)


def run(args):
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    sources = [Path(__file__), *[Path('src')/n for n in
               ['spend_baselines/data.py', 'spend_baselines/suite_data.py',
                'spend_baselines/suite_models.py', 'spend_behavior/data.py']]]
    settings = dict(SETTINGS)
    if args.smoke:
        settings.update(max_epochs=2, patience=1)
    identity = {'inputs': {name: digest(args.data_dir/name) for name in ['train.csv', 'transactions.csv']},
                'source_sha256': {str(p): digest(p) for p in sources}, 'seed': 42,
                'dates': DATES, 'settings': settings, 'components': 32, 'window_days': 180,
                'half_life_days': 90, 'item_value': 'log1p(decayed purchase-day count)',
                'row_normalization': 'L2 over training vocabulary', 'dataset': 'D0',
                'base_features': 'F0', 'target': 'direct log1p, one output',
                'device': args.device, 'threads': args.threads, 'smoke': args.smoke,
                'python': platform.python_version(),
                'libraries': {n: version(n) for n in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'torch']},
                'purpose': 'offline customer representation trial; final preset unchanged',
                'method_sources': {'SVD': 'seminar6', 'MLP': 'seminars3/8'}}
    manifest = out/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        raise ValueError('Run definition changed; use a new output directory')
    atomic_json(manifest, identity)
    train = read_objects(args.data_dir/'train.csv')
    if args.smoke:
        train = train[train.customer_id.isin(sorted(train.customer_id.unique())[:64])].reset_index(drop=True)
    cache = out/'cache'
    cache.mkdir(exist_ok=True)
    ready = cache/'ready.json'
    if ready.exists():
        marker = json.loads(ready.read_text())
        for name, sha in marker['sha256'].items():
            if digest(cache/name) != sha:
                raise ValueError('Corrupt cached histories')
        x = np.load(cache/'rfm.npy', allow_pickle=False)
        history = sparse.load_npz(cache/'histories.npz')
        names = np.array(json.loads((cache/'names.json').read_text()), dtype=str)
    else:
        print('Building causal RFM and purchase-history matrix', flush=True)
        _, _, purchase_history, checks = load_dataset(args.data_dir)
        x = feature_matrix(purchase_history.build_features(train))
        lines, _ = positive_lines(args.data_dir)
        history, names = build_matrix(train, lines)
        np.save(cache/'rfm.npy', x)
        sparse.save_npz(cache/'histories.npz', history)
        (cache/'names.json').write_text(json.dumps(names.tolist()))
        atomic_json(ready, {'rows': len(train), 'items': len(names), 'nnz': history.nnz,
                           'target_checks': checks,
                           'sha256': {n: digest(cache/n) for n in ['rfm.npy', 'histories.npz', 'names.json']}})
    frames = []
    for date in DATES:
        tr = indices(train, date)
        vi = np.flatnonzero(train.as_of_date.eq(pd.Timestamp(date)).to_numpy())
        inner = train.iloc[tr].as_of_date.max()
        ti = indices(train, inner)
        iv = np.flatnonzero(train.as_of_date.eq(inner).to_numpy())
        frame = train.iloc[vi][['Id', 'customer_id']].copy().reset_index(drop=True)
        frame['date'] = date
        frame['target_z'] = np.log1p(train.iloc[vi].spend_30d.to_numpy(float))
        for candidate in ['control', 'svd']:
            dest = out/'jobs'/date/candidate
            dest.mkdir(parents=True, exist_ok=True)
            result_path = dest/'result.json'
            if result_path.exists():
                result = json.loads(result_path.read_text())
                for name, sha in result['sha256'].items():
                    if digest(dest/name) != sha:
                        raise ValueError('Corrupt completed model artifact')
                with np.load(dest/'predictions.npz', allow_pickle=False) as saved:
                    np.testing.assert_array_equal(saved['Id'], frame.Id.to_numpy(str))
                    frame['z_'+candidate] = saved['z']
                print(f'Reused {date} {candidate}', flush=True)
                continue
            start = time.monotonic()
            a, b, c, d = x[ti], x[iv], x[tr], x[vi]
            encoder_metadata = {}
            if candidate == 'svd':
                print(f'{date}: fitting inner/outer SVD on training histories only', flush=True)
                inner_svd = HistorySVD().fit(history, ti, names)
                outer_svd = HistorySVD().fit(history, tr, names)
                a = np.column_stack([a, inner_svd.transform(history, ti)])
                b = np.column_stack([b, inner_svd.transform(history, iv)])
                c = np.column_stack([c, outer_svd.transform(history, tr)])
                vectors = outer_svd.transform(history, vi)
                d = np.column_stack([d, vectors])
                joblib.dump({'columns': outer_svd.columns, 'item_names': outer_svd.names,
                             'model': outer_svd.model}, dest/'encoder.joblib', compress=3)
                np.savez_compressed(dest/'customer_vectors.npz', Id=frame.Id.to_numpy(str),
                                    customer_id=frame.customer_id.to_numpy(str), vectors=vectors)
                encoder_metadata = {'inner_items': len(inner_svd.names), 'outer_items': len(outer_svd.names),
                                    'components': vectors.shape[1],
                                    'explained_variance_ratio': float(outer_svd.model.explained_variance_ratio_.sum())}
            print(f'{date} {candidate}: selecting epochs on {inner.date()}, rows={len(ti)}', flush=True)
            selector = Neural('mlp', 'direct', settings, 42, args.device, args.threads)
            selector.fit(a, train.iloc[ti].spend_30d.to_numpy(float), np.ones(len(ti)),
                         validation=(b, train.iloc[iv].spend_30d.to_numpy(float)))
            epoch = selector.best_epoch
            pd.DataFrame(selector.history).to_csv(dest/'selection.csv', index=False)
            del selector
            model = Neural('mlp', 'direct', settings, 42, args.device, args.threads)
            model.fit(c, train.iloc[tr].spend_30d.to_numpy(float), np.ones(len(tr)), epochs=epoch)
            if model.network.output.out_features != 1:
                raise ValueError('Expected one direct-regression output')
            model.save(dest/'model.pt')
            pd.DataFrame(model.history).to_csv(dest/'refit.csv', index=False)
            z = model.predict(d)
            np.savez_compressed(dest/'predictions.npz', Id=frame.Id.to_numpy(str),
                                customer_id=frame.customer_id.to_numpy(str), z=z)
            frame['z_'+candidate] = z
            score = float(np.sqrt(np.mean((z-frame.target_z.to_numpy())**2)))
            result = {'date': date, 'candidate': candidate, 'selected_epoch': epoch,
                      'n_train': len(tr), 'n_inner': len(ti), 'n_validation': len(vi),
                      'inner_date': str(inner.date()), 'max_target_end_train': str(train.iloc[tr].target_end.max()),
                      'max_target_end_inner': str(train.iloc[ti].target_end.max()),
                      'RMSLE': score, 'seconds': time.monotonic()-start, **encoder_metadata,
                      'sha256': {p.name: digest(p) for p in dest.iterdir() if p.is_file() and p.name != 'result.json'}}
            atomic_json(result_path, result)
            print(f'Completed {date} {candidate}: RMSLE={score:.6f}, epoch={epoch}, seconds={result["seconds"]:.1f}', flush=True)
        frames.append(frame)
        pd.concat(frames).to_csv(out/'predictions.csv.gz', index=False)
    table = paired_scores(pd.concat(frames, ignore_index=True))
    table.to_csv(out/'scores.csv', index=False)
    print(table.to_string(index=False), flush=True)
    atomic_json(out/'status.json', {'complete': True, 'smoke': args.smoke,
                                   'kaggle_upload': False, 'test_predictions_created': False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('data/raw'))
    parser.add_argument('--out-dir', type=Path, default=Path('outputs/customer-svd-v1-cpu'))
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    with threadpool_limits(limits=args.threads):
        run(args)


if __name__ == '__main__':
    main()
