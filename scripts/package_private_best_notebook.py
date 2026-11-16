"""Build a standalone EDA + single-MLP notebook for the measured private winner."""
import ast
import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHAMPION = ROOT / "outputs/champion-private/monthly_l2_003"
DELIVERY = ROOT / "outputs/private-best-notebook-delivery"
EXPECTED_CSV_SHA256 = "2e2208c73231b84bc32b8eae0429b7e58580188e909179313762a6ca8392ef53"


def extract(path, names):
    tree = ast.parse((ROOT / path).read_text())
    return "\n\n".join(
        ast.unparse(node) for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
    )


def main():
    result = json.loads((CHAMPION / "result.json").read_text())
    original_csv = CHAMPION / "submission_monthly_l2_003_cpu.csv"
    assert hashlib.sha256(original_csv.read_bytes()).hexdigest() == EXPECTED_CSV_SHA256
    assert hashlib.sha256((CHAMPION / "model.pt").read_bytes()).hexdigest() == result["sha256"]["model.pt"]
    assert result["selected_epoch"] == 14 and len(result["feature_names"]) == 83
    cells = []

    def add(kind, source, name):
        cell = {"cell_type": kind, "id": name, "metadata": {}, "source": source.strip() + "\n"}
        if kind == "code":
            compile(cell["source"], name, "exec")
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    add("markdown", r"""
# Прогноз покупок на 30 дней: EDA и лучший метод по private

Нужно предсказать `spend_30d` клиента по покупкам **до** `as_of_date`.
Единица наблюдения — клиент и дата прогноза. Качество оценивается RMSLE.

Итоговый метод — **одна MLP**, seed 42, на RFM, сезонности и месячной истории.
У сохранённого результата **Private = 2,69340, Public = 2,66351**.
Это лучший подтверждённый private среди наших отправок. Оценки Kaggle
приведены по сохранённым результатам, а не вычислены этим ноутбуком.

Сначала — короткий EDA, затем признаки, временная проверка и обучение модели.
Ноутбук содержит весь код и работает без исходного репозитория. При подготовке
кода использовался Codex.
""", "introduction")

    add("markdown", """
## 1. Подготовка

Положите четыре официальных файла (`transactions.csv`, `train.csv`, `test.csv`,
`sample_submission.csv`) в `data/raw` и выполните ячейки по порядку.
Путь можно изменить в следующей ячейке.

Зависимости: `numpy==2.3.5`, `pandas==2.2.3`, `scikit-learn==1.7.2`,
`torch==2.5.1`, `matplotlib>=3.9,<4`.
По умолчанию модель обучается с нуля на CPU: именно так получен сохранённый
private-лидер. Для запуска на V100 можно поставить `DEVICE = 'cuda'`.
Траектории обучения на разных устройствах и версиях библиотек могут отличаться.

В архиве также есть исходные веса. С `RETRAIN = False` финальная ячейка
использует их вместо нового обучения; временная проверка управляется отдельно.
""", "setup-description")
    add("code", """
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import random
from pathlib import Path
from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from IPython.display import display

FILES = ["transactions.csv", "train.csv", "test.csv", "sample_submission.csv"]
DATA_DIR = next(
    (p for p in [Path("data/raw"), Path("../data/raw"), Path("../../data/raw")]
     if all((p / name).is_file() for name in FILES)),
    Path("data/raw"),
)
OUTPUT_DIR = Path("private_best_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = "cpu"  # Можно заменить на "cuda".
THREADS = 2
SEED = 42
RUN_VALIDATION = True
RETRAIN = True
FINAL_EPOCHS = 14
SETTINGS = {
    "hidden_sizes": [128, 64], "dropout": 0.1,
    "learning_rate": 0.001, "weight_decay": 0.003,
    "batch_size": 1024, "max_epochs": 200,
    "patience": 20, "min_delta": 0.0001,
}
assert all((DATA_DIR / name).is_file() for name in FILES), DATA_DIR.resolve()
print("Данные:", DATA_DIR.resolve(), "| Устройство:", DEVICE)
""", "imports-and-settings")

    add("code", """
raw = pd.read_csv(DATA_DIR / "transactions.csv", dtype={
    "order_id": "string", "customer_id": "string", "item_id": "string",
    "item_name": "string", "region": "string",
})
train = pd.read_csv(DATA_DIR / "train.csv", dtype={"Id": "string", "customer_id": "string"},
                    parse_dates=["as_of_date", "target_end"])
test = pd.read_csv(DATA_DIR / "test.csv", dtype={"Id": "string", "customer_id": "string"},
                   parse_dates=["as_of_date", "target_end"])
sample = pd.read_csv(DATA_DIR / "sample_submission.csv", dtype={"Id": "string"})
raw["event_ts"] = pd.to_datetime(raw.event_ts)

display(pd.DataFrame([
    {"Таблица": "transactions", "Строк": len(raw), "Клиентов": raw.customer_id.nunique(),
     "Первая дата": raw.event_ts.min(), "Последняя дата": raw.event_ts.max()},
    {"Таблица": "train", "Строк": len(train), "Клиентов": train.customer_id.nunique(),
     "Первая дата": train.as_of_date.min(), "Последняя дата": train.as_of_date.max()},
    {"Таблица": "test", "Строк": len(test), "Клиентов": test.customer_id.nunique(),
     "Первая дата": test.as_of_date.min(), "Последняя дата": test.as_of_date.max()},
]).set_index("Таблица"))
""", "load-data")

    add("markdown", """
## 2. Короткий EDA

Посмотрим на пропуски, повторяющиеся записи, долю нулевых ответов и разброс трат.
Для сумм удобна шкала `log1p`: несколько крупных покупок сильно растягивают
исходное распределение. Соседние даты одного клиента зависимы, поэтому далее
используется временная проверка.
""", "eda-description")
    add("code", """
display(pd.DataFrame({
    "Пропусков": raw.isna().sum(),
    "Доля пропусков": raw.isna().mean(),
}).query("Пропусков > 0"))
print("Полных дубликатов транзакций:", int(raw.duplicated().sum()))
print(f"Доля нулевых train-ответов: {train.spend_30d.eq(0).mean():.1%}")
display(train.spend_30d.describe(percentiles=[.5, .9, .99]).to_frame("spend_30d"))
assert train.Id.is_unique and test.Id.is_unique and sample.Id.is_unique
assert not (set(train.Id) & set(test.Id))
assert train.target_end.sub(train.as_of_date).eq(pd.Timedelta(days=30)).all()
""", "eda-tables")

    cleaning = extract("src/spend_baselines/data.py", {"canonical_ids", "clean_transactions"})
    add("markdown", """
Для признаков оставляем положительные товарные покупки, убираем полные
дубликаты, возвраты и служебные позиции. Складываем покупки клиента за день.
Такая очистка далее проверяется восстановлением известных train-ответов.
""", "cleaning-description")
    add("code", cleaning + "\n\ndays, cleaning_stats = clean_transactions(raw)\n"
        "display(pd.Series(cleaning_stats, name='После очистки').to_frame())", "cleaning")
    add("code", """
by_date = train.groupby("as_of_date").spend_30d.agg(
    zero_share=lambda x: x.eq(0).mean(), mean="mean", n="size",
)
month_sales = days.groupby(days.date.dt.to_period("M")).amount.sum()

fig, axes = plt.subplots(1, 3, figsize=(15, 3.4))
axes[0].hist(np.log1p(train.spend_30d), bins=45, color="#4477aa")
axes[0].set(title="Target distribution", xlabel="log1p(spend_30d)", ylabel="Rows")
axes[1].plot(by_date.index, by_date.zero_share, marker=".", color="#228833")
axes[1].set(title="Zero-target share by forecast date", ylabel="Share", ylim=(0, 1))
axes[2].plot(month_sales.index.to_timestamp(), month_sales / 1e6, marker=".", color="#cc6677")
axes[2].set(title="Observed monthly purchase totals", ylabel="Spend, millions")
for ax in axes[1:]:
    ax.tick_params(axis="x", rotation=45)
fig.tight_layout()
plt.show()

print("Нулей много; предсказывать только средний чек недостаточно.")
print("Активность меняется по времени — используем календарь и историю клиента.")
""", "eda-plots")
    add("markdown", """
Месячные суммы здесь — описательная статистика наблюдаемых данных, а не
оценка отдельного сезонного эффекта: меняются и состав клиентов, и их активность.
Для модели важны история конкретного клиента и корректная дата её окончания.

## 3. Признаки лучшего метода

Используем **83 признака**, построенных только из прошлых транзакций:

| Блок | Число | Содержание |
|---|---:|---|
| RFM и календарь | 15 | Давность и частота покупок, суммы за 30–180 дней, размер чека, sin/cos месяца |
| Сезонность | 21 | Покупки в сопоставимых окнах год назад, отношения активности и индикаторы месяца |
| Месячная история | 47 | 12 завершённых 30-дневных окон, доля пустых окон, средние, разброс и взвешенные средние |

Все транзакционные окна заканчиваются **до `as_of_date`**. При отсутствии
покрытия истории ставим `NaN`; наблюдаемое отсутствие покупок — ноль.
`spend_30d` из официального train не используется как входной признак.
""", "features-description")

    add("code", """
RFM_COLUMNS = [
    "recency", "tenure", "n_30", "n_60", "n_90", "n_180",
    "sum_30", "sum_60", "sum_90", "sum_180", "last_amount",
    "mean_amount_180", "median_amount_180", "month_sin", "month_cos",
]

class PurchaseHistory:
    def __init__(self, days):
        self.customers = {}
        for cid, group in days.groupby("customer_id"):
            group = group.sort_values("date")
            dates = group.date.to_numpy(dtype="datetime64[D]").astype(np.int64)
            amounts = group.amount.to_numpy(dtype=float)
            self.customers[str(cid)] = (dates, amounts, np.r_[0., np.cumsum(amounts)])

    def build_features(self, objects):
        rows = []
        empty = (np.array([], dtype=int), np.array([]), np.array([0.]))
        for row in objects.itertuples():
            dates, amounts, cumulative = self.customers.get(str(row.customer_id), empty)
            t = int(np.datetime64(row.as_of_date, "D").astype(np.int64))
            right = int(np.searchsorted(dates, t, side="left"))
            f = {"recency": float(t - dates[right - 1]) if right else 181.,
                 "tenure": float(t - dates[0]) if right else 0.}
            for window in [30, 60, 90, 180]:
                left = int(np.searchsorted(dates, t - window, side="left"))
                f[f"n_{window}"] = right - left
                f[f"sum_{window}"] = float(cumulative[right] - cumulative[left])
            left = int(np.searchsorted(dates, t - 180, side="left"))
            a = amounts[left:right]
            f.update({
                "last_amount": float(amounts[right - 1]) if right else 0.,
                "mean_amount_180": float(a.mean()) if len(a) else 0.,
                "median_amount_180": float(np.median(a)) if len(a) else 0.,
                "month_sin": float(np.sin(2 * np.pi * row.as_of_date.month / 12)),
                "month_cos": float(np.cos(2 * np.pi * row.as_of_date.month / 12)),
            })
            rows.append(f)
        return pd.DataFrame(rows, index=objects.index)

    def reconstruct_targets(self, objects):
        result = np.zeros(len(objects))
        for i, row in enumerate(objects.itertuples()):
            h = self.customers.get(str(row.customer_id))
            if h is not None:
                left, right = np.searchsorted(
                    h[0], np.array([row.as_of_date, row.target_end], dtype="datetime64[D]").astype(np.int64),
                    side="left",
                )
                result[i] = np.round(h[2][right] - h[2][left], 2)
        return result

history = PurchaseHistory(days)
rebuilt = history.reconstruct_targets(train)
np.testing.assert_allclose(rebuilt, train.spend_30d.to_numpy(), rtol=0, atol=.011)
print("Проверено train-ответов:", len(train), "| Максимальная ошибка:", np.abs(rebuilt - train.spend_30d).max())
""", "purchase-history")

    season_columns = result["feature_names"][15:36]
    add("code", "SEASON_COLUMNS = " + repr(season_columns) + "\n\n" +
        extract("scripts/course10_forest_followup.py", {"seasonal_features"}), "seasonal-features")
    add("code", (ROOT / "src/spend_course10/history_features.py").read_text(), "monthly-features")
    add("code", """
objects = pd.concat([
    train.assign(official=True, is_test=False),
    test.assign(official=False, is_test=True),
], ignore_index=True).sort_values(["as_of_date", "customer_id"]).reset_index(drop=True)

rfm = history.build_features(objects)[RFM_COLUMNS].to_numpy(dtype=float)
for j, name in enumerate(RFM_COLUMNS):
    if name not in {"month_sin", "month_cos"}:
        rfm[:, j] = np.log1p(rfm[:, j])
history_start = days.date.min()
season = seasonal_features(objects, history, history_start)
monthly = monthly_history_features(objects, history, history_start)
X = np.c_[rfm.astype(np.float32), season, monthly]
feature_names = RFM_COLUMNS + SEASON_COLUMNS + HISTORY_COLUMNS
assert X.shape == (len(objects), 83) and not np.isinf(X).any()
assert "spend_30d" not in feature_names
display(pd.DataFrame({"Блок": ["RFM", "Сезонность", "Месячная история"], "Признаков": [15, 21, 47]}))
print("Матрица:", X.shape)
""", "feature-matrix")

    add("markdown", r"""
## 4. Одна MLP для `log1p` трат

Минимизируем MSE для $z=\log(1+y)$ — квадрат RMSLE:

$$L=\frac{1}{N}\sum_i(f(x_i)-\log(1+y_i))^2.$$

Пропуски заполняем медианами **обучающей части**, добавляем флаги пропусков
и стандартизуем по ней же. В полном рефите получается 103 входа:
83 исходных признака + 20 флагов. Архитектура:
`103 → Linear(128) → ReLU → Dropout(0.1) → Linear(64) → ReLU → Dropout(0.1) → Linear(1)`.

Adam, learning rate 0,001, weight decay 0,003, batch size 1024, seed 42.
При обучении выход линейный. При предсказании ограничиваем log-прогноз снизу
нулём и возвращаем `expm1`. Это одна модель и один seed.
""", "mlp-description")
    add("code", """
def fit_preprocessing(x):
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    a = imputer.fit_transform(x)
    missing = np.flatnonzero(np.isnan(x).any(axis=0))
    a = np.c_[a, np.isnan(x[:, missing])]
    scaler = StandardScaler().fit(a, sample_weight=np.ones(len(x)))
    return {"fill": imputer.statistics_.tolist(), "missing": missing.tolist(),
            "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()}

def transform(x, state):
    a = np.where(np.isnan(x), np.asarray(state["fill"]), x)
    a = np.c_[a, np.isnan(x[:, state["missing"]])]
    return ((a - np.asarray(state["mean"])) / np.asarray(state["scale"])).astype(np.float32)

class MLP(torch.nn.Module):
    def __init__(self, n_inputs):
        super().__init__()
        self.body = torch.nn.Sequential(
            torch.nn.Linear(n_inputs, 128), torch.nn.ReLU(), torch.nn.Dropout(.1),
            torch.nn.Linear(128, 64), torch.nn.ReLU(), torch.nn.Dropout(.1),
        )
        self.output = torch.nn.Linear(64, 1)

    def forward(self, x):
        return self.output(self.body(x))

def predict_log(fitted, x):
    fitted["network"].eval()
    a = torch.from_numpy(transform(x, fitted["preprocessing"])).to(DEVICE)
    with torch.no_grad():
        z = torch.cat([fitted["network"](chunk)[:, 0].clamp_min(0) for chunk in a.split(4096)])
    return z.cpu().numpy().astype(float)

def fit_mlp(indices, validation_indices=None, epochs=None):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(THREADS)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    state = fit_preprocessing(X[indices])
    tx = torch.from_numpy(transform(X[indices], state)).to(DEVICE)
    y = objects.iloc[indices].spend_30d.to_numpy(float)
    ty = torch.as_tensor(y, dtype=torch.float32, device=DEVICE)
    tw = torch.ones(len(indices), dtype=torch.float32, device=DEVICE)
    network = MLP(tx.shape[1]).to(DEVICE)
    optimizer = torch.optim.Adam(network.parameters(), lr=SETTINGS["learning_rate"],
                                 weight_decay=SETTINGS["weight_decay"])
    fitted = {"network": network, "preprocessing": state, "history": [], "best_epoch": 0}
    best, best_state, stale = np.inf, None, 0

    for epoch in range(epochs or SETTINGS["max_epochs"]):
        network.train()
        total_loss = 0.
        for batch in torch.randperm(len(tx), device=DEVICE).split(SETTINGS["batch_size"]):
            optimizer.zero_grad(set_to_none=True)
            raw_z = network(tx[batch])[:, 0]
            loss = (((raw_z - torch.log1p(ty[batch])) ** 2) * tw[batch]).sum() / tw[batch].sum().clamp_min(1e-12)
            assert torch.isfinite(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 10.)
            optimizer.step()
            total_loss += float(loss.detach()) * len(batch)
        record = {"epoch": epoch + 1, "train_loss": total_loss / len(tx)}

        if validation_indices is not None:
            z = predict_log(fitted, X[validation_indices])
            target = np.log1p(objects.iloc[validation_indices].spend_30d.to_numpy(float))
            score = float(np.sqrt(np.mean((z - target) ** 2)))
            record["inner_RMSLE"] = score
            if score < best - SETTINGS["min_delta"]:
                best, stale = score, 0
                fitted["best_epoch"] = epoch + 1
                best_state = deepcopy(network.state_dict())
            else:
                stale += 1
        fitted["history"].append(record)
        if validation_indices is not None and stale >= SETTINGS["patience"]:
            break

    if best_state is not None:
        network.load_state_dict(best_state)
    else:
        fitted["best_epoch"] = len(fitted["history"])
    network.eval()
    return fitted
""", "mlp-code")
    add("markdown", """
## 5. Временная проверка

Для прогноза на дату `cutoff` берём только train-объекты с
`target_end <= cutoff`: их 30-дневный ответ к этой дате уже известен.
Эпоху выбираем на более ранней внутренней дате, после чего обучаем новую
сеть с нуля на всей допустимой обучающей части. Медианы и стандартизация
пересчитываются для каждого обучения.

Проверяем сентябрь и октябрь 2018 года. Эти даты уже использовались при
исследовании; результат не является оценкой на новом независимом holdout.
Общий RMSLE считаем по всем объектам двух дат, а не средним двух RMSLE.
""", "validation-description")
    add("code", """
def training_indices(cutoff):
    return np.flatnonzero((objects.official & objects.target_end.le(pd.Timestamp(cutoff))).to_numpy())

def temporal_prediction(cutoff):
    cutoff = pd.Timestamp(cutoff)
    tr = training_indices(cutoff)
    vi = np.flatnonzero((objects.official & objects.as_of_date.eq(cutoff)).to_numpy())
    inner_date = objects.iloc[tr].as_of_date.max()
    ti = training_indices(inner_date)
    iv = np.flatnonzero((objects.official & objects.as_of_date.eq(inner_date)).to_numpy())
    assert objects.iloc[tr].target_end.max() <= cutoff
    assert objects.iloc[ti].target_end.max() <= inner_date
    selector = fit_mlp(ti, validation_indices=iv)
    epochs = selector["best_epoch"]
    del selector
    fitted = fit_mlp(tr, epochs=epochs)
    z = predict_log(fitted, X[vi])
    return vi, z, epochs

validation_scores = []
if RUN_VALIDATION:
    for date in ["2018-09-01", "2018-10-01"]:
        vi, z, epochs = temporal_prediction(date)
        errors = (z - np.log1p(objects.iloc[vi].spend_30d.to_numpy(float))) ** 2
        validation_scores.append({"date": date, "n": len(vi), "epochs": epochs,
                                  "RMSLE": float(np.sqrt(errors.mean())),
                                  "squared_error": float(errors.sum())})
    display(pd.DataFrame(validation_scores).drop(columns="squared_error"))
    pooled = np.sqrt(sum(r["squared_error"] for r in validation_scores) /
                     sum(r["n"] for r in validation_scores))
    print(f"Общий Sep–Oct RMSLE: {pooled:.6f}")
""", "temporal-validation")

    add("markdown", """
## 6. Полный рефит и `submission.csv`

У выбранного private-лидера число эпох на последней внутренней проверке
(октябрь) равно **14**. Фиксируем его и обучаем новую сеть на всех 50 852
известных train-объектах, доступных к ноябрьскому тесту. Датасет не расширяем,
прогнозы разных моделей не усредняем. Сохранённые веса из архива позволяют
отдельно воспроизвести тот же предиктор без обучения.
""", "refit-description")
    add("code", """
vi = np.flatnonzero(objects.is_test.to_numpy())
cutoff = objects.iloc[vi].as_of_date.min()
tr = training_indices(cutoff)
assert len(tr) == 50852 and objects.iloc[tr].target_end.max() <= cutoff

if RETRAIN:
    final_model = fit_mlp(tr, epochs=FINAL_EPOCHS)
    checkpoint = {
        "state_dict": {k: v.detach().cpu() for k, v in final_model["network"].state_dict().items()},
        "preprocessing": final_model["preprocessing"], "settings": SETTINGS,
        "seed": SEED, "epochs": FINAL_EPOCHS, "feature_names": feature_names,
    }
    torch.save(checkpoint, OUTPUT_DIR / "model_retrained.pt")
else:
    checkpoint = torch.load("best_private_model.pt", map_location="cpu", weights_only=False)
    network = MLP(len(checkpoint["preprocessing"]["mean"])).to(DEVICE)
    network.load_state_dict(checkpoint["state_dict"])
    final_model = {"network": network, "preprocessing": checkpoint["preprocessing"]}

z_test = predict_log(final_model, X[vi])
assert len(final_model["preprocessing"]["missing"]) == 20
print("Входов в сеть:", len(final_model["preprocessing"]["mean"]))
print("Параметров:", sum(p.numel() for p in final_model["network"].parameters()))

submission = pd.DataFrame({"Id": objects.iloc[vi].Id.to_numpy(str),
                           "spend_30d": np.maximum(np.expm1(z_test), 0)})
assert submission.Id.is_unique and set(submission.Id) == set(sample.Id)
submission = submission.set_index("Id").loc[sample.Id].reset_index()
assert len(submission) == 3188 and np.array_equal(submission.Id.to_numpy(str), sample.Id.to_numpy(str))
assert np.isfinite(submission.spend_30d).all() and submission.spend_30d.ge(0).all()
submission_path = OUTPUT_DIR / "submission.csv"
submission.to_csv(submission_path, index=False)
print("Сохранён:", submission_path.resolve(), "| Строк:", len(submission))

reference_path = Path("submission_reference.csv")
if reference_path.is_file():
    reference = pd.read_csv(reference_path, dtype={"Id": "string"}).set_index("Id").loc[sample.Id]
    delta = np.max(np.abs(np.log1p(submission.spend_30d.to_numpy()) - np.log1p(reference.spend_30d.to_numpy())))
    print(f"Максимальное отличие от исходного CSV в log-шкале: {delta:.3g}")
""", "final-refit-and-submission")

    nb = {"nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python", "version": "3.12"}}, "cells": cells}
    source = ROOT / "notebooks/private_best_solution.ipynb"
    source.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n")
    DELIVERY.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, DELIVERY / "solution.ipynb")
    shutil.copyfile(CHAMPION / "model.pt", DELIVERY / "best_private_model.pt")
    shutil.copyfile(original_csv, DELIVERY / "submission_reference.csv")
    requirements = "numpy==2.3.5\npandas==2.2.3\nscikit-learn==1.7.2\ntorch==2.5.1\nmatplotlib>=3.9,<4\n"
    (DELIVERY / "requirements.txt").write_text(requirements)
    (DELIVERY / "README.txt").write_text(
        "Откройте solution.ipynb. Положите четыре официальных CSV в data/raw или задайте DATA_DIR.\n"
        "Run All выполняет EDA, временную проверку и обучение выбранной MLP с нуля.\n"
        "Результат: private_best_outputs/submission.csv. Для inference из исходных весов:\n"
        "RETRAIN=False; RUN_VALIDATION=False. Никакой автоматической отправки в Kaggle.\n"
        "Исходный измеренный CSV: submission_reference.csv, Private2.69340/Public2.66351.\n"
    )
    print(source)
    print(DELIVERY)


if __name__ == "__main__":
    main()
