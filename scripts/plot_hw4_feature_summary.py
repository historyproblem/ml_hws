"""Two evidence panels for the HW4 feature analysis."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/hw4-features-20261116"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 2, figsize=(13.7, 5.1), gridspec_kw={"width_ratios": [1.15, 1]})
fig.patch.set_facecolor("#f8fafc")
for ax in axes:
    ax.set_facecolor("#f8fafc")
    ax.grid(axis="y", color="#dce3ed", linewidth=.7, zorder=0)
    ax.spines[["left", "bottom"]].set_color("#aab7c7")
risk = pd.read_csv(OUT / "eda/risk_tables.csv")
risk = risk.loc[risk.feature == "history_violation_mean"].set_index("level")
order = ["0", "(0,1]", "(1,2]", "(2,3]", "(3,5]", "(5,8]", "(8,12]", ">12"]
risk = risk.loc[order]
ax = axes[0]
bars = ax.bar(np.arange(len(order)), risk.rate, color="#4c78a8", width=.7, zorder=3)
for bar, value, n in zip(bars, risk.rate, risk.n):
    ax.text(bar.get_x() + bar.get_width()/2, value + .012, f"{value:.1%}", ha="center", fontsize=9)
    ax.text(bar.get_x() + bar.get_width()/2, .008, f"n={n}", ha="center", va="bottom", color="white", fontsize=7, rotation=90)
ax.set_xticks(np.arange(len(order)), ["0", "0–1", "1–2", "2–3", "3–5", "5–8", "8–12", ">12"])
ax.set_ylim(0, .55)
ax.yaxis.set_major_formatter(PercentFormatter(1))
ax.set_ylabel("Частота будущего fail")
ax.set_xlabel("Среднее число записанных кодов за прошлый визит")
ax.set_title("Главный индивидуальный сигнал — число кодов", loc="left", fontweight="bold", pad=16)
ax.text(0, -.25, "16 916 объектов до 2024Q1. Связь наблюдательная.\n0 = нет записанных кодов; это не доказательство отсутствия нарушений.",
        transform=ax.transAxes, fontsize=8, color="#526174", va="top")
fold = pd.read_csv(OUT / "matched_seed_fold_metrics.csv")
ax = axes[1]
quarters = ["2023-07-01", "2023-10-01", "2024-01-01"]
for case, label, color in [("baseline", "Исходные 274 признака", "#6b7789"),
                            ("plus_context", "+72 признака других заведений", "#148874")]:
    table = fold.loc[fold.case == case].groupby("forecast_period").ap.agg(["mean", "min", "max"]).loc[quarters]
    ax.plot(np.arange(3), table["mean"], color=color, marker="o", linewidth=2.4, label=label, zorder=4)
    ax.fill_between(np.arange(3), table["min"], table["max"], color=color, alpha=.12, linewidth=0)
    if case == "plus_context":
        for x, val in enumerate(table["mean"]):
            ax.text(x, val + .0022, f"{val:.4f}", color=color, ha="center", fontsize=9)
ax.set_xticks(np.arange(3), ["2023Q3", "2023Q4", "2024Q1"])
ax.set_ylim(.225, .29)
ax.set_ylabel("Average Precision на будущем квартале")
ax.set_title("Контекст улучшил все 9 парных сравнений", loc="left", fontweight="bold", pad=16)
ax.legend(frameon=False, loc="lower left", fontsize=9)
ax.text(0, -.25, "Среднее по 3 seed; полоса — их диапазон, не доверительный интервал.\nСредняя прибавка AP +0.00786; 2024Q2/Q3 не оценивались.",
        transform=ax.transAxes, fontsize=8, color="#526174", va="top")
fig.subplots_adjust(left=.065, right=.985, top=.87, bottom=.27, wspace=.28)
fig.savefig(OUT / "feature_summary.png", dpi=170, facecolor=fig.get_facecolor())
fig.savefig(OUT / "feature_summary.svg", facecolor=fig.get_facecolor())
print(OUT / "feature_summary.png")
