"""
02_eda.py
=========
Explorative Datenanalyse des Beziehungs-Panels:
- deskriptive Statistiken
- Verteilung von Zeit- und Ereignisvariable
- Korrelationen zwischen Praediktoren
- erste Kaplan-Meier-Kurve (Gesamtstichprobe + nach Ehestatus)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

DATA_PATH = Path(__file__).parent / "data" / "relationship_panel.csv"
OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["figure.dpi"] = 110
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False


def descriptive_summary(df: pd.DataFrame):
    numeric_cols = ["bonding_index", "conflict_index", "age_partner_a",
                     "age_diff_abs", "monthly_income_eur",
                     "shared_activities_per_month", "time_years"]
    print("=== Deskriptive Statistik ===")
    print(df[numeric_cols].describe().round(2).to_string())

    print("\n=== Kategoriale Verteilungen ===")
    for col in ["education", "married", "cohabiting", "has_children"]:
        print(f"\n{col}:")
        print(df[col].value_counts(normalize=True).round(3).to_string())

    print(f"\nZensierungsrate: {(1 - df['event'].mean()):.1%}")
    print(f"Beobachtete Trennungen: {df['event'].sum()} von {len(df)}")


def plot_time_event_distribution(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(df["time_years"], bins=20, color="#4C72B0", edgecolor="white")
    axes[0].set_xlabel("Beobachtungsdauer (Jahre)")
    axes[0].set_ylabel("Anzahl Paare")
    axes[0].set_title("Verteilung der Beobachtungsdauer")

    event_counts = df["event"].value_counts().sort_index()
    axes[1].bar(["zensiert (0)", "Trennung (1)"], event_counts.values,
                color=["#55A868", "#C44E52"])
    axes[1].set_title("Ereignisstatus")
    axes[1].set_ylabel("Anzahl Paare")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "eda_time_event_distribution.png")
    plt.close(fig)


def plot_correlation_heatmap(df: pd.DataFrame):
    numeric_cols = ["bonding_index", "conflict_index", "age_partner_a",
                     "age_diff_abs", "monthly_income_eur",
                     "shared_activities_per_month", "n_children", "time_years"]
    corr = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(numeric_cols)))
    ax.set_xticklabels(numeric_cols, rotation=45, ha="right")
    ax.set_yticks(range(len(numeric_cols)))
    ax.set_yticklabels(numeric_cols)
    for i in range(len(numeric_cols)):
        for j in range(len(numeric_cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                    color="white" if abs(corr.iloc[i, j]) > 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson-Korrelation")
    ax.set_title("Korrelationen zwischen Praediktoren")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "eda_correlation_heatmap.png")
    plt.close(fig)


def plot_kaplan_meier_overall(df: pd.DataFrame):
    kmf = KaplanMeierFitter()
    kmf.fit(df["time_years"], event_observed=df["event"], label="Gesamtstichprobe")

    fig, ax = plt.subplots(figsize=(7, 5))
    kmf.plot_survival_function(ax=ax, ci_show=True)
    ax.set_xlabel("Beziehungsdauer (Jahre)")
    ax.set_ylabel("Geschaetzte Wahrscheinlichkeit, noch zusammen zu sein")
    ax.set_title("Kaplan-Meier-Schaetzer: Beziehungsstabilitaet")
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "eda_km_overall.png")
    plt.close(fig)

    for years in [1, 3, 5, 10]:
        s = kmf.survival_function_at_times(years).values[0]
        print(f"  Ueberlebenswahrscheinlichkeit nach {years} Jahren: {s:.1%}")


def plot_kaplan_meier_by_group(df: pd.DataFrame, group_col: str, labels: dict):
    fig, ax = plt.subplots(figsize=(7, 5))
    kmf = KaplanMeierFitter()
    groups = []
    for value, label in labels.items():
        mask = df[group_col] == value
        kmf.fit(df.loc[mask, "time_years"], event_observed=df.loc[mask, "event"], label=label)
        kmf.plot_survival_function(ax=ax, ci_show=False)
        groups.append(df.loc[mask])

    ax.set_xlabel("Beziehungsdauer (Jahre)")
    ax.set_ylabel("Ueberlebenswahrscheinlichkeit")
    ax.set_title(f"Kaplan-Meier nach Gruppe: {group_col}")
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"eda_km_by_{group_col}.png")
    plt.close(fig)

    if len(groups) == 2:
        result = logrank_test(
            groups[0]["time_years"], groups[1]["time_years"],
            event_observed_A=groups[0]["event"], event_observed_B=groups[1]["event"]
        )
        print(f"  Log-Rank-Test ({group_col}): p = {result.p_value:.4f}")


def main():
    df = pd.read_csv(DATA_PATH)

    descriptive_summary(df)

    print("\n=== Plots ===")
    plot_time_event_distribution(df)
    plot_correlation_heatmap(df)

    print("\nKaplan-Meier (gesamt):")
    plot_kaplan_meier_overall(df)

    print("\nKaplan-Meier nach Ehestatus:")
    plot_kaplan_meier_by_group(df, "married", {0: "nicht verheiratet", 1: "verheiratet"})

    print("\nKaplan-Meier nach Kinderstatus:")
    plot_kaplan_meier_by_group(df, "has_children", {0: "keine Kinder", 1: "mit Kindern"})

    print(f"\nAlle Plots gespeichert in: {OUT_DIR}")


if __name__ == "__main__":
    main()
