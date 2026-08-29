"""
03_cox_model.py
================
Hauptmodell der Survival Analysis: Cox-Proportional-Hazards-Modell.

Schaetzt Hazard Ratios fuer die Einflussfaktoren auf das Trennungsrisiko
und leitet daraus individuelle Ueberlebensfunktionen ab.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test

DATA_PATH = Path(__file__).parent / "data" / "relationship_panel.csv"
OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["figure.dpi"] = 110
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

FEATURE_COLS = [
    "conflict_index", "bonding_index", "satisfaction_index", "age_diff_abs",
    "monthly_income_eur", "married", "cohabiting", "has_children",
    "n_children", "shared_activities_per_month", "education_avg_ord",
]


def prepare_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    model_df = df[FEATURE_COLS + ["time_years", "event"]].copy()
    # Skalierung des Einkommens auf Tausend Euro fuer interpretierbarere HRs
    model_df["monthly_income_keur"] = model_df["monthly_income_eur"] / 1000
    model_df = model_df.drop(columns=["monthly_income_eur"])
    return model_df


def fit_cox_model(model_df: pd.DataFrame) -> CoxPHFitter:
    cph = CoxPHFitter()
    cph.fit(model_df, duration_col="time_years", event_col="event")
    return cph


def print_summary(cph: CoxPHFitter):
    print("=== Cox-PH-Modell: Zusammenfassung ===")
    summary = cph.summary[["coef", "exp(coef)", "se(coef)", "p", "exp(coef) lower 95%", "exp(coef) upper 95%"]]
    print(summary.round(3).to_string())

    print(f"\nConcordance Index (C-Index): {cph.concordance_index_:.3f}")
    print(f"Log-Likelihood-Ratio-Test p-Wert: {cph.log_likelihood_ratio_test().p_value:.5f}")

    print("\n--- Interpretation der Hazard Ratios ---")
    for var, row in cph.summary.iterrows():
        hr = row["exp(coef)"]
        p = row["p"]
        richtung = "erhoeht" if hr > 1 else "senkt"
        signifikanz = "signifikant" if p < 0.05 else "nicht signifikant"
        pct = abs(hr - 1) * 100
        print(f"  {var}: HR={hr:.2f} -> {richtung} das Trennungsrisiko um {pct:.1f}% "
              f"pro Einheit ({signifikanz}, p={p:.3f})")


def check_proportional_hazards(cph: CoxPHFitter, model_df: pd.DataFrame):
    print("\n=== Test der Proportional-Hazards-Annahme (Schoenfeld-Residuen) ===")
    results = proportional_hazard_test(cph, model_df, time_transform="rank")
    print(results.summary.round(3).to_string())
    violations = results.summary[results.summary["p"] < 0.05]
    if len(violations) > 0:
        print(f"\nWARNUNG: PH-Annahme moeglicherweise verletzt fuer: {list(violations.index)}")
    else:
        print("\nKeine Verletzung der PH-Annahme auf 5%-Niveau erkennbar.")


def plot_hazard_ratios(cph: CoxPHFitter):
    fig, ax = plt.subplots(figsize=(7, 5))
    cph.plot(ax=ax)
    ax.set_title("Cox-PH: Hazard Ratios (log-Skala) mit 95%-KI")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "cox_hazard_ratios.png")
    plt.close(fig)


def plot_survival_curves_by_conflict(cph: CoxPHFitter, model_df: pd.DataFrame):
    """Individuelle Ueberlebenskurven fuer niedriges/mittleres/hohes Konfliktniveau,
    bei sonst mittleren (Median-)Werten der uebrigen Praediktoren."""
    other_cols = [c for c in model_df.columns if c not in ("conflict_index", "time_years", "event")]
    baseline = model_df[other_cols].median()
    scenarios = pd.DataFrame([baseline.values] * 3, columns=baseline.index)
    scenarios.insert(0, "conflict_index", [0.5, 2.0, 3.5])  # niedrig / mittel / hoch

    fig, ax = plt.subplots(figsize=(7, 5))
    cph.predict_survival_function(scenarios).plot(ax=ax)
    ax.legend(["niedriger Konflikt (0.5)", "mittlerer Konflikt (2.0)", "hoher Konflikt (3.5)"])
    ax.set_xlabel("Beziehungsdauer (Jahre)")
    ax.set_ylabel("Geschaetzte Ueberlebenswahrscheinlichkeit")
    ax.set_title("Vorhergesagte Beziehungsstabilitaet nach Konfliktniveau")
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "cox_survival_by_conflict.png")
    plt.close(fig)


def main():
    df = pd.read_csv(DATA_PATH)
    model_df = prepare_model_frame(df)

    cph = fit_cox_model(model_df)
    print_summary(cph)
    check_proportional_hazards(cph, model_df)

    plot_hazard_ratios(cph)
    plot_survival_curves_by_conflict(cph, model_df)

    cph.summary.to_csv(OUT_DIR / "cox_model_summary.csv")
    print(f"\nModellzusammenfassung und Plots gespeichert in: {OUT_DIR}")

    return cph, model_df


if __name__ == "__main__":
    main()
