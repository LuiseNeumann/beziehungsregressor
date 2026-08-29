"""
04_alternative_models.py
==========================
Alternative Modellierungsansaetze zum Cox-PH-Modell:

1. Logistische Regression: Trennung innerhalb von X Jahren (binaer, X=5)
2. Random Forest Klassifikation
3. Gradient Boosting Klassifikation

Alle drei Modelle nutzen dieselben Praediktoren wie das Cox-Modell, um
Vergleichbarkeit herzustellen. Bewertung ueber Cross-Validation (ROC-AUC,
Accuracy) sowie Feature Importances (RF/GB) im Vergleich zu den Cox-HRs.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix

DATA_PATH = Path(__file__).parent / "data" / "relationship_panel.csv"
OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["figure.dpi"] = 110
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

HORIZON_YEARS = 5
FEATURES = [
    "conflict_index", "bonding_index", "satisfaction_index", "age_diff_abs",
    "monthly_income_eur", "married", "cohabiting", "has_children",
    "n_children", "shared_activities_per_month", "education_avg_ord",
]
RANDOM_STATE = 42


def build_binary_target(df: pd.DataFrame, horizon: int = HORIZON_YEARS) -> pd.DataFrame:
    """
    Definiert die binaere Zielvariable "Trennung innerhalb von `horizon`
    Jahren". Paare, die vor Ablauf des Horizonts zensiert wurden (also noch
    nicht so lange beobachtet wurden UND kein Ereignis hatten), werden aus
    der Analyse ausgeschlossen, da fuer sie der Status zum Zeitpunkt
    `horizon` unbekannt ist (uebliches Vorgehen bei Umwandlung von
    Survival- in Klassifikationsdaten).
    """
    known = df[(df["time_years"] >= horizon) | (df["event"] == 1)].copy()
    known["separated_within_horizon"] = (
        (known["time_years"] <= horizon) & (known["event"] == 1)
    ).astype(int)
    excluded = len(df) - len(known)
    print(f"[Hinweis] {excluded} Faelle mit unklarem {horizon}-Jahres-Status ausgeschlossen "
          f"(zensiert vor Ablauf des Horizonts).")
    return known


def prepare_features(df: pd.DataFrame):
    X = df[FEATURES].copy()
    y = df["separated_within_horizon"].values
    return X, y


def train_classifiers_for_horizon(full_df: pd.DataFrame, horizon: float) -> dict:
    """
    Trainiert LogReg/RF/GB fuer einen beliebigen Zeithorizont (Jahre) und
    gibt die fertig trainierten Objekte zurueck. Wird von
    06_predict_own_relationship.py fuer das Modell-Ensemble genutzt, damit
    an jedem Zeitpunkt (1/3/5/10 Jahre) alle vier Modelltypen (inkl. Cox)
    verglichen werden koennen -- nicht nur bei dem fest antrainierten
    5-Jahres-Horizont.
    """
    horizon_df = build_binary_target(full_df, horizon=horizon)
    X, y = prepare_features(horizon_df)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_scaled, y)

    rf = RandomForestClassifier(n_estimators=300, max_depth=4,
                                 min_samples_leaf=5, random_state=RANDOM_STATE)
    rf.fit(X, y)

    gb = GradientBoostingClassifier(n_estimators=200, max_depth=2,
                                     learning_rate=0.05, random_state=RANDOM_STATE)
    gb.fit(X, y)

    return {"logreg": logreg, "scaler": scaler, "rf": rf, "gb": gb, "n_train": len(y)}


def evaluate_logistic_regression(X, y):
    print("\n=== Logistische Regression (Trennung <= 5 Jahre) ===")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    model = LogisticRegression(max_iter=1000)
    auc_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring="roc_auc")
    acc_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring="accuracy")

    print(f"5-Fold CV ROC-AUC: {auc_scores.mean():.3f} (+/- {auc_scores.std():.3f})")
    print(f"5-Fold CV Accuracy: {acc_scores.mean():.3f} (+/- {acc_scores.std():.3f})")

    model.fit(X_scaled, y)
    coef_df = pd.Series(model.coef_[0], index=X.columns).sort_values(key=abs, ascending=False)
    print("\nStandardisierte Koeffizienten (Odds-Ratio-Richtung):")
    print(coef_df.round(3).to_string())

    return model, scaler, auc_scores, coef_df


def evaluate_tree_models(X, y):
    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    print("\n=== Random Forest ===")
    rf = RandomForestClassifier(n_estimators=300, max_depth=4,
                                 min_samples_leaf=5, random_state=RANDOM_STATE)
    rf_auc = cross_val_score(rf, X, y, cv=cv, scoring="roc_auc")
    print(f"5-Fold CV ROC-AUC: {rf_auc.mean():.3f} (+/- {rf_auc.std():.3f})")
    rf.fit(X, y)
    rf_importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("Feature Importances:")
    print(rf_importance.round(3).to_string())
    results["random_forest"] = (rf, rf_auc, rf_importance)

    print("\n=== Gradient Boosting ===")
    gb = GradientBoostingClassifier(n_estimators=200, max_depth=2,
                                     learning_rate=0.05, random_state=RANDOM_STATE)
    gb_auc = cross_val_score(gb, X, y, cv=cv, scoring="roc_auc")
    print(f"5-Fold CV ROC-AUC: {gb_auc.mean():.3f} (+/- {gb_auc.std():.3f})")
    gb.fit(X, y)
    gb_importance = pd.Series(gb.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("Feature Importances:")
    print(gb_importance.round(3).to_string())
    results["gradient_boosting"] = (gb, gb_auc, gb_importance)

    return results


def plot_model_comparison(logreg_auc, tree_results):
    labels = ["Log. Regression", "Random Forest", "Gradient Boosting"]
    means = [logreg_auc.mean(), tree_results["random_forest"][1].mean(),
             tree_results["gradient_boosting"][1].mean()]
    stds = [logreg_auc.std(), tree_results["random_forest"][1].std(),
            tree_results["gradient_boosting"][1].std()]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(labels, means, yerr=stds, capsize=5, color=["#4C72B0", "#55A868", "#C44E52"])
    ax.set_ylabel("ROC-AUC (5-Fold CV)")
    ax.set_ylim(0.4, 1.0)
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=1, label="Zufallsniveau")
    ax.set_title(f"Modellvergleich: Trennung innerhalb {HORIZON_YEARS} Jahren")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "model_comparison_auc.png")
    plt.close(fig)


def plot_feature_importance_comparison(coef_df, rf_importance, gb_importance):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)

    coef_df.abs().sort_values().plot(kind="barh", ax=axes[0], color="#4C72B0")
    axes[0].set_title("Log. Regression\n(|Koeffizient|)")

    rf_importance.sort_values().plot(kind="barh", ax=axes[1], color="#55A868")
    axes[1].set_title("Random Forest\n(Feature Importance)")

    gb_importance.sort_values().plot(kind="barh", ax=axes[2], color="#C44E52")
    axes[2].set_title("Gradient Boosting\n(Feature Importance)")

    fig.suptitle("Vergleich der wichtigsten Einflussfaktoren ueber Modelle hinweg")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "feature_importance_comparison.png")
    plt.close(fig)


def main():
    df = pd.read_csv(DATA_PATH)
    df = build_binary_target(df)
    X, y = prepare_features(df)

    print(f"\nZielvariable 'Trennung innerhalb {HORIZON_YEARS} Jahre': "
          f"{y.mean():.1%} positive Faelle (n={len(y)})")

    logreg_model, scaler, logreg_auc, coef_df = evaluate_logistic_regression(X, y)
    tree_results = evaluate_tree_models(X, y)

    plot_model_comparison(logreg_auc, tree_results)
    plot_feature_importance_comparison(
        coef_df, tree_results["random_forest"][2], tree_results["gradient_boosting"][2]
    )

    print(f"\nPlots gespeichert in: {OUT_DIR}")


if __name__ == "__main__":
    main()
