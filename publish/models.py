"""
models.py
==========
Modell-Backend fuer die Flask-Web-App. Enthaelt die gleiche Modelllogik wie
die Notebook-Skripte (01-04), aber ohne Plotting/Print-Ausgaben, fuer den
Einsatz in einer laufenden Server-Anwendung.

Alle Modelle werden EINMAL beim Start des Servers trainiert (siehe
load_or_train_models) und danach fuer jede Anfrage wiederverwendet
(zustandslos, kein Re-Training pro Request -> schnelle Antwortzeiten).
"""

from pathlib import Path
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "relationship_panel.csv"

HORIZONS = [1, 3, 5, 10]
EDUCATION_ORDER = {"Hauptschule": 1, "Realschule": 2, "Abitur": 3, "Hochschule": 4}

COX_FEATURE_COLS = [
    "conflict_index", "bonding_index", "satisfaction_index", "age_diff_abs",
    "monthly_income_keur", "married", "cohabiting", "has_children",
    "n_children", "shared_activities_per_month", "education_avg_ord",
]
CLF_FEATURES = [
    "conflict_index", "bonding_index", "satisfaction_index", "age_diff_abs",
    "monthly_income_eur", "married", "cohabiting", "has_children",
    "n_children", "shared_activities_per_month", "education_avg_ord",
]

RANDOM_STATE = 42


def _prepare_cox_frame(df: pd.DataFrame) -> pd.DataFrame:
    model_df = df.copy()
    model_df["monthly_income_keur"] = model_df["monthly_income_eur"] / 1000
    return model_df[COX_FEATURE_COLS + ["time_years", "event"]]


def _build_binary_target(df: pd.DataFrame, horizon: float) -> pd.DataFrame:
    known = df[(df["time_years"] >= horizon) | (df["event"] == 1)].copy()
    known["separated_within_horizon"] = (
        (known["time_years"] <= horizon) & (known["event"] == 1)
    ).astype(int)
    return known


def _train_classifiers_for_horizon(df: pd.DataFrame, horizon: float) -> dict:
    horizon_df = _build_binary_target(df, horizon)
    X = horizon_df[CLF_FEATURES]
    y = horizon_df["separated_within_horizon"].values

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


class ModelBundle:
    """Haelt alle trainierten Modelle + Referenzdaten im Speicher des laufenden
    Server-Prozesses. Wird einmal beim App-Start instanziiert."""

    def __init__(self):
        df = pd.read_csv(DATA_PATH)
        self.raw_df = df

        self.cox_model_df = _prepare_cox_frame(df)
        self.cph = CoxPHFitter()
        self.cph.fit(self.cox_model_df, duration_col="time_years", event_col="event")

        self.classifiers_by_horizon = {
            h: _train_classifiers_for_horizon(df, horizon=h) for h in HORIZONS
        }

        kmf = KaplanMeierFitter()
        kmf.fit(df["time_years"], event_observed=df["event"])
        self.population_km = kmf.survival_function_

        self.population_avg_cox_surv = self.cph.predict_survival_function(
            self.cox_model_df[COX_FEATURE_COLS]
        ).mean(axis=1)

        self.cox_summary = self.cph.summary.copy()

    # ---------------------------------------------------------------

    def _cox_survival_curve(self, user_values: dict):
        row = {col: user_values[col] for col in COX_FEATURE_COLS}
        frame = pd.DataFrame([row])
        return self.cph.predict_survival_function(frame).iloc[:, 0]

    def _cox_survival_at(self, user_values: dict, years: float, surv_curve=None):
        if surv_curve is None:
            surv_curve = self._cox_survival_curve(user_values)
        return float(np.interp(years, surv_curve.index, surv_curve.values))

    def _classifier_survival_at(self, clf_bundle: dict, user_values: dict):
        row = pd.DataFrame([{col: user_values[col] for col in CLF_FEATURES}])
        row_scaled = clf_bundle["scaler"].transform(row)
        p_sep_logreg = clf_bundle["logreg"].predict_proba(row_scaled)[0, 1]
        p_sep_rf = clf_bundle["rf"].predict_proba(row)[0, 1]
        p_sep_gb = clf_bundle["gb"].predict_proba(row)[0, 1]
        return {
            "logreg": 1 - p_sep_logreg,
            "rf": 1 - p_sep_rf,
            "gb": 1 - p_sep_gb,
        }

    def predict(self, user_values: dict) -> dict:
        """Haupt-Vorhersagefunktion fuer die API. Erwartet ein dict mit allen
        Feldern aus COX_FEATURE_COLS (monthly_income_keur/eur) sowie
        'already_together_years'."""
        surv_curve = self._cox_survival_curve(user_values)

        per_horizon = []
        for h in HORIZONS:
            p_cox = self._cox_survival_at(user_values, h, surv_curve)
            clf_bundle = self.classifiers_by_horizon[h]
            p_clf = self._classifier_survival_at(clf_bundle, user_values)
            all_probs = {"cox": p_cox, **p_clf}
            median_p = float(np.median(list(all_probs.values())))
            per_horizon.append({
                "horizon": h,
                "cox": round(all_probs["cox"], 4),
                "logreg": round(all_probs["logreg"], 4),
                "rf": round(all_probs["rf"], 4),
                "gb": round(all_probs["gb"], 4),
                "median": round(median_p, 4),
            })

        already = float(user_values.get("already_together_years", 0) or 0)
        conditional = []
        if already > 0:
            p_already = self._cox_survival_at(user_values, already, surv_curve)
            for h in HORIZONS:
                p_future = self._cox_survival_at(user_values, already + h, surv_curve)
                p_cond = min(p_future / p_already, 1.0) if p_already > 0 else None
                conditional.append({"horizon": h, "probability": round(p_cond, 4) if p_cond is not None else None})
        conditional_result = {
            "already_together_years": already,
            "p_survived_so_far": round(p_already, 4) if already > 0 else None,
            "future_horizons": conditional,
        } if already > 0 else None

        # Kurven fuer den Chart: individuelle Kurve + Populationsdurchschnitt,
        # auf ein gemeinsames Zeitraster (0-15 Jahre, 0.25-Jahres-Schritte) interpoliert
        time_grid = np.arange(0, 15.25, 0.25)
        individual_curve = np.interp(time_grid, surv_curve.index, surv_curve.values)
        population_curve = np.interp(
            time_grid, self.population_avg_cox_surv.index, self.population_avg_cox_surv.values
        )

        return {
            "per_horizon": per_horizon,
            "conditional": conditional_result,
            "chart": {
                "time_years": time_grid.round(2).tolist(),
                "individual_survival": individual_curve.round(4).tolist(),
                "population_survival": population_curve.round(4).tolist(),
            },
            "model_meta": {
                "n_training_couples": int(len(self.raw_df)),
                "cox_concordance": round(float(self.cph.concordance_index_), 3),
            },
        }


_bundle_instance = None


def get_model_bundle() -> ModelBundle:
    """Singleton-Zugriff: Modelle werden nur beim ersten Aufruf trainiert."""
    global _bundle_instance
    if _bundle_instance is None:
        _bundle_instance = ModelBundle()
    return _bundle_instance
