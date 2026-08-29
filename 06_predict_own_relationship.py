"""
06_predict_own_relationship.py
================================
Interaktives Skript: Eigene Beziehungsparameter eingeben und daraus per
ENSEMBLE aus allen 4 Modellen (Cox-PH, logistische Regression, Random
Forest, Gradient Boosting) eine statistische Einordnung ableiten.

Fuer jeden Zeithorizont (1/3/5/10 Jahre) wird:
  - das Cox-Modell einmal auf die volle Beziehungsdauer gefittet und daraus
    die Survival-Wahrscheinlichkeit am jeweiligen Horizont abgelesen,
  - fuer LogReg/RF/GB JEWEILS EIN EIGENES Modell auf die binaere Zielgroesse
    "Trennung bis Horizont X" trainiert (siehe 04_alternative_models.py,
    train_classifiers_for_horizon),
  - aus den 4 resultierenden "noch zusammen"-Wahrscheinlichkeiten der
    MEDIAN gebildet -> das ist die ausgewiesene Ensemble-Schaetzung.

Zusaetzlich: bedingte Vorhersage auf Basis der bereits gelebten
Beziehungsdauer (Survival-Konditionierung: P(T > t+s | T > s) =
S(t+s) / S(s)), da eine bereits ueberstandene Zeit ohne Trennung das
zukuenftige Risiko senkt (je laenger man schon zusammen ist, desto
stabiler tendenziell -- "je laenger es gut ging, desto wahrscheinlicher
geht es weiter gut").

WICHTIGER HINWEIS (bitte unbedingt lesen, auch im Terminal-Output):
--------------------------------------------------------------------
Dieses Modell basiert auf einer kleinen Stichprobe (n=170) mit groesstenteils
SIMULIERTEN Struktur- und Zeitvariablen (siehe 01_data_preparation.py) und
moderater Modellguete (C-Index/ROC-AUC ~0.65-0.70). Es ist ein methodisches
Demonstrationsobjekt, KEIN klinisch oder wissenschaftlich validiertes
Vorhersageinstrument fuer reale individuelle Beziehungen. Die Ausgabe ist
eine statistische Einordnung im Sinne von Abschnitt 7 der
Projektbeschreibung -- keine Prognose, auf die man sich verlassen sollte.
"""

import sys
import importlib.util
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))


def _load_module(path, name):
    """Laedt Module, deren Dateiname mit einer Ziffer beginnt (kein
    gueltiger Python-Modulname fuer normalen 'import')."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cox_mod = _load_module(BASE_DIR / "03_cox_model.py", "cox_mod")
alt_mod = _load_module(BASE_DIR / "04_alternative_models.py", "alt_mod")

DATA_PATH = BASE_DIR / "data" / "relationship_panel.csv"
OUT_DIR = BASE_DIR / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["figure.dpi"] = 110
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

HORIZONS = [1, 3, 5, 10]
EDUCATION_OPTIONS = ["Hauptschule", "Realschule", "Abitur", "Hochschule"]


# ---------------------------------------------------------------- Eingabe

def ask_float(prompt: str, low: float, high: float, default: float) -> float:
    while True:
        raw = input(f"{prompt} [{low}-{high}, Enter fuer Default={default}]: ").strip()
        if raw == "":
            return default
        try:
            val = float(raw)
            if low <= val <= high:
                return val
            print(f"  Bitte einen Wert zwischen {low} und {high} eingeben.")
        except ValueError:
            print("  Bitte eine Zahl eingeben.")


def ask_int(prompt: str, low: int, high: int, default: int) -> int:
    while True:
        raw = input(f"{prompt} [{low}-{high}, Enter fuer Default={default}]: ").strip()
        if raw == "":
            return default
        try:
            val = int(raw)
            if low <= val <= high:
                return val
            print(f"  Bitte einen Wert zwischen {low} und {high} eingeben.")
        except ValueError:
            print("  Bitte eine ganze Zahl eingeben.")


def ask_yes_no(prompt: str, default: bool) -> int:
    default_str = "j" if default else "n"
    while True:
        raw = input(f"{prompt} (j/n) [Enter fuer {default_str}]: ").strip().lower()
        if raw == "":
            return int(default)
        if raw in ("j", "ja", "y", "yes"):
            return 1
        if raw in ("n", "nein", "no"):
            return 0
        print("  Bitte mit j oder n antworten.")


def ask_choice(prompt: str, options: list, default: str) -> str:
    options_str = ", ".join(f"{i+1}={opt}" for i, opt in enumerate(options))
    default_idx = options.index(default) + 1
    while True:
        raw = input(f"{prompt} ({options_str}) [Enter fuer {default_idx}={default}]: ").strip()
        if raw == "":
            return default
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1]
        print(f"  Bitte eine Zahl von 1 bis {len(options)} eingeben.")


def collect_user_inputs() -> dict:
    print("=" * 70)
    print("EIGENE BEZIEHUNG: PARAMETER-EINGABE")
    print("=" * 70)
    print("Bitte schaetze die folgenden Werte moeglichst ehrlich ein.")
    print("Fuer Konflikt/Bindung/Zufriedenheit: 0 = 'nie/gar nicht', 10 = 'sehr haeufig/sehr stark'.\n")

    conflict_raw = ask_float(
        "Wie haeufig kommt es zu Kritik, Streit oder Rueckzug in eurer Beziehung?",
        0, 10, 3
    )
    bonding_raw = ask_float(
        "Wie stark fuehlt ihr euch verbunden (gemeinsame Werte, Ziele, Harmonie)?",
        0, 10, 7
    )
    satisfaction_raw = ask_float(
        "Wie zufrieden bist du insgesamt mit eurer Beziehung? (unabhaengig von Konflikt/Bindung)",
        0, 10, 7
    )
    age_a = ask_float("Dein Alter", 16, 90, 30)
    age_b = ask_float("Alter deiner/deines Partnerin/Partners", 16, 90, 30)
    income = ask_float("Gemeinsames monatliches Nettoeinkommen (EUR)", 500, 20000, 3000)

    education_a = ask_choice("Dein hoechster Bildungsabschluss", EDUCATION_OPTIONS, "Abitur")
    education_b = ask_choice("Bildungsabschluss deiner/deines Partnerin/Partners", EDUCATION_OPTIONS, "Abitur")

    married = ask_yes_no("Seid ihr verheiratet?", False)
    cohabiting = ask_yes_no("Lebt ihr zusammen?", True)
    has_children = ask_yes_no("Habt ihr gemeinsame Kinder?", False)
    n_children = 0
    if has_children:
        n_children = ask_int("Wie viele gemeinsame Kinder?", 1, 10, 1)
    activities = ask_int(
        "Wie oft im Monat unternehmt ihr bewusst gemeinsame Aktivitaeten?",
        0, 30, 6
    )
    already_together_years = ask_float(
        "Wie viele Jahre seid ihr schon zusammen? (fuer bedingte Vorhersage)",
        0, 60, 2
    )

    education_order = {"Hauptschule": 1, "Realschule": 2, "Abitur": 3, "Hochschule": 4}
    education_avg_ord = (education_order[education_a] + education_order[education_b]) / 2

    # Skalierung 0-10 (Nutzereingabe) -> 0-4 (DPS-Item-Skala im Modell)
    conflict_index = conflict_raw / 10 * 4
    bonding_index = bonding_raw / 10 * 4

    return {
        "conflict_index": conflict_index,
        "bonding_index": bonding_index,
        "satisfaction_index": satisfaction_raw,
        "age_diff_abs": abs(age_a - age_b),
        "monthly_income_eur": income,
        "monthly_income_keur": income / 1000,
        "married": married,
        "cohabiting": cohabiting,
        "has_children": has_children,
        "n_children": n_children,
        "shared_activities_per_month": activities,
        "education_avg_ord": education_avg_ord,
        "already_together_years": already_together_years,
    }


# ------------------------------------------------------------- Modelle

def fit_all_models(df: pd.DataFrame):
    """Fittet das Cox-Modell einmal (volle Zeitachse) sowie fuer jeden
    Horizont eigene LogReg/RF/GB-Klassifikatoren."""
    cox_model_df = cox_mod.prepare_model_frame(df)
    cph = cox_mod.fit_cox_model(cox_model_df)

    classifiers_by_horizon = {}
    for h in HORIZONS:
        classifiers_by_horizon[h] = alt_mod.train_classifiers_for_horizon(df, horizon=h)

    return cph, cox_model_df, classifiers_by_horizon


def cox_survival_at(cph, cox_feature_cols, user_values, years):
    row = {col: user_values[col] for col in cox_feature_cols}
    frame = pd.DataFrame([row])
    surv = cph.predict_survival_function(frame).iloc[:, 0]
    return float(np.interp(years, surv.index, surv.values)), surv


def classifier_survival_at(clf_bundle, user_values, feature_cols):
    """Gibt P(noch zusammen bis Horizont) = 1 - P(separated) je Modell zurueck."""
    row = pd.DataFrame([{col: user_values[col] for col in feature_cols}])

    row_scaled = clf_bundle["scaler"].transform(row)
    p_sep_logreg = clf_bundle["logreg"].predict_proba(row_scaled)[0, 1]

    p_sep_rf = clf_bundle["rf"].predict_proba(row)[0, 1]
    p_sep_gb = clf_bundle["gb"].predict_proba(row)[0, 1]

    return {
        "logreg": 1 - p_sep_logreg,
        "rf": 1 - p_sep_rf,
        "gb": 1 - p_sep_gb,
    }


def population_average_cox_survival(cph, cox_model_df, cox_feature_cols):
    X = cox_model_df[cox_feature_cols]
    surv_funcs = cph.predict_survival_function(X)
    return surv_funcs.mean(axis=1)


# --------------------------------------------------------------- Main

def main():
    df = pd.read_csv(DATA_PATH)
    cph, cox_model_df, classifiers_by_horizon = fit_all_models(df)
    cox_feature_cols = [c for c in cox_model_df.columns if c not in ("time_years", "event")]
    clf_feature_cols = alt_mod.FEATURES

    user_values = collect_user_inputs()

    print("\n" + "=" * 70)
    print("ENSEMBLE-VORHERSAGE (Median aus Cox, log. Regression, RF, GB)")
    print("=" * 70)

    ensemble_rows = []
    individual_surv_for_plot = None

    for h in HORIZONS:
        p_cox, surv_curve = cox_survival_at(cph, cox_feature_cols, user_values, h)
        if h == max(HORIZONS):
            individual_surv_for_plot = surv_curve

        clf_bundle = classifiers_by_horizon[h]
        p_clf = classifier_survival_at(clf_bundle, user_values, clf_feature_cols)

        all_probs = {"cox": p_cox, **p_clf}
        median_p = float(np.median(list(all_probs.values())))

        ensemble_rows.append({
            "horizon": h,
            "cox": all_probs["cox"],
            "logreg": all_probs["logreg"],
            "rf": all_probs["rf"],
            "gb": all_probs["gb"],
            "median": median_p,
            "n_train": clf_bundle["n_train"],
        })

    ensemble_df = pd.DataFrame(ensemble_rows).set_index("horizon")

    print("\nGeschaetzte Wahrscheinlichkeit, zum jeweiligen Zeitpunkt noch zusammen zu sein:")
    for h, row in ensemble_df.iterrows():
        print(f"  Nach {h:>2} Jahren -> Ensemble-Median: {row['median']:.1%}")
    print("\nZur Transparenz -- Einzelmodelle je Horizont:")
    print(ensemble_df[["cox", "logreg", "rf", "gb", "median"]].round(3).to_string())

    # ------------------------------ Bedingte Vorhersage (bereits gelebte Zeit)
    already = user_values["already_together_years"]
    if already > 0:
        p_already, _ = cox_survival_at(cph, cox_feature_cols, user_values, already)
        print("\n" + "=" * 70)
        print(f"BEDINGTE VORHERSAGE (ihr seid schon {already:.1f} Jahre zusammen)")
        print("=" * 70)
        print(
            f"Ueberlebt bis heute (Cox-Modell): {p_already:.1%} -- ihr habt diese\n"
            f"Phase also schon 'ueberstanden'. Die bedingte Wahrscheinlichkeit,\n"
            f"WEITERE X Jahre zusammen zu bleiben (gegeben, dass ihr schon\n"
            f"{already:.1f} Jahre zusammen seid), ist hoeher als die unbedingte\n"
            f"Wahrscheinlichkeit ab Beziehungsbeginn:\n"
        )
        for h in HORIZONS:
            p_future_unconditional, _ = cox_survival_at(cph, cox_feature_cols, user_values, already + h)
            p_conditional = p_future_unconditional / p_already if p_already > 0 else float("nan")
            p_conditional = min(p_conditional, 1.0)
            print(f"  Noch {h:>2} weitere Jahre zusammen (ab jetzt): {p_conditional:.1%}")

    # ------------------------------------------------------- Populationsvergleich
    population_avg_surv = population_average_cox_survival(cph, cox_model_df, cox_feature_cols)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    individual_surv_for_plot.plot(ax=ax, label="Deine Beziehung (Cox-Schaetzung)",
                                   linewidth=2.5, color="#C44E52")
    population_avg_surv.plot(ax=ax, label="Populationsdurchschnitt (n=170)", linewidth=2,
                              linestyle="--", color="#4C72B0")
    ax.scatter(ensemble_df.index, ensemble_df["median"], color="black", zorder=5,
               label="Ensemble-Median (4 Modelle)", marker="D")
    ax.set_xlabel("Beziehungsdauer (Jahre)")
    ax.set_ylabel("Geschaetzte Wahrscheinlichkeit, noch zusammen zu sein")
    ax.set_title("Individuelle Ensemble-Vorhersage vs. Populationsdurchschnitt")
    ax.set_ylim(0, 1.02)
    ax.legend()
    fig.tight_layout()
    out_path = OUT_DIR / "own_relationship_prediction.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"\nPlot gespeichert unter: {out_path}")

    print("\n" + "=" * 70)
    print("EINORDNUNG")
    print("=" * 70)
    print(
        "Diese Zahlen sind eine statistische Einordnung auf Basis eines kleinen,\n"
        "methodisch limitierten Modell-Ensembles (n=170, groesstenteils simulierte\n"
        "Struktur-/Zeitvariablen, ROC-AUC/C-Index ~0.65-0.70) -- keine verlaessliche\n"
        "Prognose fuer eure konkrete Beziehung. Sie zeigen vor allem, in welche\n"
        "Richtung die von dir eingegebenen Faktoren im Modell wirken, nicht was\n"
        "tatsaechlich passieren wird."
    )


if __name__ == "__main__":
    main()
