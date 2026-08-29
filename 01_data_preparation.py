"""
01_data_preparation.py
=======================
Datengrundlage fuer das Projekt "Vorhersage der Beziehungsstabilitaet".

WICHTIGER HINWEIS ZUR DATENGRUNDLAGE
-------------------------------------
Es wird der oeffentliche "Divorce Predictors Data Set" (UCI ML Repository,
Yoentem et al. 2019, Gottman Divorce Predictors Scale, DPS) verwendet:
    170 Paare (Tuerkei), 54 Items (Skala 0-4), Zielvariable Class
    (0 = verheiratet/gluecklich, 1 = geschieden).
Quelle: https://archive.ics.uci.edu/dataset/497/divorce+predictors+data+set

Dieser Datensatz ist QUERSCHNITTLICH (kein Beobachtungszeitraum, keine
Trennungsdauer). Fuer eine echte Survival-Analyse (Cox-PH, Kaplan-Meier)
wird zwingend eine Zeitvariable benoetigt, die in keinem oeffentlich frei
zugaenglichen Datensatz zusammen mit vergleichbar reichhaltigen
Beziehungsmerkmalen vorliegt (echte Panel-Scheidungsdaten, z. B. Lillard &
Panis 2003, liegen nur hinter Zugangsbeschraenkungen / Lizenzen).

Loesung dieses Projekts (transparent dokumentiert):
1. Die 54 DPS-Items werden zu einem empirisch gestuetzten "Konfliktindex"
   und einem "Bindungsindex" verdichtet (Faktoren aus echten Item-Antworten).
2. Demografische / strukturelle Variablen (Alter, Bildung, Einkommen,
   Ehestatus, Kinder, Zusammenleben) werden -- da im DPS-Datensatz nicht
   erhoben -- realistisch simuliert und mit den echten Konflikt-/
   Bindungsindizes korreliert.
3. Eine Ereigniszeit (time, event) wird ueber ein Weibull-Hazard-Modell
   erzeugt, dessen linearer Praediktor von den ECHTEN DPS-Indizes sowie den
   simulierten Strukturvariablen abhaengt. Rechtszensierung wird durch ein
   festes Beobachtungsende simuliert.

Damit ist der Kern der Konfliktverhalten-Information real (aus dem UCI DPS
Datensatz), waehrend die Zeit-bis-Trennung-Struktur (fuer Cox/KM) simuliert,
aber inhaltlich plausibel kalibriert ist. Alle Zufallsprozesse sind geseedet
und im Code klar gekennzeichnet.
"""

import numpy as np
import pandas as pd
from pathlib import Path

RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)

DATA_DIR = Path(__file__).parent / "data"
RAW_PATH = DATA_DIR / "divorce.csv"
OUT_PATH = DATA_DIR / "relationship_panel.csv"


def load_real_dps_data() -> pd.DataFrame:
    """Laedt den echten UCI 'Divorce Predictors Data Set' (54 Items)."""
    df = pd.read_csv(RAW_PATH, sep=";")
    item_cols = [c for c in df.columns if c.startswith("Atr")]
    assert len(item_cols) == 54, f"Erwartet 54 Items, gefunden {len(item_cols)}"
    df = df.rename(columns={"Class": "divorced_reference"})
    return df, item_cols


def build_conflict_and_bonding_indices(df: pd.DataFrame, item_cols: list) -> pd.DataFrame:
    """
    Verdichtet die 54 echten DPS-Items zu zwei interpretierbaren Indizes.

    Nach Gottman/Yoentem et al. lassen sich die Items grob in zwei Bloecke
    teilen:
      - Items 1-20:  gemeinsame Werte, Ziele, Harmonie  -> "Bonding"
      - Items 21-54: Wissen ueber Partner, aber v.a. Konflikt-/
                     Kommunikationsverhalten (Kritik, Verachtung,
                     Rueckzug)  -> "Conflict"
    Diese Zuordnung ist eine vereinfachte, aber literaturkonforme
    Aggregation (statt einer vollen Faktorenanalyse), um die Item-Ebene auf
    zwei sinnvolle, stetige Praediktoren zu reduzieren.
    """
    bonding_items = item_cols[0:20]
    conflict_items = item_cols[20:54]

    out = pd.DataFrame(index=df.index)
    out["bonding_index"] = df[bonding_items].mean(axis=1)      # 0 (niedrig) - 4 (hoch)
    out["conflict_index"] = df[conflict_items].mean(axis=1)    # 0 (niedrig) - 4 (hoch)
    out["divorced_reference"] = df["divorced_reference"].values
    return out


def simulate_structural_variables(n: int) -> pd.DataFrame:
    """
    Simuliert demografische/strukturelle Variablen, die im DPS-Datensatz
    nicht erhoben wurden. Verteilungen orientieren sich an realistischen
    Kennwerten fuer Paare in laengeren Beziehungen (grobe Anlehnung an
    destatis-Kennzahlen zu Erstheiratsalter / Bildungsniveau in DE).
    """
    age_partner_a = rng.normal(34, 8, n).clip(18, 70)
    age_diff = rng.normal(0, 3.5, n).clip(-15, 15)
    age_partner_b = (age_partner_a + age_diff).clip(18, 75)

    education_levels = ["Hauptschule", "Realschule", "Abitur", "Hochschule"]
    education_p = [0.15, 0.30, 0.25, 0.30]
    education_a = rng.choice(education_levels, size=n, p=education_p)
    # Bildung der Partner ist typischerweise positiv assortativ (Homogamie):
    # mit 65% Wahrscheinlichkeit gleiches Niveau wie Partner A, sonst neu gezogen
    same_level = rng.uniform(0, 1, n) < 0.65
    education_b_random = rng.choice(education_levels, size=n, p=education_p)
    education_b = np.where(same_level, education_a, education_b_random)

    income_base = {"Hauptschule": 1900, "Realschule": 2400,
                   "Abitur": 2900, "Hochschule": 3800}
    income = np.array([
        rng.normal((income_base[a] + income_base[b]) / 2, 600)
        for a, b in zip(education_a, education_b)
    ]).clip(800, None)

    married = rng.choice([0, 1], size=n, p=[0.35, 0.65])
    cohabiting = np.where(married == 1, 1, rng.choice([0, 1], size=n, p=[0.4, 0.6]))
    has_children = rng.choice([0, 1], size=n, p=[0.45, 0.55])
    n_children = np.where(
        has_children == 1, rng.poisson(1.6, n).clip(1, 5), 0
    )
    shared_activities_per_month = rng.poisson(6, n).clip(0, None)

    return pd.DataFrame({
        "age_partner_a": age_partner_a.round(1),
        "age_partner_b": age_partner_b.round(1),
        "age_diff_abs": np.abs(age_diff).round(1),
        "education_a": education_a,
        "education_b": education_b,
        "monthly_income_eur": income.round(0),
        "married": married,
        "cohabiting": cohabiting,
        "has_children": has_children,
        "n_children": n_children,
        "shared_activities_per_month": shared_activities_per_month,
    })


EDUCATION_ORDER = {"Hauptschule": 1, "Realschule": 2, "Abitur": 3, "Hochschule": 4}


def add_education_ordinal_and_satisfaction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fuegt eine ordinale Bildungskennzahl (Mittelwert beider Partner) sowie
    einen separaten Zufriedenheits-Index hinzu.

    Der Zufriedenheits-Index ist konzeptionell von conflict_index/
    bonding_index unterschieden (eigene Selbsteinschaetzungsfrage in
    06_predict_own_relationship.py), aber in der Simulation plausibel mit
    beiden korreliert: hohe Bindung und niedriger Konflikt erhoehen im
    Mittel die Zufriedenheit, zusaetzlich individuelles Rauschen (Menschen
    bewerten "Zufriedenheit" nicht rein mechanisch aus Konflikt/Bindung).
    Skala 0-10, analog zur Nutzereingabe im Vorhersage-Skript.
    """
    out = df.copy()
    out["education_a_ord"] = out["education_a"].map(EDUCATION_ORDER)
    out["education_b_ord"] = out["education_b"].map(EDUCATION_ORDER)
    out["education_avg_ord"] = (out["education_a_ord"] + out["education_b_ord"]) / 2

    z_bonding = (out["bonding_index"] - out["bonding_index"].mean()) / out["bonding_index"].std()
    z_conflict = (out["conflict_index"] - out["conflict_index"].mean()) / out["conflict_index"].std()
    noise = rng.normal(0, 1.0, len(out))
    satisfaction_raw = 5.5 + 1.4 * z_bonding - 1.2 * z_conflict + 0.9 * noise
    out["satisfaction_index"] = satisfaction_raw.clip(0, 10).round(2)
    return out


def simulate_survival_times(df: pd.DataFrame, max_followup_years: float = 15.0) -> pd.DataFrame:
    """
    Erzeugt Zeit-bis-Trennung (time) und Zensierungsindikator (event) ueber
    ein Weibull-Hazard-Modell:

        h(t | x) = h0(t) * exp(beta' x)

    Der lineare Praediktor beta'x kombiniert die ECHTEN DPS-Indizes
    (conflict_index erhoeht, bonding_index senkt das Trennungsrisiko) mit
    den simulierten Strukturvariablen. Alle Koeffizienten sind so gewaehlt,
    dass Vorzeichen und relative Groessenordnung mit publizierten
    Scheidungsforschungsbefunden konsistent sind:
      - hoher Konflikt -> stark erhoehtes Risiko
      - hohe Bindung / gemeinsame Aktivitaeten -> reduziertes Risiko
      - Ehe & Kinder & Zusammenleben -> stabilisierend (survival-literaturkonform)
      - grosser Altersunterschied -> leicht erhoehtes Risiko
      - niedriges Einkommen -> leicht erhoehtes Risiko (Stressfaktor)
      - hohe Zufriedenheit -> reduziertes Risiko (zusaetzlich zu Bindung/Konflikt)
      - hoeheres Bildungsniveau -> leicht stabilisierend (Scheidungsforschung,
        Effekt ist klein und teils kontrovers, daher niedrig gewichtet)
    """
    n = len(df)

    # z-standardisierte Praediktoren fuer den linearen Prediktor
    z_conflict = (df["conflict_index"] - df["conflict_index"].mean()) / df["conflict_index"].std()
    z_bonding = (df["bonding_index"] - df["bonding_index"].mean()) / df["bonding_index"].std()
    z_income = (df["monthly_income_eur"] - df["monthly_income_eur"].mean()) / df["monthly_income_eur"].std()
    z_activities = (df["shared_activities_per_month"] - df["shared_activities_per_month"].mean()) / df["shared_activities_per_month"].std()
    z_agediff = (df["age_diff_abs"] - df["age_diff_abs"].mean()) / df["age_diff_abs"].std()
    z_satisfaction = (df["satisfaction_index"] - df["satisfaction_index"].mean()) / df["satisfaction_index"].std()
    z_education = (df["education_avg_ord"] - df["education_avg_ord"].mean()) / df["education_avg_ord"].std()

    linear_predictor = (
        0.85 * z_conflict
        - 0.55 * z_bonding
        - 0.30 * z_activities
        - 0.45 * df["married"]
        - 0.25 * df["cohabiting"]
        - 0.35 * df["has_children"]
        + 0.20 * z_agediff
        - 0.15 * z_income
        - 0.30 * z_satisfaction
        - 0.10 * z_education
    )

    # Weibull-Baseline-Hazard: h0(t) = (shape/scale) * (t/scale)^(shape-1)
    shape = 1.4          # shape > 1: steigendes Grundrisiko mit der Zeit
    scale_base = 9.0      # Jahre, Basis-Skalenparameter

    # Individuelle Skalenparameter ueber den linearen Prediktor (AFT-Parametrisierung)
    individual_scale = scale_base * np.exp(-linear_predictor / shape)

    u = rng.uniform(0, 1, n)
    true_event_time = individual_scale * (-np.log(u)) ** (1 / shape)

    censor_time = max_followup_years
    time_observed = np.minimum(true_event_time, censor_time)
    event_observed = (true_event_time <= censor_time).astype(int)

    out = df.copy()
    out["time_years"] = time_observed.round(2)
    out["event"] = event_observed
    return out


def main():
    real_df, item_cols = load_real_dps_data()
    indices_df = build_conflict_and_bonding_indices(real_df, item_cols)

    structural_df = simulate_structural_variables(len(indices_df))

    panel = pd.concat([indices_df.reset_index(drop=True),
                        structural_df.reset_index(drop=True)], axis=1)
    panel = add_education_ordinal_and_satisfaction(panel)
    panel = simulate_survival_times(panel)

    panel.insert(0, "couple_id", [f"C{idx:04d}" for idx in range(1, len(panel) + 1)])

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    panel.to_csv(OUT_PATH, index=False)

    print(f"Panel gespeichert: {OUT_PATH}  ({panel.shape[0]} Paare, {panel.shape[1]} Spalten)")
    print(panel.head(8).to_string())
    print("\nEreignisrate (Trennung beobachtet):", panel["event"].mean().round(3))
    print("Mediane Beobachtungsdauer (Jahre):", panel["time_years"].median())


if __name__ == "__main__":
    main()
