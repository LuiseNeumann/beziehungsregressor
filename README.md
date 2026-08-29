# Vorhersage der Beziehungsstabilität mittels statistischer Modelle

Projekt-Implementierung nach der beiliegenden Projektbeschreibung: Survival
Analysis (Cox-PH, Kaplan-Meier) plus Vergleichsmodelle (logistische
Regression, Random Forest, Gradient Boosting) zur Schätzung von
Trennungsrisiken.

## Datengrundlage

Realdaten: **Divorce Predictors Data Set** (UCI ML Repository, Yöntem et
al. 2019), 170 Paare, 54 Gottman-DPS-Items, Klassifikation
verheiratet/geschieden. Bezogen von einem öffentlichen GitHub-Mirror, da
UCI/Kaggle im Ausführungsnetzwerk nicht direkt erreichbar waren
(`data/divorce_dps_raw.csv`).

**Wichtig:** Dieser Datensatz ist querschnittlich und enthält **keine
Zeit-bis-Trennung**. Da öffentlich frei zugängliche echte
Panel-/Survival-Daten zu Beziehungen mit vergleichbar reichhaltigen
Kovariaten praktisch nicht verfügbar sind, wird in
`01_data_preparation.py` transparent dokumentiert eine Zeit- und
Ereignisvariable **simuliert** – kalibriert über ein Weibull-Hazard-Modell,
dessen linearer Prädiktor auf den *echten* DPS-Konflikt-/Bindungsindizes
sowie zusätzlich simulierten Strukturvariablen (Alter, Bildung, Einkommen,
Ehestatus, Kinder) basiert. Alle Annahmen und Koeffizienten sind im
Docstring des Skripts offengelegt.

Für eine reale Anwendung müsste Schritt 1 durch echte Längsschnittdaten
ersetzt werden (z. B. lizenzierte Scheidungspanels wie Lillard & Panis
2003, oder eigene Panelerhebung).

## Struktur

```
relationship_stability/
├── data/
│   ├── divorce_dps_raw.csv        # echte Rohdaten (UCI DPS)
│   └── relationship_panel.csv     # aufbereitetes Panel (generiert)
├── 01_data_preparation.py         # Datenladung + Feature Engineering + Zeitsimulation
├── 02_eda.py                      # Explorative Datenanalyse, Kaplan-Meier
├── 03_cox_model.py                # Cox-Proportional-Hazards-Hauptmodell
├── 04_alternative_models.py       # Log. Regression, Random Forest, Gradient Boosting
├── 05_final_report.py             # führt 01-04 aus + erzeugt Fazit
├── 06_predict_own_relationship.py # interaktive Eingabe eigener Parameter + Vorhersage
├── requirements.txt
└── output/                        # alle Plots, CSVs, Fazit (generiert)
```

## Ausführung

```bash
pip install -r requirements.txt
python 05_final_report.py
```

Das führt die komplette Pipeline aus und legt alle Grafiken sowie
`output/fazit.txt` an. Einzelne Skripte (z. B. nur `03_cox_model.py`)
können auch separat ausgeführt werden, sofern `data/relationship_panel.csv`
bereits existiert (durch vorherigen Lauf von `01_data_preparation.py`).

## Eigene Beziehung einschätzen

```bash
python 06_predict_own_relationship.py
```

Fragt interaktiv nach Konflikthäufigkeit, Bindungsgefühl, **Zufriedenheit
(separat)**, Alter, Einkommen, **Bildungsabschluss beider Partner**,
Ehestatus, **Zusammenleben**, Kindern, gemeinsamen Aktivitäten und
**bereits gelebter Beziehungsdauer** (Enter = Default-Wert übernehmen).

Für jeden Zeithorizont (1/3/5/10 Jahre) werden **alle vier Modelle**
herangezogen: das live trainierte Cox-Modell sowie je ein eigens für
diesen Horizont trainiertes LogReg-, Random-Forest- und
Gradient-Boosting-Modell. Ausgegeben wird der **Median der vier
Wahrscheinlichkeiten** pro Horizont (plus Einzelmodell-Werte zur
Transparenz), außerdem eine **bedingte Vorhersage** ("ihr habt X Jahre
schon überstanden — wie stehen die Chancen für die nächsten Y Jahre?")
und ein Vergleichsplot mit dem Populationsdurchschnitt
(`output/own_relationship_prediction.png`).

**Bitte die Einordnung am Ende des Skripts ernst nehmen:** Das Modell
basiert auf n=170 mit größtenteils simulierten Struktur-/Zeitvariablen und
einer moderaten Modellgüte (C-Index/ROC-AUC ~0.65–0.70) — es ist eine
methodische Demonstration, keine verlässliche Prognose für eine reale,
konkrete Beziehung.

## Wichtigste Ergebnisse (mit Beispiel-Seed 42)

- Kaplan-Meier: ca. 74 % Wahrscheinlichkeit, nach 5 Jahren noch zusammen zu
  sein; ca. 54 % nach 10 Jahren.
- Cox-PH: stärkster Risikofaktor ist der Konfliktindex, stärkster
  protektiver Faktor ist der Ehestatus.
- Vergleichsmodelle (ROC-AUC ~0.63–0.66) bestätigen moderate,
  probabilistische statt deterministische Vorhersagekraft – wie in der
  Projektbeschreibung erwartet.

## Limitationen

Siehe Abschnitt 8 der Projektbeschreibung. Zusätzlich: Die
Ergebnisse sind durch die synthetische Zeitkomponente in ihrer externen
Validität eingeschränkt und dienen primär der methodischen
Demonstration des vollständigen Survival-Analysis-Workflows.
