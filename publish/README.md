# Beziehungsstabilität – Web-App

Web-Version des Survival-Analysis-Projekts: Formular → Ensemble-Vorhersage
(Cox-PH, log. Regression, Random Forest, Gradient Boosting) → interaktive
Kurve im Vergleich zum Populationsdurchschnitt.

## Struktur

```
webapp/
├── app.py              # Flask-Routen (/, /api/predict, /ads.txt)
├── models.py            # Modell-Training + Vorhersage-Logik (Singleton)
├── data/
│   ├── divorce_dps_raw.csv
│   └── relationship_panel.csv
├── templates/index.html
├── static/css/style.css
├── static/js/main.js
├── requirements.txt
├── Procfile              # für Railway/Heroku (gunicorn)
└── DEPLOY.md             # Anleitung: live schalten + AdSense
```

## Lokal starten

```bash
pip install -r requirements.txt
python app.py
# -> http://127.0.0.1:5000
```

## Live schalten & Werbung einbinden

Siehe `DEPLOY.md` – Schritt-für-Schritt für Railway-Deployment und
Google-AdSense-Einrichtung (inkl. rechtlicher Hinweise zu Impressum/DSGVO).

## Wichtig

Modell-Limitationen und Disclaimer sind direkt auf der Seite sichtbar
(Abschnitt "Was diese Zahlen sind – und was nicht"). Bei Änderungen an den
Modellannahmen bitte auch diesen Text im Blick behalten, damit er weiterhin
zutrifft.
