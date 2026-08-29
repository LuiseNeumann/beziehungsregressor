"""
app.py
=======
Flask-Server fuer die "Wie lange haelt meine Beziehung?"-Web-App.

Routen:
  GET  /            -> Frontend (Formular + Ergebnis-Anzeige)
  POST /api/predict  -> JSON-API, nimmt Beziehungsparameter entgegen,
                        gibt Ensemble-Vorhersage (Cox/LogReg/RF/GB) zurueck
  GET  /ads.txt       -> Platzhalter fuer AdSense-Verifizierung (siehe DEPLOY.md)

Die Modelle werden beim ersten Request einmalig trainiert und dann im
Arbeitsspeicher des Prozesses wiederverwendet (siehe models.py).
"""

from flask import Flask, request, jsonify, render_template, Response
from models import get_model_bundle, EDUCATION_ORDER

app = Flask(__name__)

EDUCATION_OPTIONS = list(EDUCATION_ORDER.keys())


@app.route("/")
def index():
    return render_template("index.html", education_options=EDUCATION_OPTIONS)


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}

    errors = _validate_input(data)
    if errors:
        return jsonify({"errors": errors}), 400

    conflict_raw = float(data["conflict"])       # 0-10
    bonding_raw = float(data["bonding"])          # 0-10
    satisfaction = float(data["satisfaction"])    # 0-10
    age_a = float(data["age_a"])
    age_b = float(data["age_b"])
    income = float(data["income"])
    education_a = data["education_a"]
    education_b = data["education_b"]
    married = int(bool(data["married"]))
    cohabiting = int(bool(data["cohabiting"]))
    has_children = int(bool(data["has_children"]))
    n_children = int(data.get("n_children") or 0)
    activities = int(data["activities"])
    already_together_years = float(data.get("already_together_years") or 0)

    education_avg_ord = (EDUCATION_ORDER[education_a] + EDUCATION_ORDER[education_b]) / 2

    user_values = {
        "conflict_index": conflict_raw / 10 * 4,
        "bonding_index": bonding_raw / 10 * 4,
        "satisfaction_index": satisfaction,
        "age_diff_abs": abs(age_a - age_b),
        "monthly_income_keur": income / 1000,
        "monthly_income_eur": income,
        "married": married,
        "cohabiting": cohabiting,
        "has_children": has_children,
        "n_children": n_children if has_children else 0,
        "shared_activities_per_month": activities,
        "education_avg_ord": education_avg_ord,
        "already_together_years": already_together_years,
    }

    bundle = get_model_bundle()
    result = bundle.predict(user_values)
    return jsonify(result)


def _validate_input(data: dict) -> list:
    errors = []
    required_numeric = {
        "conflict": (0, 10), "bonding": (0, 10), "satisfaction": (0, 10),
        "age_a": (16, 100), "age_b": (16, 100), "income": (0, 100000),
        "activities": (0, 60),
    }
    for field, (low, high) in required_numeric.items():
        if field not in data:
            errors.append(f"Feld '{field}' fehlt.")
            continue
        try:
            val = float(data[field])
            if not (low <= val <= high):
                errors.append(f"Feld '{field}' muss zwischen {low} und {high} liegen.")
        except (ValueError, TypeError):
            errors.append(f"Feld '{field}' muss eine Zahl sein.")

    for field in ("education_a", "education_b"):
        if data.get(field) not in EDUCATION_ORDER:
            errors.append(f"Feld '{field}' muss einer der bekannten Bildungsabschluesse sein.")

    if "already_together_years" in data and data["already_together_years"] not in (None, ""):
        try:
            val = float(data["already_together_years"])
            if not (0 <= val <= 80):
                errors.append("Feld 'already_together_years' muss zwischen 0 und 80 liegen.")
        except (ValueError, TypeError):
            errors.append("Feld 'already_together_years' muss eine Zahl sein.")

    return errors


@app.route("/ads.txt")
def ads_txt():
    # Platzhalter -- nach AdSense-Freischaltung durch den echten Inhalt
    # ersetzen, den Google im Publisher-Dashboard bereitstellt, z.B.:
    # google.com, pub-XXXXXXXXXXXXXXXX, DIRECT, f08c47fec0942fa0
    return Response("# ads.txt Platzhalter -- siehe DEPLOY.md\n", mimetype="text/plain")


if __name__ == "__main__":
    # Lokales Entwicklungs-Setup. In Produktion (Railway etc.) startet
    # gunicorn die App ueber den Procfile-Eintrag.
    app.run(host="0.0.0.0", port=5000, debug=True)
