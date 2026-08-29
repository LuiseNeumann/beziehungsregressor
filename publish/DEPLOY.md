# Deployment & Monetarisierung

## 1. Lokal testen

```bash
pip install -r requirements.txt
python app.py
```

Öffne `http://127.0.0.1:5000`. Modelle werden beim ersten Request trainiert
(dauert ein paar Sekunden), danach schnell.

## 2. Live schalten (Railway, empfohlen – du kennst es schon von ToDo-loo)

1. Repo auf GitHub pushen (`git init`, `git add .`, `git commit`, `git push`).
2. Auf [railway.app](https://railway.app) einloggen → "New Project" →
   "Deploy from GitHub repo" → dieses Repo auswählen.
3. Railway erkennt `Procfile` und `requirements.txt` automatisch und
   installiert/startet die App über `gunicorn app:app`.
4. Unter "Settings" → "Networking" → "Generate Domain" bekommst du eine
   öffentliche `*.up.railway.app`-URL – ab da ist die Seite **weltweit
   erreichbar**.
5. Optional: eigene Domain unter "Settings" → "Domains" → "Custom Domain"
   verbinden (DNS-CNAME beim Domain-Registrar setzen).

**Alternativen** (falls Railway-Kontingent/Preis nicht passt): Render.com
(sehr ähnlicher Ablauf, kostenloser Tier verfügbar), Fly.io, PythonAnywhere.
Alle funktionieren mit dem gleichen `Procfile`/`requirements.txt`-Setup,
ggf. mit leicht anderem Deploy-Befehl.

## 3. Werbung einbinden (Google AdSense)

Ich kann kein Werbekonto für dich anlegen – das musst du selbst tun, weil
Google eine echte Person und eine live erreichbare Domain prüft:

1. Seite zuerst live schalten (Schritt 2), mit ein paar echten Inhalten.
2. Auf [google.com/adsense](https://www.google.com/adsense) mit deiner
   Domain registrieren, den Bestätigungscode (Meta-Tag oder Script) in
   `templates/index.html` im `<head>`-Bereich einfügen, wo
   `<!-- WERBEFLAECHE: Kopf ... -->` steht.
3. Google prüft die Seite (kann einige Tage dauern). Wichtig: genug echter
   Inhalt (Datenschutzerklärung, Impressum – siehe Abschnitt 4) und Traffic
   werden meist vorausgesetzt.
4. Nach Freischaltung bekommst du eine Publisher-ID (`ca-pub-XXXXXXXX`).
   Ersetze in `app.py` die Route `/ads.txt` durch:
   ```
   google.com, pub-XXXXXXXXXXXXXXXX, DIRECT, f08c47fec0942fa0
   ```
5. Für jede der drei markierten `<div class="ad-slot">`-Stellen in
   `templates/index.html` das von AdSense generierte `<ins class="adsbygoogle">`-
   Snippet einfügen (ersetzt den Kommentar), plus das allgemeine
   `adsbygoogle.js`-Script einmal im `<head>`.

**Realistische Erwartung:** Bei einer Nischenseite ohne große Reichweite
sind AdSense-Einnahmen anfangs sehr gering (oft nur Cent-Beträge/Monat).
Traffic (SEO, Social Media, Communities) ist der eigentliche Hebel, nicht
die Anzahl der Werbeflächen.

## 4. Rechtliches (wichtig, bevor du wirklich live gehst)

Für eine öffentlich erreichbare, mit Werbung finanzierte Seite mit Bezug zu
Deutschland/EU brauchst du realistischerweise:

- **Impressum** (Pflicht nach § 5 DDG, auch für private/Studierenden-Projekte
  mit Werbeeinnahmen)
- **Datenschutzerklärung** (DSGVO) – insbesondere weil AdSense Cookies setzt
  und personenbezogene Daten verarbeitet (IP-Adressen etc.)
- **Cookie-Consent-Banner** vor dem Laden von AdSense-Skripten (technisch:
  Skript erst nach Einwilligung per JS nachladen, nicht direkt im `<head>`
  fest verdrahten)

Das ist über den Code hinausgehend – sag Bescheid, falls du dafür auch
Vorlagen (z. B. mit einem Cookie-Consent-Snippet und Impressum-Template)
möchtest.
