// ============================================================
// main.js -- Formularinteraktion, API-Aufruf, Chart-Rendering
// ============================================================

const rangeOutputs = ["conflict", "bonding", "satisfaction"];
rangeOutputs.forEach((id) => {
  const input = document.getElementById(id);
  const output = document.getElementById(id + "-out");
  input.addEventListener("input", () => { output.textContent = input.value; });
});

const hasChildrenCheckbox = document.getElementById("has_children");
const nChildrenField = document.getElementById("n_children_field");
hasChildrenCheckbox.addEventListener("change", () => {
  nChildrenField.style.display = hasChildrenCheckbox.checked ? "block" : "none";
});

// ---------------------------------------------------------------- Hero-Chart
// Zeigt zunaechst nur den Populationsdurchschnitt; nach der ersten
// Berechnung wird die individuelle Kurve ergaenzt (im gleichen Chart-Objekt).
let heroChart = null;

function renderChart(chartData, hasIndividual) {
  const ctx = document.getElementById("heroChart").getContext("2d");

  const datasets = [
    {
      label: "Populationsdurchschnitt (n=170)",
      data: chartData.population_survival,
      borderColor: "#9BA3B4",
      borderDash: [6, 4],
      borderWidth: 2,
      pointRadius: 0,
      tension: 0.15,
    },
  ];

  if (hasIndividual) {
    datasets.push({
      label: "Deine Beziehung (Ensemble-Schätzung)",
      data: chartData.individual_survival,
      borderColor: "#B5533C",
      backgroundColor: "rgba(181,83,60,0.12)",
      fill: true,
      borderWidth: 3,
      pointRadius: 0,
      tension: 0.15,
    });
  }

  if (heroChart) { heroChart.destroy(); }

  heroChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: chartData.time_years,
      datasets,
    },
    options: {
      responsive: true,
      interaction: { mode: "index", intersect: false },
      scales: {
        x: {
          title: { display: true, text: "Beziehungsdauer (Jahre)", color: "#9BA3B4" },
          ticks: { color: "#9BA3B4", maxTicksLimit: 8 },
          grid: { color: "rgba(255,255,255,0.06)" },
        },
        y: {
          min: 0, max: 1,
          title: { display: true, text: "Wahrscheinlichkeit \"noch zusammen\"", color: "#9BA3B4" },
          ticks: { color: "#9BA3B4", callback: (v) => (v * 100).toFixed(0) + "%" },
          grid: { color: "rgba(255,255,255,0.06)" },
        },
      },
      plugins: {
        legend: { labels: { color: "#EDE9DD", font: { family: "Inter" } } },
        tooltip: {
          callbacks: {
            label: (item) => `${item.dataset.label}: ${(item.raw * 100).toFixed(1)}%`,
          },
        },
      },
    },
  });
}

// Initiale (leere) Population-only-Ansicht laden, damit die Seite beim
// ersten Laden nicht ohne Chart dasteht.
async function loadInitialChart() {
  try {
    const resp = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(defaultPayload()),
    });
    if (!resp.ok) return;
    const data = await resp.json();
    renderChart(data.chart, false);
  } catch (e) {
    // still, kein Chart beim ersten Laden ist kein kritischer Fehler
  }
}

function defaultPayload() {
  return {
    conflict: 3, bonding: 7, satisfaction: 7,
    age_a: 30, age_b: 30, income: 3000,
    education_a: "Abitur", education_b: "Abitur",
    married: false, cohabiting: true, has_children: false, n_children: 0,
    activities: 6, already_together_years: 0,
  };
}

// ---------------------------------------------------------------- Formular
const form = document.getElementById("predictForm");
const resultContent = document.getElementById("resultContent");
const resultPlaceholder = document.querySelector(".result-placeholder");

form.addEventListener("submit", async (e) => {
  e.preventDefault();

  const payload = {
    conflict: Number(document.getElementById("conflict").value),
    bonding: Number(document.getElementById("bonding").value),
    satisfaction: Number(document.getElementById("satisfaction").value),
    age_a: Number(document.getElementById("age_a").value),
    age_b: Number(document.getElementById("age_b").value),
    income: Number(document.getElementById("income").value),
    education_a: document.getElementById("education_a").value,
    education_b: document.getElementById("education_b").value,
    married: document.getElementById("married").checked,
    cohabiting: document.getElementById("cohabiting").checked,
    has_children: document.getElementById("has_children").checked,
    n_children: Number(document.getElementById("n_children").value || 0),
    activities: Number(document.getElementById("activities").value),
    already_together_years: Number(document.getElementById("already_together_years").value || 0),
  };

  const submitBtn = form.querySelector(".btn-primary");
  submitBtn.disabled = true;
  submitBtn.textContent = "Berechne …";

  try {
    const resp = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await resp.json();

    if (!resp.ok) {
      alert("Bitte Eingaben prüfen:\n" + (data.errors || []).join("\n"));
      return;
    }

    renderResult(data);
    renderChart(data.chart, true);
  } catch (err) {
    alert("Es gab ein technisches Problem bei der Berechnung. Bitte später erneut versuchen.");
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Kurve berechnen";
  }
});

function renderResult(data) {
  resultPlaceholder.style.display = "none";
  resultContent.style.display = "block";

  const statGrid = document.getElementById("statGrid");
  statGrid.innerHTML = "";
  data.per_horizon.forEach((row) => {
    const pct = (row.median * 100).toFixed(0);
    const riskClass = row.median < 0.5 ? "risk" : "";
    statGrid.insertAdjacentHTML("beforeend", `
      <div class="stat-card ${riskClass}">
        <span class="stat-value">${pct}%</span>
        <span class="stat-label">noch zusammen<br>nach ${row.horizon} ${row.horizon === 1 ? "Jahr" : "Jahren"}</span>
      </div>
    `);
  });

  const conditionalBlock = document.getElementById("conditionalBlock");
  if (data.conditional) {
    const c = data.conditional;
    const rows = c.future_horizons.map(
      (f) => `<li>noch ${f.horizon} weitere ${f.horizon === 1 ? "Jahr" : "Jahre"}: <strong>${(f.probability * 100).toFixed(0)}%</strong></li>`
    ).join("");
    conditionalBlock.innerHTML = `
      <h4>Bedingte Vorhersage (ihr seid schon ${c.already_together_years} Jahre zusammen)</h4>
      <p>Diese Phase habt ihr bereits "überstanden" (geschätzt ${(c.p_survived_so_far * 100).toFixed(0)}% Wahrscheinlichkeit dafür). Ab jetzt:</p>
      <ul>${rows}</ul>
    `;
    conditionalBlock.style.display = "block";
  } else {
    conditionalBlock.style.display = "none";
  }

  const table = document.getElementById("transparencyTable");
  const header = `<tr><th>Horizont</th><th>Cox</th><th>LogReg</th><th>RF</th><th>GB</th><th>Median</th></tr>`;
  const rows = data.per_horizon.map((r) => `
    <tr>
      <td>${r.horizon} J.</td>
      <td>${(r.cox * 100).toFixed(0)}%</td>
      <td>${(r.logreg * 100).toFixed(0)}%</td>
      <td>${(r.rf * 100).toFixed(0)}%</td>
      <td>${(r.gb * 100).toFixed(0)}%</td>
      <td><strong>${(r.median * 100).toFixed(0)}%</strong></td>
    </tr>
  `).join("");
  table.innerHTML = header + rows;

  document.getElementById("modelMeta").textContent =
    `Trainiert auf ${data.model_meta.n_training_couples} Paaren · Cox C-Index ${data.model_meta.cox_concordance}`;
}

loadInitialChart();
