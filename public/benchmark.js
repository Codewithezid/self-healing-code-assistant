function esc(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

async function requestJson(path, init = {}) {
  const response = await fetch(path, init);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.detail || "Request failed");
  }
  return payload;
}

function setStatus(text) {
  const el = document.getElementById("statusText");
  if (el) {
    el.textContent = text;
  }
}

function renderCards(profiles) {
  const mount = document.getElementById("profileCards");
  if (!mount) {
    return;
  }
  const names = ["fast", "balanced", "accurate"];
  mount.innerHTML = names.map((name) => {
    const row = profiles[name];
    if (!row) {
      return `<div class="bench-card"><div class="bench-title">${name}</div><div class="bench-muted">No report yet</div></div>`;
    }
    return `
      <div class="bench-card">
        <div class="bench-title">${esc(name)} (${row.rag_enabled ? "RAG" : "No RAG"})</div>
        <div class="bench-kpi">${esc(Number(row.semantic_accuracy_percent).toFixed(2))}%</div>
        <div class="bench-muted">Accuracy</div>
        <div class="bench-muted">Latency: ${esc(Number(row.average_latency_seconds).toFixed(3))}s</div>
        <div class="bench-muted">Passes: ${esc(row.semantic_passes)}/${esc(row.total_cases)}</div>
      </div>
    `;
  }).join("");
}

function renderReports(reports) {
  const body = document.getElementById("reportRows");
  if (!body) {
    return;
  }
  body.innerHTML = reports.map((row) => `
    <tr>
      <td>${esc(row.generated_at || "-")}</td>
      <td>${esc(row.runtime_profile)}</td>
      <td>${esc(Number(row.semantic_accuracy_percent).toFixed(2))}%</td>
      <td>${esc(Number(row.average_latency_seconds).toFixed(3))}s</td>
      <td>${esc(row.semantic_passes)}/${esc(row.total_cases)}</td>
      <td>${esc(row.model || "-")}</td>
    </tr>
  `).join("");
}

function renderAblation(variants) {
  const body = document.getElementById("ablationRows");
  if (!body) {
    return;
  }
  if (!Array.isArray(variants) || variants.length === 0) {
    body.innerHTML = `<tr><td colspan="4">No ablation run yet.</td></tr>`;
    return;
  }
  body.innerHTML = variants.map((row) => `
    <tr>
      <td>${esc(row.variant)}</td>
      <td>${esc(Number(row.semantic_accuracy_percent).toFixed(2))}%</td>
      <td>${esc(row.semantic_passes)}/${esc(row.total_cases)}</td>
      <td>${esc(Number(row.average_latency_seconds).toFixed(3))}s</td>
    </tr>
  `).join("");
}

async function refreshDashboard() {
  setStatus("Refreshing...");
  const [reportsData, compareData] = await Promise.all([
    requestJson("/api/benchmark/reports?limit=30"),
    requestJson("/api/benchmark/compare?profiles=fast,balanced,accurate"),
  ]);
  renderReports(Array.isArray(reportsData.reports) ? reportsData.reports : []);
  renderCards(compareData.profiles || {});
  setStatus("Ready");
}

async function runBenchmarks() {
  setStatus("Running benchmarks... this can take a while.");
  await requestJson("/api/benchmark/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      profiles: ["fast", "balanced", "accurate"],
      limit_cases: 0,
    }),
  });
  await refreshDashboard();
  setStatus("Benchmark run complete");
}

async function runAblation() {
  setStatus("Running RAG ablation...");
  const payload = await requestJson("/api/ablation/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      provider: "mistral",
      model: "mistral-medium-latest",
      limit_cases: 2,
      max_iterations: 3,
      validation_timeout: 5,
    }),
  });
  renderAblation(payload.variants || []);
  setStatus(`Ablation complete (${payload.report_file})`);
}

window.addEventListener("load", () => {
  const runBtn = document.getElementById("runBtn");
  const ablationBtn = document.getElementById("ablationBtn");
  const refreshBtn = document.getElementById("refreshBtn");
  if (runBtn) {
    runBtn.addEventListener("click", async () => {
      runBtn.disabled = true;
      try {
        await runBenchmarks();
      } catch (err) {
        setStatus(`Error: ${err.message || "run failed"}`);
      } finally {
        runBtn.disabled = false;
      }
    });
  }
  if (refreshBtn) {
    refreshBtn.addEventListener("click", async () => {
      try {
        await refreshDashboard();
      } catch (err) {
        setStatus(`Error: ${err.message || "refresh failed"}`);
      }
    });
  }
  if (ablationBtn) {
    ablationBtn.addEventListener("click", async () => {
      ablationBtn.disabled = true;
      try {
        await runAblation();
      } catch (err) {
        setStatus(`Error: ${err.message || "ablation failed"}`);
      } finally {
        ablationBtn.disabled = false;
      }
    });
  }
  refreshDashboard().catch((err) => {
    setStatus(`Error: ${err.message || "load failed"}`);
  });
});
