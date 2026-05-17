function esc(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

async function requestJson(path) {
  const response = await fetch(path);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.detail || "Request failed");
  }
  return payload;
}

function renderSummary(summary) {
  const mount = document.getElementById("summaryCards");
  if (!mount) {
    return;
  }
  const verdict = summary.verdict_counts || {};
  mount.innerHTML = `
    <div class="ana-card"><div>Total feedback</div><div class="ana-kpi">${esc(summary.total_feedback || 0)}</div></div>
    <div class="ana-card"><div>Avg rating</div><div class="ana-kpi">${esc(Number(summary.average_rating || 0).toFixed(2))}</div></div>
    <div class="ana-card"><div>Avg confidence</div><div class="ana-kpi">${esc((Number(summary.average_confidence || 0) * 100).toFixed(1))}%</div></div>
    <div class="ana-card"><div>Avg hallucination risk</div><div class="ana-kpi">${esc((Number(summary.average_hallucination_risk || 0) * 100).toFixed(1))}%</div></div>
    <div class="ana-card"><div>Correct</div><div class="ana-kpi">${esc(verdict.correct || 0)}</div></div>
    <div class="ana-card"><div>Partially correct</div><div class="ana-kpi">${esc(verdict.partially_correct || 0)}</div></div>
    <div class="ana-card"><div>Wrong</div><div class="ana-kpi">${esc(verdict.wrong || 0)}</div></div>
  `;
}

function renderRows(items) {
  const body = document.getElementById("feedbackRows");
  if (!body) {
    return;
  }
  if (!items.length) {
    body.innerHTML = `<tr><td colspan="6">No feedback yet.</td></tr>`;
    return;
  }
  body.innerHTML = items.map((row) => `
    <tr>
      <td>${esc(row.created_at || "-")}</td>
      <td>${esc(row.verdict || "-")}</td>
      <td>${esc(row.rating || "-")}</td>
      <td>${esc(row.runtime_profile || "-")}</td>
      <td>${row.rag_enabled ? "on" : "off"}</td>
      <td>${esc(row.comment || "")}</td>
    </tr>
  `).join("");
}

async function loadAnalytics() {
  const [summaryPayload, recentPayload] = await Promise.all([
    requestJson("/api/analytics/feedback/summary?window_days=30"),
    requestJson("/api/analytics/feedback/recent?limit=100"),
  ]);
  renderSummary(summaryPayload.summary || {});
  renderRows(Array.isArray(recentPayload.items) ? recentPayload.items : []);
}

window.addEventListener("load", () => {
  loadAnalytics().catch((err) => {
    const body = document.getElementById("feedbackRows");
    if (body) {
      body.innerHTML = `<tr><td colspan="6">Failed to load analytics: ${esc(err.message || "error")}</td></tr>`;
    }
  });
});
