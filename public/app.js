const DEFAULT_APP_CONFIG = {
  defaultProvider: "mistral",
  allowedProviders: ["openai", "openrouter", "mistral"],
  authRequired: false,
  maxIterationsCap: 3,
  validationTimeoutCap: 5,
  ragAvailable: true,
  ragDefaultEnabled: false,
  correctiveRagModes: ["fast", "balanced", "aggressive"],
  correctiveRagDefaultMode: "balanced",
  runtimeProfiles: ["custom", "fast", "balanced", "accurate", "goated"],
  defaultRuntimeProfile: "custom",
  userKeysEnabled: false,
  userKeysPersistent: false,
  userKeysMaxEntries: 50
};

const APP_STATE = {
  running: false,
  arenaRunning: false,
  arenaMode: false,
  applyingProfile: false,
  qCount: 0,
  tkN: 0,
  ragMode: false,
  attachments: [],
  keysByProvider: {},
  modelsByProvider: {},
  config: { ...DEFAULT_APP_CONFIG, ...(window.APP_CONFIG || {}) }
};

const MODEL_OPTIONS = {
  openai: ["gpt-5.1", "gpt-5", "gpt-4.1"],
  openrouter: ["openai/gpt-4.1", "anthropic/claude-3.5-sonnet", "google/gemini-2.5-pro"],
  mistral: ["codestral-latest", "mistral-medium-latest", "mistral-small-latest"]
};

const DEFAULT_MODEL_BY_PROVIDER = {
  openai: "gpt-5.1",
  openrouter: "openai/gpt-4.1",
  mistral: "mistral-medium-latest"
};

function defaultModelForProvider(provider) {
  return DEFAULT_MODEL_BY_PROVIDER[provider] || "mistral-medium-latest";
}

function setModelOptionsForProvider(provider, preferredModel = "") {
  const modelSel = byId("modelSel");
  if (!modelSel) {
    return;
  }
  if (provider === "local") {
    modelSel.innerHTML = "";
    return;
  }
  const options = APP_STATE.modelsByProvider[provider]
    || MODEL_OPTIONS[provider]
    || MODEL_OPTIONS.mistral;
  modelSel.innerHTML = options
    .map((model) => `<option value="${esc(model)}">${esc(model)}</option>`)
    .join("");
  const nextModel = options.includes(preferredModel)
    ? preferredModel
    : defaultModelForProvider(provider);
  modelSel.value = nextModel;
}

function byId(id) {
  return document.getElementById(id);
}

function valueOf(id, fallback = "") {
  const el = byId(id);
  return el && "value" in el ? el.value : fallback;
}

function updateAttachmentModePill() {
  const mode = valueOf("attachMode", "rag_only");
  const pill = byId("pa");
  if (!pill) {
    return;
  }
  if (!APP_STATE.attachments.length) {
    pill.textContent = "no files";
    pill.classList.remove("active");
    return;
  }
  pill.textContent = mode === "both" ? `files:${APP_STATE.attachments.length} both` : `files:${APP_STATE.attachments.length} rag`;
  pill.classList.add("active");
}

function renderAttachmentChips() {
  const strip = byId("attachmentStrip");
  const list = byId("attachmentList");
  if (!strip || !list) {
    return;
  }
  if (!APP_STATE.attachments.length) {
    strip.style.display = "none";
    list.innerHTML = "";
    updateAttachmentModePill();
    return;
  }
  strip.style.display = "";
  list.innerHTML = APP_STATE.attachments.map((item, index) => (
    `<span class="attach-chip" title="${esc(item.filename)}">${esc(item.filename)}<button type="button" onclick="removeAttachment(${index})">×</button></span>`
  )).join("");
  updateAttachmentModePill();
}

function removeAttachment(index) {
  APP_STATE.attachments.splice(index, 1);
  renderAttachmentChips();
}

async function uploadAttachments(files) {
  if (!files || files.length === 0) {
    return;
  }
  const formData = new FormData();
  Array.from(files).forEach((file) => formData.append("files", file));
  const payload = await requestJson("/api/attachments", {
    method: "POST",
    body: formData
  });
  const rows = Array.isArray(payload.attachments) ? payload.attachments : [];
  APP_STATE.attachments = rows.map((row) => ({
    id: row.attachment_id,
    filename: row.filename,
    kind: row.kind,
    charCount: row.char_count,
    indexed: Boolean(row.indexed_to_qdrant)
  }));
  renderAttachmentChips();
  const indexedCount = rows.filter((row) => row.indexed_to_qdrant).length;
  addLog(`Attached ${rows.length} file(s). Indexed ${indexedCount}/${rows.length} into Qdrant.`);
}

function openAttachmentPicker() {
  const input = byId("attachmentInput");
  if (!input) {
    return;
  }
  input.value = "";
  input.click();
}

function checkedOf(id, fallback = false) {
  const el = byId(id);
  return el && "checked" in el ? Boolean(el.checked) : fallback;
}

function apiUrl(path) {
  return path;
}

function autoH(el) {
  el.style.height = "auto";
  el.style.height = Math.min(el.scrollHeight, 180) + "px";
}

function handleKey(e) {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    send();
  }
}

function activateNav(el) {
  document.querySelectorAll(".nav-link").forEach((n) => n.classList.remove("active"));
  el.classList.add("active");
}

function esc(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function handleProvider(preserveProfile = false) {
  if (!preserveProfile) {
    markRuntimeProfileCustom();
  }
  const provider = valueOf("providerSel", APP_STATE.config.defaultProvider || "mistral");
  const currentModel = valueOf("modelSel", "");
  setModelOptionsForProvider(provider, currentModel);
  const modelField = byId("modelField");
  const localField = byId("localField");
  const providerPill = byId("pp");
  if (modelField) {
    modelField.style.display = provider === "local" ? "none" : "";
  }
  if (localField) {
    localField.style.display = provider === "local" ? "" : "none";
  }
  if (providerPill) {
    providerPill.textContent = provider;
  }
  setKeyControlsVisibility();
  void syncProviderRuntime(provider, currentModel);
  updatePills();
}

function markRuntimeProfileCustom() {
  if (APP_STATE.applyingProfile) {
    return;
  }
  const runtimeProfile = byId("runtimeProfile");
  if (runtimeProfile && runtimeProfile.value !== "custom") {
    runtimeProfile.value = "custom";
  }
}

function updatePills() {
  const provider = valueOf("providerSel", APP_STATE.config.defaultProvider || "mistral");
  const model = provider !== "local"
    ? valueOf("modelSel", defaultModelForProvider(provider))
    : valueOf("localPath", "Qwen/Qwen2.5-Coder-0.5B-Instruct").split("/").pop();
  const modelPill = byId("pm");
  const retryPill = byId("pi");
  const profilePill = byId("pf");
  if (modelPill) {
    modelPill.textContent = model || "local-model";
  }
  if (profilePill) {
    profilePill.textContent = valueOf("runtimeProfile", APP_STATE.config.defaultRuntimeProfile);
  }
  if (retryPill) {
    retryPill.textContent = valueOf("maxIter", "3") + " retries";
  }
  const ragPill = byId("pr");
  if (ragPill) {
    ragPill.classList.toggle("active", APP_STATE.ragMode);
  }
  const correctivePill = byId("pc");
  if (correctivePill) {
    correctivePill.textContent = valueOf("correctiveRagMode", APP_STATE.config.correctiveRagDefaultMode);
    correctivePill.classList.toggle("active", APP_STATE.ragMode);
  }
}

function selectedProvider() {
  return valueOf("providerSel", APP_STATE.config.defaultProvider || "mistral");
}

function selectedKeyId() {
  return valueOf("savedKeySel", "");
}

function setKeyControlsVisibility() {
  const provider = selectedProvider();
  const field = byId("savedKeyField");
  const enabled = APP_STATE.config.userKeysEnabled && provider !== "local";
  if (field) {
    field.style.display = enabled ? "" : "none";
  }
}

function populateSavedKeys(provider, preferredKeyId = "") {
  const sel = byId("savedKeySel");
  if (!sel) {
    return;
  }
  const keys = APP_STATE.keysByProvider[provider] || [];
  const options = keys.length === 0
    ? ['<option value="" disabled selected>No saved keys yet</option>']
    : keys.map((item) => {
      const title = `${item.label} (${item.masked_key})`;
      return `<option value="${esc(item.key_id)}">${esc(title)}</option>`;
    });
  sel.innerHTML = options.join("");
  if (preferredKeyId && keys.some((item) => item.key_id === preferredKeyId)) {
    sel.value = preferredKeyId;
    return;
  }
  if (keys.length > 0) {
    sel.value = keys[0].key_id;
  }
}

async function loadSavedKeys(provider, preferredKeyId = "") {
  if (!APP_STATE.config.userKeysEnabled || provider === "local") {
    populateSavedKeys(provider, "");
    return;
  }
  try {
    const rows = await requestJson(`/api/keys?provider=${encodeURIComponent(provider)}`);
    APP_STATE.keysByProvider[provider] = Array.isArray(rows) ? rows : [];
    populateSavedKeys(provider, preferredKeyId);
  } catch (err) {
    APP_STATE.keysByProvider[provider] = [];
    populateSavedKeys(provider, "");
    addLog(`Saved key load failed: ${err.message || "request failed"}`);
  }
}

async function refreshProviderModels(provider, preferredModel = "") {
  if (provider === "local") {
    setModelOptionsForProvider(provider, preferredModel);
    return;
  }
  const activeKeyId = selectedKeyId();
  let path = `/api/providers/${encodeURIComponent(provider)}/models`;
  if (activeKeyId) {
    path += `?key_id=${encodeURIComponent(activeKeyId)}`;
  }
  try {
    const payload = await requestJson(path);
    if (Array.isArray(payload.models) && payload.models.length > 0) {
      const curated = MODEL_OPTIONS[provider] || [];
      APP_STATE.modelsByProvider[provider] = payload.source === "saved_key"
        ? payload.models
        : (() => {
            const filtered = payload.models.filter((model) => curated.includes(model));
            return filtered.length > 0 ? filtered : curated;
          })();
    }
  } catch (err) {
    addLog(`Model sync failed: ${err.message || "request failed"}`);
  }
  setModelOptionsForProvider(provider, preferredModel);
  updatePills();
}

async function syncProviderRuntime(provider, preferredModel = "") {
  await loadSavedKeys(provider);
  await refreshProviderModels(provider, preferredModel);
}

async function handleSavedKeyChange() {
  markRuntimeProfileCustom();
  await refreshProviderModels(selectedProvider(), valueOf("modelSel", ""));
}

async function saveApiKey() {
  const provider = selectedProvider();
  if (provider === "local") {
    addLog("Local provider does not use API keys.");
    return;
  }
  const apiKey = window.prompt(`Paste ${provider} API key:`) || "";
  const trimmedKey = apiKey.trim();
  if (!trimmedKey) {
    addLog("Key add canceled.");
    return;
  }
  try {
    const payload = await requestJson("/api/keys", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        provider,
        api_key: trimmedKey,
        label: `${provider}-key`
      })
    });
    await loadSavedKeys(provider, payload.key && payload.key.key_id ? payload.key.key_id : "");
    if (Array.isArray(payload.models) && payload.models.length > 0) {
      APP_STATE.modelsByProvider[provider] = payload.models;
    }
    setModelOptionsForProvider(provider, valueOf("modelSel", defaultModelForProvider(provider)));
    updatePills();
    addLog("API key saved and validated.");
  } catch (err) {
    addLog(`Key save failed: ${err.message || "request failed"}`);
    addErrMsg(err.message || "Failed to save key.");
  }
}

async function removeSelectedApiKey() {
  const provider = selectedProvider();
  const keyId = selectedKeyId();
  if (!keyId) {
    addLog("Choose a saved key to remove.");
    return;
  }
  try {
    await requestJson(`/api/keys/${encodeURIComponent(keyId)}`, { method: "DELETE" });
    await loadSavedKeys(provider, "");
    await refreshProviderModels(provider, valueOf("modelSel", defaultModelForProvider(provider)));
    addLog("Saved key removed.");
  } catch (err) {
    addLog(`Key delete failed: ${err.message || "request failed"}`);
    addErrMsg(err.message || "Failed to delete key.");
  }
}

function syncRag(preserveProfile = false) {
  if (!preserveProfile) {
    markRuntimeProfileCustom();
  }
  APP_STATE.ragMode = checkedOf("ragToggle", APP_STATE.config.ragDefaultEnabled);
  const ragPill = byId("pr");
  if (ragPill) {
    ragPill.classList.toggle("active", APP_STATE.ragMode);
  }
  const correctiveMode = byId("correctiveRagMode");
  if (correctiveMode) {
    correctiveMode.disabled = !APP_STATE.ragMode || !APP_STATE.config.ragAvailable;
  }
  updatePills();
}

function syncCorrectiveRagMode(preserveProfile = false) {
  if (!preserveProfile) {
    markRuntimeProfileCustom();
  }
  updatePills();
}

function applyRuntimeProfile() {
  const profile = valueOf("runtimeProfile", APP_STATE.config.defaultRuntimeProfile);
  let profileModel = "";
  APP_STATE.applyingProfile = true;
  try {
    if (profile === "fast") {
      byId("providerSel").value = "mistral";
      profileModel = "codestral-latest";
      byId("maxIter").value = "1";
      byId("iN").textContent = "1";
      byId("timeoutR").value = "3";
      byId("tN").textContent = "3";
      byId("ragToggle").checked = false;
      byId("correctiveRagMode").value = "fast";
    } else if (profile === "balanced") {
      byId("providerSel").value = "mistral";
      profileModel = "mistral-medium-latest";
      byId("maxIter").value = "2";
      byId("iN").textContent = "2";
      byId("timeoutR").value = "5";
      byId("tN").textContent = "5";
      byId("ragToggle").checked = true;
      byId("correctiveRagMode").value = "balanced";
    } else if (profile === "accurate") {
      byId("providerSel").value = "mistral";
      profileModel = "mistral-medium-latest";
      byId("maxIter").value = "3";
      byId("iN").textContent = "3";
      byId("timeoutR").value = "5";
      byId("tN").textContent = "5";
      byId("ragToggle").checked = true;
      byId("correctiveRagMode").value = "aggressive";
    } else if (profile === "goated") {
      byId("providerSel").value = "mistral";
      profileModel = "mistral-large-latest";
      byId("maxIter").value = "6";
      byId("iN").textContent = "6";
      byId("timeoutR").value = "12";
      byId("tN").textContent = "12";
      byId("ragToggle").checked = true;
      byId("correctiveRagMode").value = "aggressive";
    }
  } finally {
    APP_STATE.applyingProfile = false;
  }
  handleProvider(true);
  if (profileModel) {
    byId("modelSel").value = profileModel;
  }
  syncRag(true);
  syncCorrectiveRagMode(true);
}

function clearAll() {
  APP_STATE.qCount = 0;
  APP_STATE.arenaRunning = false;
  document.getElementById("msgs").innerHTML = "";
  document.getElementById("msgs").style.display = "none";
  const arenaRag = byId("arenaMsgsRag");
  const arenaNormal = byId("arenaMsgsNormal");
  if (arenaRag) {
    arenaRag.innerHTML = "";
  }
  if (arenaNormal) {
    arenaNormal.innerHTML = "";
  }
  if (APP_STATE.arenaMode) {
    const arenaWrap = byId("arenaWrap");
    if (arenaWrap) {
      arenaWrap.style.display = "";
    }
    document.getElementById("welcome").style.display = "none";
  } else {
    const arenaWrap = byId("arenaWrap");
    if (arenaWrap) {
      arenaWrap.style.display = "none";
    }
    document.getElementById("welcome").style.display = "";
  }
  document.getElementById("sQ").textContent = "0";
  document.getElementById("sI").textContent = "-";
  document.getElementById("sR").textContent = "-";
  setStat("idle");
  resetPipe();
  document.getElementById("actLog").innerHTML =
    '<div class="log-entry"><span class="log-time">--</span><span class="log-txt" style="color:var(--ink4)">No activity yet.</span></div>';
  APP_STATE.attachments = [];
  renderAttachmentChips();
}

function toggleArenaMode() {
  APP_STATE.arenaMode = !APP_STATE.arenaMode;
  const arenaBtn = byId("arenaToggleBtn");
  const arenaWrap = byId("arenaWrap");
  const chatArea = byId("chatArea");
  const welcome = byId("welcome");
  const msgs = byId("msgs");
  if (arenaBtn) {
    arenaBtn.classList.toggle("on", APP_STATE.arenaMode);
  }
  if (chatArea) {
    chatArea.classList.toggle("arena-on", APP_STATE.arenaMode);
  }
  if (APP_STATE.arenaMode) {
    if (arenaWrap) {
      arenaWrap.style.display = "";
    }
    if (welcome) {
      welcome.style.display = "none";
    }
    if (msgs) {
      msgs.style.display = "none";
    }
    addLog("Coding Arena enabled: RAG vs Normal side-by-side.");
  } else {
    if (arenaWrap) {
      arenaWrap.style.display = "none";
    }
    const hasMessages = msgs && msgs.children.length > 0;
    if (welcome) {
      welcome.style.display = hasMessages ? "none" : "";
    }
    if (msgs) {
      msgs.style.display = hasMessages ? "" : "none";
    }
    addLog("Coding Arena disabled.");
  }
}

function fillExample(btn) {
  const ta = document.getElementById("prompt");
  ta.value = btn.dataset.prompt || "";
  autoH(ta);
  ta.focus();
}

function scrollD() {
  const area = document.getElementById("chatArea");
  area.scrollTo({ top: area.scrollHeight, behavior: "smooth" });
}

function setStat(status) {
  const el = document.getElementById("sStat");
  const color = {
    idle: "var(--ink3)",
    running: "var(--amber)",
    done: "var(--green)",
    error: "var(--red)"
  };
  el.textContent = status;
  el.style.color = color[status] || color.idle;
}

function addLog(msg) {
  const now = new Date();
  const stamp = [
    now.getHours().toString().padStart(2, "0"),
    now.getMinutes().toString().padStart(2, "0"),
    now.getSeconds().toString().padStart(2, "0")
  ].join(":");
  const el = document.getElementById("actLog");
  if (el.textContent.includes("No activity yet.")) {
    el.innerHTML = "";
  }
  const row = document.createElement("div");
  row.className = "log-entry";
  row.innerHTML = `<span class="log-time">${stamp}</span><span class="log-txt">${esc(msg)}</span>`;
  el.prepend(row);
  if (el.children.length > 12) {
    el.removeChild(el.lastChild);
  }
}

function resetPipe() {
  document.querySelectorAll("#pipe .ps").forEach((step) => {
    step.className = "ps wait";
  });
}

function setPipe(activeStage) {
  const order = ["retrieve_context", "generate_code", "execute_code", "check_result", "retry_or_end"];
  document.querySelectorAll("#pipe .ps").forEach((step) => {
    const stage = step.dataset.stage;
    step.className = "ps wait";
    if (!activeStage) {
      return;
    }
    if (order.indexOf(stage) < order.indexOf(activeStage)) {
      step.className = "ps done";
      return;
    }
    if (stage === activeStage) {
      step.className = "ps running";
    }
  });
}

function donePipe() {
  document.querySelectorAll("#pipe .ps").forEach((step) => {
    step.className = "ps done";
  });
}

function addThink() {
  const id = "tk" + (++APP_STATE.tkN);
  const msgs = document.getElementById("msgs");
  const d = document.createElement("div");
  d.className = "msg-row";
  d.id = id;
  d.innerHTML = '<div class="msg-ai-header"><div class="ai-avatar">lg</div><span class="ai-name">Code Assistant</span></div><div class="thinking-row"><div class="dots"><span></span><span></span><span></span></div>Generating...</div>';
  msgs.appendChild(d);
  scrollD();
  return id;
}

function rmThink(id) {
  const el = document.getElementById(id);
  if (el) {
    el.remove();
  }
}

function addUserMsg(text) {
  const d = document.createElement("div");
  d.className = "msg-row";
  d.innerHTML = `<div class="msg-user-row"><div class="msg-user-bubble">${esc(text)}</div></div>`;
  document.getElementById("msgs").appendChild(d);
  scrollD();
}

function mkCode(lang, code) {
  return `<div class="code-wrap"><div class="code-bar"><span class="code-lang-tag">${esc(lang)}</span><button class="code-copy" type="button">copy</button></div><pre class="code-pre">${esc(code)}</pre></div>`;
}

function mkExec(kind, message) {
  const labels = { ok: "success", err: "error", run: "running" };
  return `<div class="exec-box ${kind}"><div class="exec-bar"><span class="exec-dot"></span>${labels[kind]}</div><div class="exec-body">${esc(message)}</div></div>`;
}

function attachCopyButtons(scope) {
  scope.querySelectorAll(".code-copy").forEach((btn) => {
    btn.addEventListener("click", async function () {
      const code = this.closest(".code-wrap").querySelector("pre").textContent;
      await navigator.clipboard.writeText(code);
      this.textContent = "copied";
      this.classList.add("copied");
      window.setTimeout(() => {
        this.textContent = "copy";
        this.classList.remove("copied");
      }, 1800);
    });
  });
}

function addErrMsg(msg) {
  const d = document.createElement("div");
  d.className = "msg-row";
  d.innerHTML = `<div class="msg-ai-header"><div class="ai-avatar" style="background:var(--red)">!</div><span class="ai-name">Error</span></div><div class="ai-body">${mkExec("err", msg)}</div>`;
  document.getElementById("msgs").appendChild(d);
  scrollD();
}

function renderAssistantBody(data) {
  if (data.json_mode) {
    return mkCode("json", JSON.stringify(data, null, 2));
  }

  const chunks = [];
  if (data.solution.prefix) {
    chunks.push(`<p>${esc(data.solution.prefix)}</p>`);
  } else {
    chunks.push("<p>The assistant returned a runnable Python solution.</p>");
  }

  chunks.push(mkExec("run", `Validation timeout: ${data.validation_timeout}s`));
  chunks.push(
    mkExec(
      (data.semantic_validation_passed ?? data.validation_passed) ? "ok" : "err",
      (data.semantic_validation_passed ?? data.validation_passed)
        ? `Validated successfully after ${data.iterations} iteration(s).`
        : `Reached ${data.iterations} iteration(s). ${data.validation_message}`
    )
  );

  if (data.validation_message) {
    chunks.push(`<p>${esc(data.validation_message)}</p>`);
  }
  if (data.runtime_profile) {
    chunks.push(`<p><span class="inline-code">profile</span> ${esc(data.runtime_profile)}</p>`);
  }
  if (typeof data.confidence_score === "number") {
    chunks.push(`<p><span class="inline-code">confidence</span> ${esc((data.confidence_score * 100).toFixed(1))}%</p>`);
  }
  if (data.traceback_summary) {
    chunks.push(`<p><span class="inline-code">traceback</span> ${esc(data.traceback_summary)}</p>`);
  }
  if (typeof data.hallucination_risk === "number") {
    chunks.push(`<p><span class="inline-code">hallucination-risk</span> ${esc((data.hallucination_risk * 100).toFixed(1))}%</p>`);
  }

  if (data.rag_enabled) {
    const sources = Array.isArray(data.rag_sources) ? data.rag_sources : [];
    chunks.push(`<p><span class="inline-code">corrective-rag</span> Mode: ${esc(data.corrective_rag_mode || "balanced")}</p>`);
    if (sources.length > 0) {
      chunks.push(`<p><span class="inline-code">rag</span> Retrieved context from ${esc(sources.join(", "))}</p>`);
    } else {
      chunks.push("<p><span class=\"inline-code\">rag</span> Enabled, but no project files were retrieved for this request.</p>");
    }
  }

  if (data.failure_diagnostics && data.failure_diagnostics.category && data.failure_diagnostics.category !== "none") {
    chunks.push(
      `<p><span class="inline-code">failure</span> ${esc(data.failure_diagnostics.category)} at ${esc(data.failure_diagnostics.stage)} - ${esc(data.failure_diagnostics.summary || "")}</p>`
    );
  }

  chunks.push(mkCode("python", data.combined_code));

  if (Array.isArray(data.events) && data.events.length > 0) {
    const events = data.events.map((event) => {
      const label = event.iteration ? `Attempt ${event.iteration}` : "Event";
      return `<p><span class="inline-code">${esc(label)}</span> ${esc(event.stage)} - ${esc(event.detail)}</p>`;
    }).join("");
    chunks.push(events);
  }

  if (data.generated_tests) {
    chunks.push("<p><span class=\"inline-code\">generated-tests</span> Suggested unit tests:</p>");
    chunks.push(mkCode("python", data.generated_tests));
  }
  if (typeof data.regression_test_passed === "boolean") {
    chunks.push(mkExec(data.regression_test_passed ? "ok" : "err", data.regression_test_output || "Regression test status unavailable."));
  }
  if (data.repair_diff) {
    chunks.push("<p><span class=\"inline-code\">repair-diff</span> Incremental patch:</p>");
    chunks.push(mkCode("diff", data.repair_diff));
  }

  return chunks.join("");
}

function renderAssistantBodyCompact(data) {
  const chunks = [];
  const iterations = Number(data.iterations || 1);
  const statusText = (data.semantic_validation_passed ?? data.validation_passed) ? "passed" : "failed";
  chunks.push(`<p><span class="inline-code">status</span> ${statusText} after ${iterations} iteration(s)</p>`);

  if (data.solution && data.solution.prefix) {
    chunks.push(`<p>${esc(data.solution.prefix)}</p>`);
  } else {
    chunks.push("<p>No explanation text was returned by the model. Showing validated code below.</p>");
  }

  if (data.validation_message) {
    chunks.push(`<p><span class="inline-code">validation</span> ${esc(data.validation_message)}</p>`);
  }
  if (typeof data.confidence_score === "number") {
    chunks.push(`<p><span class="inline-code">confidence</span> ${esc((data.confidence_score * 100).toFixed(1))}%</p>`);
  }
  if (typeof data.hallucination_risk === "number") {
    chunks.push(`<p><span class="inline-code">hallucination-risk</span> ${esc((data.hallucination_risk * 100).toFixed(1))}%</p>`);
  }

  if (data.rag_enabled) {
    const sources = Array.isArray(data.rag_sources) ? data.rag_sources : [];
    chunks.push(`<p><span class="inline-code">rag-mode</span> ${esc(data.corrective_rag_mode || "balanced")}</p>`);
    if (sources.length > 0) {
      chunks.push(`<p><span class="inline-code">sources</span> ${esc(sources.join(", "))}</p>`);
    }
  }

  if (Array.isArray(data.events) && data.events.length > 0) {
    const eventLines = data.events.map((event) => {
      const attempt = event.iteration ? `attempt ${event.iteration}` : "system";
      return `<p><span class="inline-code">${esc(attempt)}</span> ${esc(event.stage)} - ${esc(event.detail || "")}</p>`;
    }).join("");
    chunks.push(eventLines);
  }

  chunks.push(mkCode("python", data.combined_code || ""));
  return chunks.join("");
}

function addAiMsg(data) {
  const d = document.createElement("div");
  d.className = "msg-row";
  const tag = (data.model || "").split("/").pop();
  d.innerHTML = `<div class="msg-ai-header"><div class="ai-avatar">lg</div><span class="ai-name">Code Assistant</span><span class="ai-model-tag">${esc(tag)}</span></div><div class="ai-body">${renderAssistantBody(data)}</div>`;
  const controls = document.createElement("div");
  controls.className = "input-hints";
  controls.style.marginTop = "8px";
  controls.innerHTML = `
    <span class="hint-pill">Feedback:</span>
    <button class="tbtn" type="button">Correct</button>
    <button class="tbtn" type="button">Partial</button>
    <button class="tbtn" type="button">Wrong</button>
  `;
  const buttons = controls.querySelectorAll("button");
  if (buttons.length === 3) {
    buttons[0].addEventListener("click", () => submitFeedback(data, "correct", 5));
    buttons[1].addEventListener("click", () => submitFeedback(data, "partially_correct", 3));
    buttons[2].addEventListener("click", () => submitFeedback(data, "wrong", 1));
  }
  d.appendChild(controls);
  document.getElementById("msgs").appendChild(d);
  attachCopyButtons(d);
  scrollD();
}

async function submitFeedback(data, verdict, rating) {
  try {
    await requestJson("/api/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        thread_id: data.thread_id || "unknown",
        verdict,
        rating,
        provider: data.provider || "unknown",
        model: data.model || "unknown",
        runtime_profile: data.runtime_profile || "custom",
        rag_enabled: Boolean(data.rag_enabled),
        corrective_rag_mode: data.corrective_rag_mode || "balanced",
        confidence_score: Number(data.confidence_score || 0),
        hallucination_risk: Number(data.hallucination_risk || 0),
        comment: ""
      })
    });
    addLog(`Feedback saved: ${verdict}.`);
  } catch (err) {
    addLog(`Feedback save failed: ${err.message || "request failed"}`);
  }
}

function addArenaMessage(targetId, role, html) {
  const mount = byId(targetId);
  if (!mount) {
    return;
  }
  const row = document.createElement("div");
  row.className = `arena-msg ${role}`;
  row.innerHTML = `<div class="bubble">${html}</div>`;
  mount.appendChild(row);
  attachCopyButtons(row);
  mount.scrollTop = mount.scrollHeight;
}

function updateStats(data) {
  document.getElementById("sI").textContent = data.iterations || "1";
  const passed = (data.semantic_validation_passed ?? data.validation_passed);
  document.getElementById("sR").textContent = passed ? "success" : "needs fix";
  setStat(passed ? "done" : "error");
}

function reflectEvents(data, showEvents) {
  donePipe();
  addLog(`Completed in ${data.iterations} iteration(s) using ${data.provider}.`);
  if (showEvents && Array.isArray(data.events)) {
    data.events.forEach((event) => addLog(`${event.stage}: ${event.detail}`));
  }
}

function setSliderCaps(config) {
  const maxIter = document.getElementById("maxIter");
  const timeout = document.getElementById("timeoutR");

  maxIter.max = String(config.maxIterationsCap);
  timeout.max = String(config.validationTimeoutCap);

  if (Number(maxIter.value) > config.maxIterationsCap) {
    maxIter.value = String(config.maxIterationsCap);
    document.getElementById("iN").textContent = maxIter.value;
  }
  if (Number(timeout.value) > config.validationTimeoutCap) {
    timeout.value = String(config.validationTimeoutCap);
    document.getElementById("tN").textContent = timeout.value;
  }
}

function applyProviders(config) {
  const providerSel = document.getElementById("providerSel");
  const allowed = new Set(config.allowedProviders || ["mistral", "openai", "openrouter"]);
  Array.from(providerSel.options).forEach((option) => {
    option.hidden = !allowed.has(option.value);
    option.disabled = !allowed.has(option.value);
  });

  const nextProvider = allowed.has(providerSel.value)
    ? providerSel.value
    : (allowed.has(config.defaultProvider) ? config.defaultProvider : (config.allowedProviders[0] || "mistral"));
  providerSel.value = nextProvider;
  handleProvider();
}

function applyRagConfig(config) {
  const ragToggle = byId("ragToggle");
  const correctiveMode = byId("correctiveRagMode");
  const runtimeProfile = byId("runtimeProfile");
  if (!ragToggle) {
    return;
  }
  if (runtimeProfile) {
    const profiles = Array.isArray(config.runtimeProfiles) && config.runtimeProfiles.length > 0
      ? config.runtimeProfiles
      : ["custom", "fast", "balanced", "accurate", "goated"];
    runtimeProfile.innerHTML = profiles
      .map((profile) => `<option value="${esc(profile)}">${esc(profile)}</option>`)
      .join("");
    runtimeProfile.value = config.defaultRuntimeProfile || "custom";
  }
  ragToggle.checked = Boolean(config.ragDefaultEnabled);
  if (correctiveMode) {
    const modes = Array.isArray(config.correctiveRagModes) && config.correctiveRagModes.length > 0
      ? config.correctiveRagModes
      : ["fast", "balanced", "aggressive"];
    correctiveMode.innerHTML = modes
      .map((mode) => `<option value="${esc(mode)}">${esc(mode)}</option>`)
      .join("");
    correctiveMode.value = config.correctiveRagDefaultMode || "balanced";
    correctiveMode.disabled = !config.ragAvailable;
  }
  ragToggle.disabled = !config.ragAvailable;
  if ((config.defaultRuntimeProfile || "custom") !== "custom") {
    applyRuntimeProfile();
  } else {
    syncRag(true);
    updatePills();
  }
  if (!config.ragAvailable) {
    addLog("Project RAG is unavailable for this deployment.");
  }
}

async function requestJson(path, init = {}) {
  const response = await fetch(apiUrl(path), init);
  const contentType = response.headers.get("content-type") || "";
  const text = await response.text();
  let data = null;

  if (text) {
    if (contentType.includes("application/json")) {
      try {
        data = JSON.parse(text);
      } catch (err) {
        throw new Error("The backend returned malformed JSON.");
      }
    } else {
      const looksLikeHtml = /^\s*</.test(text);
      throw new Error(
        looksLikeHtml
          ? "The app could not reach backend API routes on this same domain."
          : "The backend did not return JSON."
      );
    }
  }

  if (!response.ok) {
    throw new Error((data && data.detail) || "Request failed.");
  }
  return data || {};
}

async function send() {
  if (APP_STATE.running || APP_STATE.arenaRunning) {
    return;
  }

  const promptEl = document.getElementById("prompt");
  const prompt = promptEl.value.trim();
  if (!prompt) {
    return;
  }

  if (APP_STATE.config.authRequired) {
    addErrMsg("Backend token auth is enabled. This simplified UI does not ask end-users for tokens.");
    setStat("error");
    return;
  }

  if (!APP_STATE.arenaMode) {
    document.getElementById("welcome").style.display = "none";
    document.getElementById("msgs").style.display = "";
  } else {
    const arenaWrap = byId("arenaWrap");
    const msgs = byId("msgs");
    const welcome = byId("welcome");
    if (arenaWrap) {
      arenaWrap.style.display = "";
    }
    if (msgs) {
      msgs.style.display = "none";
    }
    if (welcome) {
      welcome.style.display = "none";
    }
  }

  const payload = {
    prompt,
    provider: valueOf("providerSel", APP_STATE.config.defaultProvider || "mistral"),
    model: valueOf("modelSel", defaultModelForProvider(valueOf("providerSel", APP_STATE.config.defaultProvider || "mistral"))),
    provider_key_id: valueOf("savedKeySel", ""),
    local_model: valueOf("localPath", "Qwen/Qwen2.5-Coder-0.5B-Instruct"),
    max_iterations: Number(valueOf("maxIter", "3")),
    validation_timeout: Number(valueOf("timeoutR", "5")),
    show_events: checkedOf("showEvents", false),
    json_mode: false,
    tracing: false,
    rag_enabled: checkedOf("ragToggle", APP_STATE.config.ragDefaultEnabled),
    corrective_rag_mode: valueOf("correctiveRagMode", APP_STATE.config.correctiveRagDefaultMode),
    runtime_profile: valueOf("runtimeProfile", APP_STATE.config.defaultRuntimeProfile),
    attachment_ids: APP_STATE.attachments.map((item) => item.id),
    attachment_mode: valueOf("attachMode", "rag_only")
  };

  if (APP_STATE.arenaMode) {
    addArenaMessage("arenaMsgsRag", "user", esc(prompt));
    addArenaMessage("arenaMsgsNormal", "user", esc(prompt));
  } else {
    addUserMsg(prompt);
  }
  promptEl.value = "";
  promptEl.style.height = "auto";
  APP_STATE.qCount += 1;
  document.getElementById("sQ").textContent = String(APP_STATE.qCount);
  document.getElementById("sendBtn").disabled = true;
  APP_STATE.running = !APP_STATE.arenaMode;
  APP_STATE.arenaRunning = APP_STATE.arenaMode;

  setStat("running");
  resetPipe();
  setPipe(APP_STATE.arenaMode || payload.rag_enabled ? "retrieve_context" : "generate_code");
  addLog(`Query: ${prompt.slice(0, 44)}${prompt.length > 44 ? "..." : ""}`);

  let tk = "";
  try {
    if (APP_STATE.arenaMode) {
      const provider = payload.provider;
      const model = provider !== "local"
        ? valueOf("modelSel", defaultModelForProvider(provider))
        : valueOf("localPath", "Qwen/Qwen2.5-Coder-0.5B-Instruct");
      const ragModelTag = byId("arenaRagModel");
      const normalModelTag = byId("arenaNormalModel");
      if (ragModelTag) {
        ragModelTag.textContent = model;
      }
      if (normalModelTag) {
        normalModelTag.textContent = model;
      }
      addArenaMessage("arenaMsgsRag", "ai", '<div class="arena-loading">running...</div>');
      addArenaMessage("arenaMsgsNormal", "ai", '<div class="arena-loading">running...</div>');
      const ragPayload = {
        ...payload,
        runtime_profile: "goated",
        rag_enabled: true,
        corrective_rag_mode: "aggressive",
        max_iterations: Math.max(payload.max_iterations, 6),
        validation_timeout: Math.max(payload.validation_timeout, 12),
        show_events: true,
        attachment_ids: payload.attachment_ids,
        attachment_mode: payload.attachment_mode
      };
      const normalPayload = {
        ...payload,
        runtime_profile: "custom",
        rag_enabled: false,
        corrective_rag_mode: "fast",
        max_iterations: 1,
        validation_timeout: Math.min(payload.validation_timeout, 5),
        show_events: true,
        attachment_ids: payload.attachment_mode === "both" ? payload.attachment_ids : [],
        attachment_mode: payload.attachment_mode
      };
      const [ragData, normalData] = await Promise.all([
        requestJson("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(ragPayload)
        }),
        requestJson("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(normalPayload)
        })
      ]);
      const ragMount = byId("arenaMsgsRag");
      const normalMount = byId("arenaMsgsNormal");
      if (ragMount && ragMount.lastElementChild) {
        ragMount.lastElementChild.remove();
      }
      if (normalMount && normalMount.lastElementChild) {
        normalMount.lastElementChild.remove();
      }
      addArenaMessage("arenaMsgsRag", "ai", renderAssistantBodyCompact(ragData));
      addArenaMessage("arenaMsgsNormal", "ai", renderAssistantBodyCompact(normalData));
      setPipe("execute_code");
      setPipe("check_result");
      setPipe("retry_or_end");
      const ragPassed = Boolean(ragData.semantic_validation_passed ?? ragData.validation_passed);
      const normalPassed = Boolean(normalData.semantic_validation_passed ?? normalData.validation_passed);
      const bothPassed = ragPassed && normalPassed;
      document.getElementById("sR").textContent = bothPassed ? "both success" : "check arena";
      document.getElementById("sI").textContent = `${ragData.iterations || 1}/${normalData.iterations || 1}`;
      setStat(bothPassed ? "done" : "error");
      addLog(`Arena done: RAG=${ragPassed ? "ok" : "fail"}, Normal=${normalPassed ? "ok" : "fail"}.`);
    } else {
      tk = addThink();
      const data = await requestJson("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      rmThink(tk);
      setPipe("execute_code");
      setPipe("check_result");
      setPipe("retry_or_end");
      addAiMsg(data);
      updateStats(data);
      reflectEvents(data, payload.show_events);
    }
  } catch (err) {
    if (tk) {
      rmThink(tk);
    }
    setStat("error");
    resetPipe();
    if (APP_STATE.arenaMode) {
      addArenaMessage("arenaMsgsRag", "ai", mkExec("err", err.message || "Arena request failed."));
      addArenaMessage("arenaMsgsNormal", "ai", mkExec("err", err.message || "Arena request failed."));
    } else {
      addErrMsg(err.message || "Something went wrong.");
    }
    addLog(`Error: ${err.message || "Request failed."}`);
  } finally {
    APP_STATE.running = false;
    APP_STATE.arenaRunning = false;
    document.getElementById("sendBtn").disabled = false;
    scrollD();
  }
}

async function boot() {
  APP_STATE.modelsByProvider = { ...MODEL_OPTIONS };

  const localPath = byId("localPath");
  const attachmentInput = byId("attachmentInput");
  if (localPath) {
    localPath.addEventListener("input", () => {
      markRuntimeProfileCustom();
      updatePills();
    });
  }
  if (attachmentInput) {
    attachmentInput.addEventListener("change", async (event) => {
      try {
        const files = event.target && event.target.files ? event.target.files : [];
        if (!files || files.length === 0) {
          return;
        }
        await uploadAttachments(files);
      } catch (err) {
        addErrMsg(err.message || "Attachment upload failed.");
        addLog(`Attachment upload failed: ${err.message || "request failed"}`);
      } finally {
        event.target.value = "";
      }
    });
  }

  updatePills();
  renderAttachmentChips();
  syncRag();

  try {
    const backendConfig = await requestJson("/api/config");
    APP_STATE.config = {
      ...APP_STATE.config,
      allowedProviders: backendConfig.allowed_providers,
      defaultProvider: backendConfig.default_provider,
      authRequired: backendConfig.auth_required,
      maxIterationsCap: backendConfig.max_iterations_cap,
      validationTimeoutCap: backendConfig.validation_timeout_cap,
      rateLimitRequests: backendConfig.rate_limit_requests,
      rateLimitWindowSeconds: backendConfig.rate_limit_window_seconds,
      ragAvailable: backendConfig.rag_available,
      ragDefaultEnabled: backendConfig.rag_default_enabled,
      correctiveRagModes: backendConfig.corrective_rag_modes,
      correctiveRagDefaultMode: backendConfig.corrective_rag_default_mode,
      runtimeProfiles: backendConfig.runtime_profiles,
      defaultRuntimeProfile: backendConfig.default_runtime_profile,
      userKeysEnabled: Boolean(backendConfig.user_keys_enabled),
      userKeysPersistent: Boolean(backendConfig.user_keys_persistent),
      userKeysMaxEntries: Number(backendConfig.user_keys_max_entries || 50)
    };
    setSliderCaps(APP_STATE.config);
    applyProviders(APP_STATE.config);
    applyRagConfig(APP_STATE.config);
    updatePills();
    addLog("Backend config loaded.");
    addLog(
      `Rate limit: ${APP_STATE.config.rateLimitRequests} request(s) per ${APP_STATE.config.rateLimitWindowSeconds}s.`
    );
    if (APP_STATE.config.userKeysEnabled) {
      addLog(
        APP_STATE.config.userKeysPersistent
          ? "BYOK enabled (encrypted persistent key vault)."
          : "BYOK enabled (ephemeral vault; saved keys reset on backend restart)."
      );
      setKeyControlsVisibility();
      await syncProviderRuntime(selectedProvider(), valueOf("modelSel", ""));
    } else {
      setKeyControlsVisibility();
    }
    if (APP_STATE.config.authRequired) {
      addLog("Backend auth token is enabled. Disable CODE_ASSISTANT_ACCESS_TOKEN for this embedded UI.");
      setStat("error");
      return;
    }
  } catch (err) {
    addLog("Config check failed. Ensure this app is served by the backend (same domain).");
    setStat("error");
    return;
  }

  try {
    const health = await requestJson("/api/health");
    if (health.status !== "ok") {
      throw new Error("Backend unavailable.");
    }
    addLog("Backend ready.");
  } catch (err) {
    addLog("Backend health check failed. Start the backend before sending prompts.");
    setStat("error");
  }
}

window.addEventListener("load", boot);
