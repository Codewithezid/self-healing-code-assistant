const DEFAULT_APP_CONFIG = {
  defaultProvider: "mistral",
  allowedProviders: ["openai", "mistral"],
  authRequired: false,
  maxIterationsCap: 3,
  validationTimeoutCap: 5,
  ragAvailable: true,
  ragDefaultEnabled: false,
  correctiveRagModes: ["fast", "balanced", "aggressive"],
  correctiveRagDefaultMode: "balanced",
  runtimeProfiles: ["custom", "fast", "balanced", "accurate"],
  defaultRuntimeProfile: "custom",
  userKeysEnabled: false,
  userKeysPersistent: false,
  userKeysMaxEntries: 50
};

const APP_STATE = {
  running: false,
  applyingProfile: false,
  qCount: 0,
  tkN: 0,
  jMode: false,
  ragMode: false,
  keysByProvider: {},
  modelsByProvider: {},
  config: { ...DEFAULT_APP_CONFIG, ...(window.APP_CONFIG || {}) }
};

const MODEL_OPTIONS = {
  openai: ["gpt-5.1", "gpt-5", "gpt-4.1"],
  mistral: ["codestral-latest", "mistral-medium-latest", "mistral-small-latest"]
};

const DEFAULT_MODEL_BY_PROVIDER = {
  openai: "gpt-5.1",
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
    ? ['<option value="" disabled selected>no saved keys</option>']
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
      const filtered = payload.models.filter((model) => curated.includes(model));
      APP_STATE.modelsByProvider[provider] = filtered.length > 0 ? filtered : curated;
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
      const curated = MODEL_OPTIONS[provider] || [];
      const filtered = payload.models.filter((model) => curated.includes(model));
      APP_STATE.modelsByProvider[provider] = filtered.length > 0 ? filtered : curated;
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

function toggleJson() {
  const cb = byId("jsonToggle");
  if (!cb) {
    return;
  }
  cb.checked = !cb.checked;
  syncJson();
}

function syncJson() {
  APP_STATE.jMode = checkedOf("jsonToggle", false);
  const button = byId("jsonBtn");
  if (button) {
    button.style.cssText = APP_STATE.jMode ? "background:var(--ink);color:var(--paper);border-color:var(--ink)" : "";
  }
  const jsonPill = byId("pj");
  if (jsonPill) {
    jsonPill.classList.toggle("active", APP_STATE.jMode);
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
  document.getElementById("msgs").innerHTML = "";
  document.getElementById("msgs").style.display = "none";
  document.getElementById("welcome").style.display = "";
  document.getElementById("sQ").textContent = "0";
  document.getElementById("sI").textContent = "-";
  document.getElementById("sR").textContent = "-";
  setStat("idle");
  resetPipe();
  document.getElementById("actLog").innerHTML =
    '<div class="log-entry"><span class="log-time">--</span><span class="log-txt" style="color:var(--ink4)">No activity yet.</span></div>';
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
      data.validation_passed ? "ok" : "err",
      data.validation_passed
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

  return chunks.join("");
}

function addAiMsg(data) {
  const d = document.createElement("div");
  d.className = "msg-row";
  const tag = (data.model || "").split("/").pop();
  d.innerHTML = `<div class="msg-ai-header"><div class="ai-avatar">lg</div><span class="ai-name">Code Assistant</span><span class="ai-model-tag">${esc(tag)}</span></div><div class="ai-body">${renderAssistantBody(data)}</div>`;
  document.getElementById("msgs").appendChild(d);
  attachCopyButtons(d);
  scrollD();
}

function updateStats(data) {
  document.getElementById("sI").textContent = data.iterations || "1";
  document.getElementById("sR").textContent = data.validation_passed ? "success" : "needs fix";
  setStat(data.validation_passed ? "done" : "error");
}

function reflectEvents(data, showEvents) {
  donePipe();
  addLog(`Completed in ${data.iterations} iteration(s) using ${data.provider}.`);
  if (data.tracing_requested) {
    addLog("Tracing was requested. Actual LangSmith traces still depend on backend LANGCHAIN_* environment settings.");
  }
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
  const allowed = new Set(config.allowedProviders || ["mistral", "openai"]);
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
      : ["custom", "fast", "balanced", "accurate"];
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
  if (APP_STATE.running) {
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

  document.getElementById("welcome").style.display = "none";
  document.getElementById("msgs").style.display = "";

  const payload = {
    prompt,
    provider: valueOf("providerSel", APP_STATE.config.defaultProvider || "mistral"),
    model: valueOf("modelSel", defaultModelForProvider(valueOf("providerSel", APP_STATE.config.defaultProvider || "mistral"))),
    provider_key_id: valueOf("savedKeySel", ""),
    local_model: valueOf("localPath", "Qwen/Qwen2.5-Coder-0.5B-Instruct"),
    max_iterations: Number(valueOf("maxIter", "3")),
    validation_timeout: Number(valueOf("timeoutR", "5")),
    show_events: checkedOf("showEvents", false),
    json_mode: checkedOf("jsonToggle", false),
    tracing: checkedOf("tracing", false),
    rag_enabled: checkedOf("ragToggle", APP_STATE.config.ragDefaultEnabled),
    corrective_rag_mode: valueOf("correctiveRagMode", APP_STATE.config.correctiveRagDefaultMode),
    runtime_profile: valueOf("runtimeProfile", APP_STATE.config.defaultRuntimeProfile)
  };

  addUserMsg(prompt);
  promptEl.value = "";
  promptEl.style.height = "auto";
  APP_STATE.qCount += 1;
  document.getElementById("sQ").textContent = String(APP_STATE.qCount);
  document.getElementById("sendBtn").disabled = true;
  APP_STATE.running = true;

  setStat("running");
  resetPipe();
  setPipe(payload.rag_enabled ? "retrieve_context" : "generate_code");
  addLog(`Query: ${prompt.slice(0, 44)}${prompt.length > 44 ? "..." : ""}`);

  const tk = addThink();

  try {
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
  } catch (err) {
    rmThink(tk);
    setStat("error");
    resetPipe();
    addErrMsg(err.message || "Something went wrong.");
    addLog(`Error: ${err.message || "Request failed."}`);
  } finally {
    APP_STATE.running = false;
    document.getElementById("sendBtn").disabled = false;
    scrollD();
  }
}

async function boot() {
  APP_STATE.modelsByProvider = { ...MODEL_OPTIONS };

  const localPath = byId("localPath");
  if (localPath) {
    localPath.addEventListener("input", () => {
      markRuntimeProfileCustom();
      updatePills();
    });
  }

  updatePills();
  syncJson();
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
