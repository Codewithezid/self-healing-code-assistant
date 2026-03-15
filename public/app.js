const DEFAULT_APP_CONFIG = {
  defaultProvider: "mistral",
  allowedProviders: ["mistral"],
  authRequired: false,
  maxIterationsCap: 3,
  validationTimeoutCap: 5
};

const APP_STATE = {
  running: false,
  qCount: 0,
  tkN: 0,
  jMode: false,
  config: { ...DEFAULT_APP_CONFIG, ...(window.APP_CONFIG || {}) }
};

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

function handleProvider() {
  const provider = document.getElementById("providerSel").value;
  document.getElementById("modelField").style.display = provider === "mistral" ? "" : "none";
  document.getElementById("localField").style.display = provider === "local" ? "" : "none";
  document.getElementById("pp").textContent = provider;
  updatePills();
}

function updatePills() {
  const provider = document.getElementById("providerSel").value;
  const model = provider === "mistral"
    ? document.getElementById("modelSel").value
    : document.getElementById("localPath").value.split("/").pop();
  document.getElementById("pm").textContent = model || "local-model";
  document.getElementById("pi").textContent = document.getElementById("maxIter").value + " retries";
}

function toggleJson() {
  const cb = document.getElementById("jsonToggle");
  cb.checked = !cb.checked;
  syncJson();
}

function syncJson() {
  APP_STATE.jMode = document.getElementById("jsonToggle").checked;
  const button = document.getElementById("jsonBtn");
  button.style.cssText = APP_STATE.jMode ? "background:var(--ink);color:var(--paper);border-color:var(--ink)" : "";
  document.getElementById("pj").classList.toggle("active", APP_STATE.jMode);
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
  const order = ["generate_code", "execute_code", "check_result", "retry_or_end"];
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
  const allowed = new Set(config.allowedProviders || ["mistral"]);
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
    provider: document.getElementById("providerSel").value,
    model: document.getElementById("modelSel").value,
    local_model: document.getElementById("localPath").value,
    max_iterations: Number(document.getElementById("maxIter").value),
    validation_timeout: Number(document.getElementById("timeoutR").value),
    show_events: document.getElementById("showEvents").checked,
    json_mode: document.getElementById("jsonToggle").checked,
    tracing: document.getElementById("tracing").checked
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
  setPipe("generate_code");
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
  document.getElementById("localPath").addEventListener("input", updatePills);

  updatePills();
  syncJson();

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
      rateLimitWindowSeconds: backendConfig.rate_limit_window_seconds
    };
    setSliderCaps(APP_STATE.config);
    applyProviders(APP_STATE.config);
    addLog("Backend config loaded.");
    addLog(
      `Rate limit: ${APP_STATE.config.rateLimitRequests} request(s) per ${APP_STATE.config.rateLimitWindowSeconds}s.`
    );
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
