// Shared helpers and the service status pill in the header.
"use strict";

window.app = {
  async json(url, options = {}) {
    const response = await fetch(url, options);
    let body = null;
    try { body = await response.json(); } catch (_) { body = null; }
    if (!response.ok) {
      const detail = body && body.detail ? body.detail : `Request failed (${response.status})`;
      throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
    }
    return body;
  },
  busy(button, isBusy, label) {
    if (!button) return;
    if (isBusy) {
      button.dataset.label = button.textContent;
      button.setAttribute("aria-busy", "true");
      button.disabled = true;
      button.innerHTML = '<span class="spinner" aria-hidden="true"></span>' + (label || button.textContent);
    } else {
      button.removeAttribute("aria-busy");
      button.disabled = false;
      if (button.dataset.label) button.textContent = button.dataset.label;
    }
  },
};

(async function statusPill() {
  const pill = document.getElementById("status-pill");
  if (!pill) return;
  const text = pill.querySelector("span");
  try {
    const health = await window.app.json("/api/health");
    window.app.health = health;
    document.dispatchEvent(new CustomEvent("health", { detail: health }));
    const modelOk = health.ollama.reachable && health.ollama.missing_models.length === 0;
    const indexOk = health.store.reachable && health.store.points > 0;
    if (modelOk && indexOk) {
      pill.dataset.state = "ok"; text.textContent = "Ready";
    } else if (!health.ollama.reachable) {
      pill.dataset.state = "down"; text.textContent = "Ollama is not running";
    } else if (!modelOk) {
      pill.dataset.state = "warn"; text.textContent = "Models missing";
    } else {
      pill.dataset.state = "warn"; text.textContent = "Legal index not built";
    }
  } catch (_) {
    pill.dataset.state = "down"; text.textContent = "Services unavailable";
  }
})();
