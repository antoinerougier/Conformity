// Overview page: service health details and index rebuild.
"use strict";

(function () {
  const set = (key, text, ok) => {
    const el = document.querySelector(`[data-health="${key}"]`);
    if (!el) return;
    el.textContent = text;
    el.className = ok ? "" : "muted";
    el.style.color = ok ? "var(--low)" : "var(--critical)";
  };

  document.addEventListener("health", (event) => {
    const h = event.detail;
    if (!h.ollama.reachable) set("ollama", "Ollama is not running", false);
    else if (h.ollama.missing_models.length) set("ollama", "Missing: " + h.ollama.missing_models.join(", "), false);
    else set("ollama", h.llm_model, true);
    if (!h.store.reachable) set("store", "Qdrant unreachable", false);
    else if (!h.store.points) set("store", "Not built yet", false);
    else set("store", `${h.store.points} passages`, true);
    set("reranker", h.reranker || "Unavailable", Boolean(h.reranker && h.reranker !== "none"));
  });

  const button = document.getElementById("reindex");
  const status = document.getElementById("reindex-status");
  if (!button) return;

  async function poll() {
    const s = await window.app.json("/api/index");
    status.textContent = s.message;
    if (s.state === "running") { setTimeout(poll, 2000); return; }
    window.app.busy(button, false);
    if (s.state === "done") status.textContent = `Index rebuilt: ${s.chunks} passages.`;
  }

  button.addEventListener("click", async () => {
    window.app.busy(button, true, "Rebuilding");
    try {
      await window.app.json("/api/index", { method: "POST" });
      poll();
    } catch (error) {
      status.textContent = error.message;
      window.app.busy(button, false);
    }
  });
})();
