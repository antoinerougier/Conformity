// Report page: live audit progress, interactive checklist, report assistant.
"use strict";

(function () {
  const $ = (id) => document.getElementById(id);

  // ------------------------------------------------------------ live progress (SSE)
  const progressView = document.querySelector(".progress-view[data-status='auditing']");
  if (progressView) {
    const sessionId = progressView.dataset.session;
    const list = $("progress-list");
    const running = $("progress-running");
    const seen = list.querySelectorAll("li[data-state='done']").length;
    let received = 0;
    const source = new EventSource(`/api/sessions/${sessionId}/events`);
    source.addEventListener("progress", (event) => {
      received += 1;
      if (received <= seen) return; // already rendered server-side
      const entry = JSON.parse(event.data);
      const li = document.createElement("li");
      li.dataset.state = "done";
      const strong = document.createElement("strong");
      strong.textContent = entry.label;
      li.appendChild(strong);
      entry.trace.forEach((line) => { const s = document.createElement("small"); s.textContent = line; li.appendChild(s); });
      list.insertBefore(li, running);
    });
    source.addEventListener("done", () => { source.close(); window.location.reload(); });
    source.addEventListener("failed", () => { source.close(); window.location.reload(); });
  }

  // ------------------------------------------------------------ checklist
  const checklist = $("checklist");
  if (checklist) {
    const sessionId = checklist.dataset.session;
    const boxes = checklist.querySelectorAll("input[type=checkbox][data-item]");
    const refresh = () => {
      const done = Array.from(boxes).filter((b) => b.checked).length;
      $("checklist-count").textContent = `${done} of ${boxes.length} done`;
      $("checklist-bar").style.width = `${boxes.length ? (100 * done) / boxes.length : 0}%`;
    };
    boxes.forEach((box) => box.addEventListener("change", async () => {
      refresh();
      try {
        await window.app.json(`/api/sessions/${sessionId}/checklist/${encodeURIComponent(box.dataset.item)}`, {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ done: box.checked }),
        });
      } catch (_) {
        box.checked = !box.checked; // revert when the server refused the change
        refresh();
      }
    }));
  }

  // ------------------------------------------------------------ table of contents highlight
  const tocLinks = document.querySelectorAll(".toc a");
  if (tocLinks.length && "IntersectionObserver" in window) {
    const byId = new Map(Array.from(tocLinks).map((a) => [a.getAttribute("href").slice(1), a]));
    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        tocLinks.forEach((a) => a.removeAttribute("aria-current"));
        const link = byId.get(entry.target.id);
        if (link) link.setAttribute("aria-current", "true");
      });
    }, { rootMargin: "-30% 0px -60% 0px" });
    byId.forEach((_, id) => { const el = $(id); if (el) observer.observe(el); });
  }

  // ------------------------------------------------------------ report assistant
  const chat = $("chat");
  if (!chat) return;
  const sessionId = chat.dataset.session;
  const toggle = $("chat-toggle");
  const log = $("chat-log");
  const input = $("chat-input");
  const open = (state) => {
    chat.dataset.open = String(state);
    toggle.setAttribute("aria-expanded", String(state));
    if (state) input.focus();
  };
  toggle.addEventListener("click", () => open(chat.dataset.open !== "true"));
  $("chat-close").addEventListener("click", () => { open(false); toggle.focus(); });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape" && chat.dataset.open === "true") open(false); });

  const bubble = (role, text) => {
    const empty = $("chat-empty");
    if (empty) empty.remove();
    const div = document.createElement("div");
    div.className = `msg msg--${role}`;
    div.textContent = text;
    log.appendChild(div);
    log.scrollTop = log.scrollHeight;
    return div;
  };

  async function ask(question) {
    bubble("user", question);
    const answer = bubble("assistant", "");
    answer.setAttribute("aria-busy", "true");
    try {
      const response = await fetch(`/api/sessions/${sessionId}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });
      if (!response.ok) throw new Error((await response.json()).detail || "The assistant is unavailable.");
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        answer.textContent += decoder.decode(value, { stream: true });
        log.scrollTop = log.scrollHeight;
      }
      const sources = JSON.parse(response.headers.get("X-Sources") || "[]");
      if (sources.length) {
        const s = document.createElement("span");
        s.className = "sources";
        s.textContent = "Sources: " + sources.join(", ");
        answer.appendChild(s);
      }
    } catch (e) {
      answer.textContent = e.message;
    } finally {
      answer.removeAttribute("aria-busy");
    }
  }

  $("chat-form").addEventListener("submit", (event) => {
    event.preventDefault();
    const q = input.value.trim();
    if (!q) return;
    input.value = "";
    ask(q);
  });
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("chat-form").requestSubmit(); }
  });
  log.querySelectorAll("[data-suggest]").forEach((b) => b.addEventListener("click", () => { open(true); ask(b.textContent); }));
})();
