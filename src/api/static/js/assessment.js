// Assessment page: start an interview, answer rounds, launch the audit.
"use strict";

(function () {
  const $ = (id) => document.getElementById(id);

  // ------------------------------------------------------------ start form
  const startForm = $("start-form");
  if (startForm) {
    const fileInput = $("file");
    const dropzone = $("dropzone");
    const showName = () => { $("file-name").textContent = fileInput.files.length ? fileInput.files[0].name : ""; };
    fileInput.addEventListener("change", showName);
    ["dragenter", "dragover"].forEach((t) => dropzone.addEventListener(t, () => { dropzone.dataset.dragging = "true"; }));
    ["dragleave", "drop"].forEach((t) => dropzone.addEventListener(t, () => { dropzone.dataset.dragging = "false"; }));

    startForm.addEventListener("submit", async (event) => {
      event.preventDefault();
      const error = $("start-error");
      error.textContent = "";
      const description = $("description").value.trim();
      if (!fileInput.files.length && description.length < 40) {
        error.textContent = "Upload a document, or describe the project in at least a few sentences.";
        return;
      }
      const data = new FormData();
      if (fileInput.files.length) data.append("file", fileInput.files[0]);
      data.append("description", description);
      const button = $("start-button");
      window.app.busy(button, true, "Reading");
      $("start-working").dataset.visible = "true";
      try {
        const session = await window.app.json("/api/sessions", { method: "POST", body: data });
        window.location.href = `/assessment/${session.session_id}`;
      } catch (e) {
        error.textContent = e.message;
        window.app.busy(button, false);
        $("start-working").dataset.visible = "false";
      }
    });
  }

  // ------------------------------------------------------------ import a saved report
  const importForm = $("import-form");
  if (importForm) {
    importForm.addEventListener("submit", async (event) => {
      event.preventDefault();
      const input = $("import-file");
      if (!input.files.length) { $("import-error").textContent = "Choose a JSON report exported from this application."; return; }
      const data = new FormData();
      data.append("file", input.files[0]);
      try {
        const session = await window.app.json("/api/reports/import", { method: "POST", body: data });
        window.location.href = `/report/${session.session_id}`;
      } catch (e) {
        $("import-error").textContent = e.message;
      }
    });
  }

  // ------------------------------------------------------------ interview
  const root = document.querySelector("[data-session]");
  const sessionId = root ? root.dataset.session : null;
  const answersForm = $("answers-form");

  function collectAnswers() {
    const answers = [];
    answersForm.querySelectorAll("fieldset[data-qid]").forEach((fs) => {
      const qid = fs.dataset.qid;
      let answer = "";
      if (fs.dataset.type === "text") {
        answer = fs.querySelector("textarea").value.trim();
      } else {
        const picked = fs.querySelector("input[type=radio]:checked");
        const detail = fs.querySelector("textarea.detail").value.trim();
        answer = [picked ? picked.value : "", detail].filter(Boolean).join(". ");
      }
      answers.push({ question_id: qid, answer, skipped: answer.length === 0 });
    });
    return answers;
  }

  async function send(finish) {
    const error = $("answers-error");
    error.textContent = "";
    const button = finish ? $("finish-button") : $("answers-button");
    window.app.busy(button, true, finish ? "Saving" : "Sending");
    $("answers-working").dataset.visible = "true";
    try {
      const session = await window.app.json(`/api/sessions/${sessionId}/answers`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ answers: collectAnswers(), finish }),
      });
      if (finish && session.status === "ready") { await startAudit(); return; }
      window.location.reload();
    } catch (e) {
      error.textContent = e.message;
      window.app.busy(button, false);
      $("answers-working").dataset.visible = "false";
    }
  }

  async function startAudit() {
    await window.app.json(`/api/sessions/${sessionId}/audit`, { method: "POST" });
    window.location.href = `/report/${sessionId}`;
  }

  if (answersForm) {
    answersForm.addEventListener("submit", (event) => { event.preventDefault(); send(false); });
    $("finish-button").addEventListener("click", () => send(true));
  }

  const auditButton = $("audit-button");
  if (auditButton) {
    auditButton.addEventListener("click", async () => {
      window.app.busy(auditButton, true, "Starting");
      try { await startAudit(); } catch (e) {
        $("audit-error").textContent = e.message;
        window.app.busy(auditButton, false);
      }
    });
  }
})();
