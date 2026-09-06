// Deployed backend. For local dev, change to "http://127.0.0.1:8000".
const API_URL = "https://jailbreak-api-backend.onrender.com";

// The backend is on Render's free tier: it spins down after 15 idle minutes and
// takes ~50s to come back. Start that boot on page load, so it overlaps with the
// visitor reading the page and typing a prompt rather than beginning when they
// click Check. Failure is expected and ignored — a sleeping instance often drops
// the connection outright, and it wakes anyway.
fetch(`${API_URL}/ping`).catch(() => {});

// Escape user/model text before inserting into the DOM (defense in depth — this
// is a security tool, it shouldn't be XSS-able by its own input).
function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

function riskColor(risk) {
  if (risk >= 0.7) return "#dc3545"; // red
  if (risk >= 0.4) return "#fd7e14"; // orange
  return "#198754"; // green
}

function statusLine(msg, cls) {
  const el = document.getElementById("result");
  el.innerHTML = `<div class="status-line ${cls}">${esc(msg)}</div>`;
}

function renderResult(data) {
  const el = document.getElementById("result");
  const malicious = data.verdict === "malicious" || data.detected === true;
  const risk = typeof data.risk_score === "number" ? data.risk_score : 0;
  const scanners = Array.isArray(data.scanners) ? data.scanners : [];

  const rows = scanners.map((s) => {
    const flagged = s.flagged;
    const matched = s.matched
      ? `<span class="matched">matched: <code>${esc(s.matched)}</code></span>`
      : "";
    return `
      <div class="scanner-row">
        <div>
          <span class="name">${esc(s.scanner)}</span>
          ${matched}
        </div>
        <div class="text-end">
          <span class="pill ${flagged ? "flag" : "clear"}">${flagged ? "flagged" : "clear"}</span>
          <div class="small text-muted mt-1">risk ${(s.risk_score ?? 0).toFixed(2)}</div>
        </div>
      </div>`;
  }).join("");

  // Fallback for an older backend that only returns {detected} with no breakdown.
  const breakdown = scanners.length
    ? rows
    : `<div class="text-muted small">No per-scanner breakdown returned.</div>`;

  el.innerHTML = `
    <div class="verdict">
      <span class="verdict-badge ${malicious ? "malicious" : "safe"}">
        ${malicious ? "⚠ MALICIOUS" : "✓ SAFE"}
      </span>
      <span class="text-muted small">aggregate risk ${(risk * 100).toFixed(0)}%</span>
    </div>
    <div class="meter">
      <span style="width:${Math.round(risk * 100)}%;background:${riskColor(risk)}"></span>
    </div>
    <div class="fw-bold mb-2 small text-uppercase text-muted">Scanner breakdown</div>
    ${breakdown}`;
}

let inFlight = false;

async function checkPrompt() {
  // Without this the button stays live through a 50s cold start and an impatient
  // visitor stacks up requests — straight into the backend's own rate limiter.
  if (inFlight) return;

  const promptText = document.getElementById("promptInput").value;
  if (!promptText.trim()) {
    statusLine("Please enter a prompt to analyze.", "text-muted");
    return;
  }

  const button = document.getElementById("checkBtn");
  inFlight = true;
  if (button) { button.disabled = true; button.textContent = "Checking…"; }
  statusLine("Analyzing…", "text-muted");

  // Past a few seconds this is no longer ordinary latency, it is the container
  // booting. Say so, or the page just looks hung.
  const slow = setTimeout(() => {
    statusLine(
      "Waking the server up. It sleeps when idle, so the first check can take up to a minute.",
      "text-muted");
  }, 4000);

  try {
    const response = await fetch(`${API_URL}/detect`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: promptText }),
    });

    if (response.status === 422) {
      statusLine("Invalid input: the prompt is empty or too long.", "text-warning");
      return;
    }

    // The backend rate-limits /detect per IP. Surfacing it as "Unexpected error:
    // 429" told the visitor nothing and looked like a crash.
    if (response.status === 429) {
      const wait = response.headers.get("Retry-After");
      statusLine(
        `Too many checks in a row. Try again in ${wait || "a few"} second${wait === "1" ? "" : "s"}.`,
        "text-warning");
      return;
    }

    // New backend: 200 + rich body. Legacy backend: 403 on a detected jailbreak.
    if (response.status === 403) {
      renderResult({ verdict: "malicious", detected: true, risk_score: 0.7, scanners: [] });
      return;
    }
    if (!response.ok) {
      statusLine("Unexpected error: " + response.status, "text-muted");
      return;
    }
    renderResult(await response.json());
  } catch (error) {
    statusLine("Network error: " + error, "text-danger");
  } finally {
    clearTimeout(slow);
    inFlight = false;
    if (button) { button.disabled = false; button.textContent = "Check Prompt"; }
  }
}

// Example chips: fill the textarea and run.
document.addEventListener("click", (e) => {
  const btn = e.target.closest(".example");
  if (!btn) return;
  document.getElementById("promptInput").value = btn.dataset.text;
  checkPrompt();
});
