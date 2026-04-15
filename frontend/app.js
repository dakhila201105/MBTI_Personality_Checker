/**
 * Calls POST /predict and renders emotion, confidence, and suggestions.
 * Expects the UI to be served from the same origin as the FastAPI app.
 */
const $ = (id) => document.getElementById(id);

function emotionClass(name) {
  const n = (name || "").toLowerCase();
  if (n === "stress") return "emotion-stress";
  if (n === "anxiety") return "emotion-anxiety";
  if (n === "depression") return "emotion-depression";
  return "emotion-neutral";
}

async function analyze() {
  const text = $("input-text").value.trim();
  const status = $("status");
  const results = $("results");

  if (!text) {
    status.textContent = "Please enter some text.";
    results.classList.add("hidden");
    return;
  }

  status.textContent = "Analyzing…";
  results.classList.add("hidden");

  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || res.statusText);
    }

    const data = await res.json();
    $("out-emotion").textContent = data.emotion;
    $("out-emotion").className = emotionClass(data.emotion);
    $("out-confidence").textContent = `${(data.confidence * 100).toFixed(1)}%`;
    $("out-suggestion").textContent = data.suggestion;

    const tipsEl = $("out-tips");
    tipsEl.innerHTML = "";
    if (data.extra_tips && data.extra_tips.length) {
      const ul = document.createElement("ul");
      data.extra_tips.forEach((t) => {
        const li = document.createElement("li");
        li.textContent = t;
        ul.appendChild(li);
      });
      tipsEl.appendChild(document.createTextNode("Extra ideas:"));
      tipsEl.appendChild(ul);
    }

    status.textContent = `Model: ${data.model_used || "baseline"}`;
    results.classList.remove("hidden");
  } catch (e) {
    console.error(e);
    status.textContent = "Error: " + (e.message || "request failed");
    results.classList.add("hidden");
  }
}

document.addEventListener("DOMContentLoaded", () => {
  $("analyze-btn").addEventListener("click", analyze);
});
