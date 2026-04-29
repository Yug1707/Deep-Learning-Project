const state = {
  genres: [],
  prediction: null,
  feedbackMode: "correct",
};

const statusText = document.querySelector("#statusText");
const bufferMeta = document.querySelector("#bufferMeta");
const predictionMeta = document.querySelector("#predictionMeta");
const predictionList = document.querySelector("#predictionList");
const predictForm = document.querySelector("#predictForm");
const predictButton = document.querySelector("#predictButton");
const thresholdInput = document.querySelector("#thresholdInput");
const topKInput = document.querySelector("#topKInput");
const feedbackForm = document.querySelector("#feedbackForm");
const genrePicker = document.querySelector("#genrePicker");
const submitFeedbackButton = document.querySelector("#submitFeedbackButton");
const feedbackMessage = document.querySelector("#feedbackMessage");
const correctButton = document.querySelector("#correctButton");
const wrongButton = document.querySelector("#wrongButton");
const reloadButton = document.querySelector("#reloadButton");

function pct(value) {
  return `${Math.round(value * 1000) / 10}%`;
}

function setStatus(text, mode = "") {
  statusText.textContent = text;
  statusText.className = mode;
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const payload = await response.json();
      detail = payload.detail || detail;
    } catch {
      // Keep default detail.
    }
    throw new Error(detail);
  }
  return response.json();
}

function renderGenres() {
  genrePicker.innerHTML = "";
  for (const genre of state.genres) {
    const label = document.createElement("label");
    label.className = "genre-option";
    label.innerHTML = `
      <input type="checkbox" value="${genre.genre_id}" />
      <span>${genre.name}</span>
    `;
    genrePicker.appendChild(label);
  }
}

function renderPredictions(prediction) {
  predictionList.className = "prediction-list";
  predictionList.innerHTML = "";
  predictionMeta.textContent = `${prediction.num_chunks} chunk(s), threshold ${prediction.threshold}`;

  const rows = prediction.top_k.length ? prediction.top_k : prediction.predicted_genres;
  if (!rows.length) {
    predictionList.className = "prediction-list empty-state";
    predictionList.textContent = "No genre crossed the threshold.";
    return;
  }

  for (const item of rows) {
    const row = document.createElement("div");
    row.className = "prediction-row";
    row.innerHTML = `
      <div class="genre-name">${item.name}</div>
      <div class="bar-track"><div class="bar-fill" style="width:${pct(item.probability)}"></div></div>
      <div class="score">${pct(item.probability)}</div>
    `;
    predictionList.appendChild(row);
  }
}

function setFeedbackEnabled(enabled) {
  feedbackForm.classList.toggle("is-disabled", !enabled);
  submitFeedbackButton.disabled = !enabled;
}

function setFeedbackMode(mode) {
  state.feedbackMode = mode;
  correctButton.classList.toggle("is-active", mode === "correct");
  wrongButton.classList.toggle("is-active", mode === "wrong");
  genrePicker.classList.toggle("is-hidden", mode !== "wrong");
}

async function refreshStatus() {
  try {
    const status = await api("/api/status");
    bufferMeta.textContent = `${status.feedback_buffer_size}/${status.feedback_trigger_size}`;
    if (!status.checkpoint_exists) {
      setStatus(`Missing AST checkpoint: ${status.active_checkpoint_path}`, "status-error");
    } else if (status.trainer.state === "training") {
      setStatus(`Training ${status.trainer.current_run_id}`, "status-training");
    } else if (status.model_loaded) {
      setStatus(`AST model loaded from ${status.active_checkpoint_path}`);
    } else {
      setStatus(`AST checkpoint ready: ${status.active_checkpoint_path}`);
    }
  } catch (error) {
    setStatus(error.message, "status-error");
  }
}

async function loadGenres() {
  state.genres = await api("/api/genres");
  renderGenres();
}

predictForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  feedbackMessage.textContent = "";
  const fileInput = document.querySelector("#audioFile");
  if (!fileInput.files.length) return;

  const formData = new FormData();
  formData.append("file", fileInput.files[0]);
  if (thresholdInput.value !== "") {
    formData.append("threshold", thresholdInput.value);
  }
  formData.append("top_k", topKInput.value || "5");

  predictButton.disabled = true;
  predictButton.textContent = "Predicting...";
  try {
    const prediction = await api("/api/predict", {
      method: "POST",
      body: formData,
    });
    state.prediction = prediction;
    renderPredictions(prediction);
    setFeedbackEnabled(true);
  } catch (error) {
    predictionList.className = "prediction-list empty-state";
    predictionList.textContent = error.message;
    setFeedbackEnabled(false);
  } finally {
    predictButton.disabled = false;
    predictButton.textContent = "Predict";
    refreshStatus();
  }
});

correctButton.addEventListener("click", () => setFeedbackMode("correct"));
wrongButton.addEventListener("click", () => setFeedbackMode("wrong"));

feedbackForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!state.prediction) return;

  const checked = Array.from(genrePicker.querySelectorAll("input:checked")).map(
    (item) => item.value,
  );
  const payload = {
    prediction_id: state.prediction.prediction_id,
    is_correct: state.feedbackMode === "correct",
    corrected_genre_ids: state.feedbackMode === "wrong" ? checked : null,
    notes: document.querySelector("#notesInput").value,
  };

  submitFeedbackButton.disabled = true;
  try {
    const response = await api("/api/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    feedbackMessage.textContent = response.message;
    refreshStatus();
  } catch (error) {
    feedbackMessage.textContent = error.message;
  } finally {
    submitFeedbackButton.disabled = false;
  }
});

reloadButton.addEventListener("click", async () => {
  reloadButton.disabled = true;
  try {
    await api("/api/reload", { method: "POST" });
    await refreshStatus();
  } catch (error) {
    setStatus(error.message, "status-error");
  } finally {
    reloadButton.disabled = false;
  }
});

setFeedbackEnabled(false);
setFeedbackMode("correct");
loadGenres()
  .then(refreshStatus)
  .catch((error) => setStatus(error.message, "status-error"));

setInterval(refreshStatus, 10000);

