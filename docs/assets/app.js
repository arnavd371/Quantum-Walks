const state = {
  data: null,
  currentGraph: "grid",
  currentTimeIndex: 0,
};

const selectors = {
  graphSelect: document.getElementById("graph-select"),
  timeSlider: document.getElementById("time-slider"),
  timeLabel: document.getElementById("time-label"),
  graphLabel: document.getElementById("graph-label"),
  nodeCount: document.getElementById("node-count"),
  graphDescription: document.getElementById("graph-description"),
  graphParams: document.getElementById("graph-params"),
  peakNode: document.getElementById("peak-node"),
  spreadLabel: document.getElementById("spread-label"),
  legendTime: document.getElementById("legend-time"),
  resetButton: document.getElementById("reset-button"),
  chartArea: document.getElementById("bar-chart"),
};

function formatParams(params = {}) {
  const keys = Object.keys(params);
  if (!keys.length) return "—";
  return keys
    .map((key) => `${key}=${params[key]}`)
    .join(" · ");
}

function computeSpread(probabilities) {
  if (!probabilities.length) return 0;
  const mean = probabilities.reduce((acc, p, idx) => acc + p * idx, 0);
  const variance = probabilities.reduce(
    (acc, p, idx) => acc + p * (idx - mean) ** 2,
    0
  );
  return Math.sqrt(variance);
}

function updateMeta(graphKey) {
  const entry = state.data[graphKey];
  selectors.graphLabel.textContent = entry.label;
  selectors.nodeCount.textContent = entry.nodes;
  selectors.graphDescription.textContent = entry.description;
  selectors.graphParams.textContent = formatParams(entry.parameters);
}

function updateLegend(probabilities, timeLabel) {
  const maxProb = Math.max(...probabilities);
  const peakIndex = probabilities.indexOf(maxProb);
  selectors.peakNode.textContent = `Node ${peakIndex} (${maxProb.toFixed(3)})`;
  selectors.spreadLabel.textContent = `Std. dev ≈ ${computeSpread(
    probabilities
  ).toFixed(3)}`;
  selectors.legendTime.textContent = timeLabel;
}

function renderBars(graphKey) {
  const entry = state.data[graphKey];
  const probs = entry.probabilities[state.currentTimeIndex];
  const timeLabel = `t = ${entry.times[state.currentTimeIndex].toFixed(2)}`;
  const maxProb = Math.max(...probs) || 1;
  selectors.chartArea.innerHTML = "";

  const barWrapper = document.createElement("div");
  barWrapper.className = "bar-chart__bars";

  probs.forEach((prob, idx) => {
    const bar = document.createElement("div");
    bar.className = "bar";
    bar.style.height = `${(prob / maxProb) * 100}%`;
    bar.title = `Node ${idx}: P=${prob.toFixed(4)}`;

    const barLabel = document.createElement("span");
    barLabel.className = "bar__label";
    barLabel.textContent = idx;

    bar.appendChild(barLabel);
    barWrapper.appendChild(bar);
  });

  selectors.chartArea.appendChild(barWrapper);
  selectors.timeLabel.textContent = timeLabel;
  updateLegend(probs, timeLabel);
}

function populateSelector() {
  selectors.graphSelect.innerHTML = "";
  Object.entries(state.data).forEach(([key, entry]) => {
    const option = document.createElement("option");
    option.value = key;
    option.textContent = entry.label;
    selectors.graphSelect.appendChild(option);
  });
  selectors.graphSelect.value = state.currentGraph;
}

function onGraphChange(event) {
  state.currentGraph = event.target.value;
  state.currentTimeIndex = 0;
  const entry = state.data[state.currentGraph];
  selectors.timeSlider.max = entry.times.length - 1;
  selectors.timeSlider.value = 0;
  updateMeta(state.currentGraph);
  renderBars(state.currentGraph);
}

function onTimeChange(event) {
  state.currentTimeIndex = Number(event.target.value);
  renderBars(state.currentGraph);
}

function onReset() {
  state.currentGraph = "grid";
  state.currentTimeIndex = 0;
  selectors.graphSelect.value = state.currentGraph;
  selectors.timeSlider.value = 0;
  onGraphChange({ target: selectors.graphSelect });
}

async function bootstrap() {
  try {
    const response = await fetch("assets/sample-data.json");
    const json = await response.json();
    state.data = json;
    populateSelector();
    onGraphChange({ target: selectors.graphSelect });
    selectors.graphSelect.addEventListener("change", onGraphChange);
    selectors.timeSlider.addEventListener("input", onTimeChange);
    selectors.resetButton.addEventListener("click", onReset);
  } catch (error) {
    console.error("Unable to load sample data:", error);
    selectors.graphDescription.textContent =
      "Failed to load sample data. Check that assets/sample-data.json is present.";
  }
}

bootstrap();
