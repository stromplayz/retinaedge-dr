/* RetinaEdge-DR playground — in-browser inference via ONNX Runtime Web.
 *
 * Preprocessing parity with the repo's val pipeline (dataset.build_train_val_transforms):
 *   LongestMaxSize(resize_longest) -> CenterCrop(h, w, pad_if_needed) -> /255 -> ImageNet normalize
 * The exported ONNX graph embeds temperature scaling + softmax, so the output is
 * directly a (1,5) probability vector over ICDRSS grades 0-4.
 */
"use strict";

const GRADE_COLORS = ["#34d399", "#a3e635", "#fbbf24", "#fb923c", "#f87171"];
const GRADE_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"];

const els = {
  tabs: document.querySelectorAll(".tab[data-tab]"),
  dropzone: document.getElementById("dropzone"),
  fileInput: document.getElementById("file-input"),
  preview: document.getElementById("preview"),
  dropHint: document.getElementById("drop-hint"),
  preprocNote: document.getElementById("preproc-note"),
  modelList: document.getElementById("model-list"),
  runBtn: document.getElementById("run-btn"),
  clearBtn: document.getElementById("clear-btn"),
  status: document.getElementById("status"),
  results: document.getElementById("results"),
  registryBody: document.querySelector("#registry-table tbody"),
  ladder: document.getElementById("ladder"),
  archBest: document.getElementById("arch-best"),
};

let MANIFEST = null;
let imageBitmap = null; // decoded image, kept for re-runs
const sessions = new Map(); // model.id -> Promise<ort.InferenceSession>

/* ------------------------------------------------------------------ tabs */
els.tabs.forEach((tab) =>
  tab.addEventListener("click", () => {
    els.tabs.forEach((t) => {
      t.classList.toggle("is-active", t === tab);
      t.setAttribute("aria-selected", t === tab ? "true" : "false");
    });
    document.querySelectorAll(".panel").forEach((p) => {
      const active = p.id === `panel-${tab.dataset.tab}`;
      p.classList.toggle("is-active", active);
      p.hidden = !active;
    });
  })
);

/* ------------------------------------------------------------------ utils */
const setStatus = (msg) => { els.status.textContent = msg || ""; };
const fmtPct = (x) => `${(x * 100).toFixed(1)}%`;

function softmaxDefensive(arr) {
  const sum = arr.reduce((a, b) => a + b, 0);
  if (sum > 0.98 && sum < 1.02) return Array.from(arr);
  const max = Math.max(...arr);
  const exp = Array.from(arr, (v) => Math.exp(v - max));
  const s = exp.reduce((a, b) => a + b, 0);
  return exp.map((v) => v / s);
}

/* ------------------------------------------------------- manifest + cards */
async function loadManifest() {
  const res = await fetch("models/manifest.json", { cache: "no-store" });
  if (!res.ok) throw new Error(`manifest fetch failed: HTTP ${res.status}`);
  MANIFEST = await res.json();
  renderModelCards();
  renderRegistry();
  renderLadder();
}

function renderModelCards() {
  els.modelList.innerHTML = "";
  MANIFEST.models.forEach((m) => {
    const item = document.createElement("label");
    item.className = "model-item";
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.value = m.id;
    cb.checked = !!m.recommended;
    if (cb.checked) item.classList.add("is-checked");
    cb.addEventListener("change", () => {
      item.classList.toggle("is-checked", cb.checked);
      updateRunBtn();
    });
    const txt = document.createElement("span");
    txt.style.flex = "1";
    const name = document.createElement("div");
    name.className = "mi-name";
    name.textContent = m.label;
    const meta = document.createElement("div");
    meta.className = "mi-meta";
    const camp = m.metrics && m.metrics.campaign_best;
    meta.textContent =
      `${m.size_mb} MB · ${m.input.h}px · QWK ${m.metrics.val_qwk}` +
      (camp ? ` · campaign acc ${camp.acc_refer.toFixed(3)}` : "");
    txt.append(name, meta);
    item.append(cb, txt);
    if (m.recommended) {
      const b = document.createElement("span");
      b.className = "badge rec";
      b.textContent = "flagship";
      item.append(b);
    }
    if (m.quant === "int8") {
      const b = document.createElement("span");
      b.className = "badge int8";
      b.textContent = "int8";
      item.append(b);
    }
    els.modelList.append(item);
  });
  els.preprocNote.textContent =
    `Preprocessing per model: longest side → ${MANIFEST.models[0]?.input.resize_longest ?? 256}px, ` +
    `center crop ${MANIFEST.models[0]?.input.h ?? 224}px, ImageNet normalize (matches the training val pipeline).`;
  updateRunBtn();
}

function renderRegistry() {
  els.registryBody.innerHTML = "";
  MANIFEST.models.forEach((m) => {
    const tr = document.createElement("tr");
    const camp = m.metrics.campaign_best;
    const cells = [
      m.label,
      m.stage,
      m.quant.toUpperCase(),
      `${m.size_mb} MB`,
      String(m.metrics.val_qwk ?? "—"),
      m.metrics.auc_refer != null ? String(m.metrics.auc_refer) : "—",
      camp ? camp.acc_refer.toFixed(4) : "—",
    ];
    cells.forEach((c, i) => {
      const td = document.createElement("td");
      if ([3, 4, 5, 6].includes(i)) td.className = "num";
      td.textContent = c;
      tr.append(td);
    });
    const td = document.createElement("td");
    if (m.recommended) { const b = document.createElement("span"); b.className = "badge rec"; b.textContent = "flagship"; td.append(b); }
    tr.append(td);
    els.registryBody.append(tr);
  });
}

const LADDER = [
  ["s1-baseline", "budget epochs at the config default resolution"],
  ["s2-longer-ema", "longer schedule + EMA weight averaging"],
  ["s3-sharper", "higher input resolution"],
  ["s4-backbone", "stronger backbone (EfficientNet-Lite0)"],
  ["s5-fusion", "stronger backbone + sharper + EMA (full send)"],
  ["s6-mixup", "v0.3.0 — mixup α=0.2 + label smoothing at escalated resolution"],
  ["s7-distill", "v0.3.0 — self-distillation from the soup teacher (T=3)"],
];

function renderLadder() {
  const bestStage = MANIFEST.models[0]?.metrics?.campaign_best?.stage;
  els.ladder.innerHTML = "";
  LADDER.forEach(([name, desc]) => {
    const li = document.createElement("li");
    if (bestStage && name <= bestStage) li.classList.add("done");
    li.innerHTML = `<span class="l-name">${name}</span><br><span class="l-desc">${desc}</span>`;
    els.ladder.append(li);
  });
  const camp = MANIFEST.models[0]?.metrics?.campaign_best;
  if (camp) {
    els.archBest.textContent =
      `${fmtPct(camp.acc_refer)} referable accuracy (CI95 lo ${fmtPct(camp.ci95_lo)}), QWK ${camp.qwk} ` +
      `at stage ${camp.stage}/${camp.variant}`;
  }
}

/* -------------------------------------------------------- image handling */
function loadImageFile(file) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("could not decode image"));
    img.src = url;
  });
}

async function handleFile(file) {
  if (!file || !file.type.startsWith("image/")) { setStatus("Please choose an image file."); return; }
  try {
    imageBitmap = await loadImageFile(file);
    els.preview.src = imageBitmap.src;
    els.preview.classList.remove("hidden");
    els.dropHint.classList.add("hidden");
    setStatus("");
    updateRunBtn();
  } catch (err) {
    setStatus(err.message);
  }
}

els.dropzone.addEventListener("click", () => els.fileInput.click());
els.dropzone.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); els.fileInput.click(); } });
els.fileInput.addEventListener("change", () => handleFile(els.fileInput.files[0]));
["dragenter", "dragover"].forEach((ev) =>
  els.dropzone.addEventListener(ev, (e) => { e.preventDefault(); els.dropzone.classList.add("is-drag"); })
);
["dragleave", "drop"].forEach((ev) =>
  els.dropzone.addEventListener(ev, (e) => { e.preventDefault(); els.dropzone.classList.remove("is-drag"); })
);
els.dropzone.addEventListener("drop", (e) => handleFile(e.dataTransfer.files[0]));

els.clearBtn.addEventListener("click", () => {
  imageBitmap = null;
  els.preview.classList.add("hidden");
  els.dropHint.classList.remove("hidden");
  els.fileInput.value = "";
  els.results.innerHTML = "<p class=\"muted\">Upload an image and run one or more models to see grade probabilities, the expected ICDRSS grade, and the referable-DR alert here.</p>";
  updateRunBtn();
  setStatus("");
});

function selectedModels() {
  return MANIFEST.models.filter((m) => els.modelList.querySelector(`input[value="${m.id}"]`).checked);
}
function updateRunBtn() {
  els.runBtn.disabled = !(imageBitmap && selectedModels().length);
}

/* ---------------------------------------------------------- preprocessing */
function preprocessToTensor(img, input) {
  const longest = input.resize_longest;
  const crop = input.h;
  const scale = longest / Math.max(img.naturalWidth, img.naturalHeight);
  const sw = Math.max(1, Math.round(img.naturalWidth * scale));
  const sh = Math.max(1, Math.round(img.naturalHeight * scale));

  const stage = document.createElement("canvas");
  stage.width = sw; stage.height = sh;
  stage.getContext("2d").drawImage(img, 0, 0, sw, sh);

  // center crop with black padding when the short side < crop (pad_if_needed)
  const out = document.createElement("canvas");
  out.width = crop; out.height = crop;
  const ctx = out.getContext("2d", { willReadFrequently: true });
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, crop, crop);
  const sx = Math.max(0, (sw - crop) / 2), sy = Math.max(0, (sh - crop) / 2);
  const dw = Math.min(sw, crop), dh = Math.min(sh, crop);
  ctx.drawImage(stage, sx, sy, dw, dh, (crop - dw) / 2, (crop - dh) / 2, dw, dh);

  const data = ctx.getImageData(0, 0, crop, crop).data; // RGBA
  const mean = input.mean, std = input.std;
  const plane = crop * crop;
  const tensor = new Float32Array(3 * plane);
  for (let i = 0; i < plane; i++) {
    tensor[i]               = (data[i * 4]     / 255 - mean[0]) / std[0];
    tensor[i + plane]       = (data[i * 4 + 1] / 255 - mean[1]) / std[1];
    tensor[i + 2 * plane]   = (data[i * 4 + 2] / 255 - mean[2]) / std[2];
  }
  return new ort.Tensor("float32", tensor, [1, 3, crop, crop]);
}

/* -------------------------------------------------------------- sessions */
function getSession(model) {
  if (!sessions.has(model.id)) {
    const p = ort.InferenceSession.create(model.file, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
    sessions.set(model.id, p);
  }
  return sessions.get(model.id);
}

/* ------------------------------------------------------------------- run */
async function runOnce(model, tensor) {
  const t0 = performance.now();
  const session = await getSession(model);
  const t1 = performance.now();
  const feeds = { [session.inputNames[0]]: tensor };
  const out = await session.run(feeds);
  const t2 = performance.now();
  const raw = out[session.outputNames[0]].data;
  const probs = softmaxDefensive(Array.from(raw.slice(0, 5)));
  return { probs, loadMs: t1 - t0, inferMs: t2 - t1 };
}

function resultCard(model, r) {
  const probs = r.probs;
  const expected = probs.reduce((a, p, i) => a + p * i, 0);
  const referable = probs.slice(2).reduce((a, b) => a + b, 0);
  const argmax = probs.indexOf(Math.max(...probs));
  const isRefer = referable > 0.5 || argmax >= 2;

  const card = document.createElement("div");
  card.className = "result-card";

  const head = document.createElement("div");
  head.className = "result-head";
  head.innerHTML =
    `<span class="r-name">${model.label}</span>` +
    `<span class="r-meta">load ${r.loadMs.toFixed(0)} ms · infer ${r.inferMs.toFixed(1)} ms · E[Y] ${expected.toFixed(2)}</span>`;
  card.append(head);

  const verdict = document.createElement("span");
  verdict.className = `verdict ${isRefer ? "referable" : "nodr"}`;
  verdict.textContent = isRefer
    ? `⚠ Referable DR suspected — p(refer) ${fmtPct(referable)}`
    : `No referable DR — p(refer) ${fmtPct(referable)}`;
  card.append(verdict);

  const bars = document.createElement("div");
  bars.className = "bars";
  probs.forEach((p, i) => {
    const row = document.createElement("div");
    row.className = `bar-row${i === argmax ? " top" : ""}`;
    row.innerHTML =
      `<span class="bar-label">${i} · ${GRADE_NAMES[i]}</span>` +
      `<span class="bar-track"><span class="bar-fill" style="background:${GRADE_COLORS[i]}"></span></span>` +
      `<span class="bar-val">${(p * 100).toFixed(1)}%</span>`;
    bars.append(row);
    requestAnimationFrame(() => requestAnimationFrame(() => {
      row.querySelector(".bar-fill").style.width = `${Math.max(1.5, p * 100)}%`;
    }));
  });
  card.append(bars);

  const kpis = document.createElement("div");
  kpis.className = "kpis";
  kpis.innerHTML =
    `<div class="kpi"><div class="k">Predicted grade</div><div class="v" style="color:${GRADE_COLORS[argmax]}">${argmax} · ${GRADE_NAMES[argmax]}</div></div>` +
    `<div class="kpi"><div class="k">Expected grade E[Y]</div><div class="v">${expected.toFixed(2)}</div></div>` +
    `<div class="kpi"><div class="k">Model size</div><div class="v">${model.size_mb} MB</div></div>`;
  card.append(kpis);
  return card;
}

els.runBtn.addEventListener("click", async () => {
  const models = selectedModels();
  if (!imageBitmap || !models.length) return;
  els.runBtn.disabled = true;
  els.results.innerHTML = "";
  let ok = 0;
  for (const m of models) {
    setStatus(`Running ${m.label} …`);
    try {
      const tensor = preprocessToTensor(imageBitmap, m.input);
      const r = await runOnce(m, tensor);
      els.results.append(resultCard(m, r));
      ok += 1;
    } catch (err) {
      const div = document.createElement("div");
      div.className = "result-card";
      div.innerHTML = `<span class="r-name">${m.label}</span><p class="muted">failed: ${err.message}</p>`;
      els.results.append(div);
    }
  }
  setStatus(`${ok}/${models.length} model${models.length > 1 ? "s" : ""} done · everything ran locally in your browser.`);
  els.runBtn.disabled = false;
});

/* ------------------------------------------------------------------ boot */
loadManifest()
  .then(() => setStatus(`${MANIFEST.models.length} models loaded from the registry.`))
  .catch((err) => setStatus(`Registry failed to load: ${err.message}`));
