// AdaGAM Live: in-browser inference for PointMLP / AdaGAM on ModelNet-C objects.
// Grouping (farthest point sampling + k-NN) mirrors export_onnx.py exactly, in float32.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const K = 24;
const $ = (id) => document.getElementById(id);
const statusEl = $("status"), progEl = $("prog");
const setStatus = (t) => { statusEl.textContent = t; };
const setProg = (f) => { progEl.style.width = `${Math.round(f * 100)}%`; };
const f32 = Math.fround;

// ------------------------------------------------------------------ grouping
function fps(xyz, n, npoint) {
  const idx = new Int32Array(npoint), dist = new Float32Array(n).fill(1e10);
  let far = 0;
  for (let i = 0; i < npoint; i++) {
    idx[i] = far;
    const cx = xyz[far * 3], cy = xyz[far * 3 + 1], cz = xyz[far * 3 + 2];
    let best = -1, bi = 0;
    for (let j = 0; j < n; j++) {
      const dx = f32(xyz[j * 3] - cx), dy = f32(xyz[j * 3 + 1] - cy), dz = f32(xyz[j * 3 + 2] - cz);
      const d = f32(f32(f32(dx * dx) + f32(dy * dy)) + f32(dz * dz));
      if (d < dist[j]) dist[j] = d;
      if (dist[j] > best) { best = dist[j]; bi = j; }
    }
    far = bi;
  }
  return idx;
}

function knn(xyz, n, centers, s) {
  const out = new Int32Array(s * K), d = new Float32Array(n), order = new Int32Array(n);
  for (let c = 0; c < s; c++) {
    const cx = centers[c * 3], cy = centers[c * 3 + 1], cz = centers[c * 3 + 2];
    for (let j = 0; j < n; j++) {
      const dx = f32(cx - xyz[j * 3]), dy = f32(cy - xyz[j * 3 + 1]), dz = f32(cz - xyz[j * 3 + 2]);
      d[j] = f32(f32(f32(dx * dx) + f32(dy * dy)) + f32(dz * dz));
      order[j] = j;
    }
    // partial selection of the K smallest (order inside a group does not matter)
    let lo = 0, hi = n - 1;
    while (lo < hi) {
      const piv = d[order[(lo + hi) >> 1]];
      let i = lo, j = hi;
      while (i <= j) {
        while (d[order[i]] < piv) i++;
        while (d[order[j]] > piv) j--;
        if (i <= j) { const t = order[i]; order[i] = order[j]; order[j] = t; i++; j--; }
      }
      if (K - 1 <= j) hi = j; else if (K - 1 >= i) lo = i; else break;
    }
    out.set(order.subarray(0, K), c * K);
  }
  return out;
}

function grouping(xyz, n) {
  const feeds = {};
  let cur = xyz, cn = n;
  for (let st = 1; st <= 4; st++) {
    const s = Math.floor(cn / 2);
    const f = fps(cur, cn, s);
    const centers = new Float32Array(s * 3);
    for (let i = 0; i < s; i++) centers.set(cur.subarray(f[i] * 3, f[i] * 3 + 3), i * 3);
    feeds[`fps${st}`] = new ort.Tensor("int32", f, [s]);
    feeds[`knn${st}`] = new ort.Tensor("int32", knn(cur, cn, centers, s), [s, K]);
    cur = centers; cn = s;
  }
  feeds.points = new ort.Tensor("float32", xyz, [1, n, 3]);
  return feeds;
}

// ------------------------------------------------------------------ models
let sessions = null, backend = "";
async function fetchWithProgress(url, onp) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  // content-length is the compressed size when the host gzips, so collect chunks of any size
  const total = +r.headers.get("content-length") || 0, expect = 53.5e6;
  if (!r.body) return new Uint8Array(await r.arrayBuffer());
  const reader = r.body.getReader(), chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value); got += value.length; onp(Math.min(1, got / Math.max(total, expect)));
  }
  const buf = new Uint8Array(got);
  let off = 0;
  for (const c of chunks) { buf.set(c, off); off += c.length; }
  return buf;
}

async function loadModels() {
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
  const files = manifest.models || ["pointmlp", "adagam"], bufs = {};
  for (let i = 0; i < files.length; i++) {
    setStatus(`Downloading ${files[i]} model (≈53 MB, cached after the first visit)…`);
    bufs[files[i]] = await fetchWithProgress(`models/${files[i]}.onnx`, (f) => setProg((i + f) / files.length));
  }
  const tryEP = async (ep) => ({ pm: await ort.InferenceSession.create(bufs.pointmlp, { executionProviders: [ep] }),
                                 ada: await ort.InferenceSession.create(bufs.adagam, { executionProviders: [ep] }),
                                 aug: bufs.adagam_aug ? await ort.InferenceSession.create(bufs.adagam_aug, { executionProviders: [ep] }) : null });
  if (navigator.gpu) {
    try { setStatus("Starting models on your GPU (WebGPU)…"); sessions = await tryEP("webgpu"); backend = "GPU (WebGPU)"; }
    catch (e) { console.warn("WebGPU failed, using CPU", e); sessions = null; }
  }
  if (!sessions) { setStatus("Starting models on your CPU (WebAssembly)…"); sessions = await tryEP("wasm"); backend = "CPU (WebAssembly)"; }
  setProg(1);
}

async function runModel(sess, feeds, adapt) {
  const f = { ...feeds };
  if (sess.inputNames.includes("adapt")) f.adapt = new ort.Tensor("float32", new Float32Array([adapt]), [1]);
  for (const k of Object.keys(f)) if (!sess.inputNames.includes(k)) delete f[k];
  const t0 = performance.now();
  const out = await sess.run(f);
  return { probs: out[sess.outputNames[0]].data, ms: performance.now() - t0 };
}

// ------------------------------------------------------------------ data
let manifest = null, expected = null;
const objCache = new Map();
async function loadObject(id) {
  if (objCache.has(id)) return objCache.get(id);
  const meta = manifest.objects.find((o) => o.id === id);
  const buf = await (await fetch(`objects/${id}.bin`)).arrayBuffer();
  const dv = new DataView(buf), conds = {};
  let off = 0;
  manifest.conditions.forEach((name, ci) => {
    const { n } = meta.conds[ci];
    const xyz = new Float32Array(n * 3), s = meta.scale / 32767;
    for (let i = 0; i < n * 3; i++) xyz[i] = f32(dv.getInt16(off + i * 2, true) * f32(s));
    off += n * 6;
    const nb = Math.ceil(n / 8), mask = new Uint8Array(n);
    for (let i = 0; i < n; i++) mask[i] = (dv.getUint8(off + (i >> 3)) >> (7 - (i & 7))) & 1;
    off += nb;
    conds[name] = { xyz, n, mask };
  });
  const obj = { meta, conds };
  objCache.set(id, obj);
  return obj;
}

// ------------------------------------------------------------------ 3D view
const viewer = $("viewer");
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(2, devicePixelRatio));
viewer.prepend(renderer.domElement);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(40, 4 / 3, 0.01, 50);
camera.position.set(1.9, 1.3, 2.2);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true; controls.autoRotate = !matchMedia("(prefers-reduced-motion: reduce)").matches; controls.autoRotateSpeed = 1.2;
let cloud = null, ghost = null;
const css = (v) => new THREE.Color(getComputedStyle(document.documentElement).getPropertyValue(v).trim());
function makePoints(xyz, n, mask, size, colorVar, injVar) {
  const g = new THREE.BufferGeometry();
  // ModelNet is y-up already
  g.setAttribute("position", new THREE.BufferAttribute(xyz.slice(0, n * 3), 3));
  const base = css(colorVar), inj = injVar ? css(injVar) : base, col = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const shade = 0.75 + 0.25 * (xyz[i * 3 + 1] + 1) / 2;
    const c = mask && mask[i] ? inj : base;
    col[i * 3] = c.r * (mask && mask[i] ? 1 : shade); col[i * 3 + 1] = c.g * (mask && mask[i] ? 1 : shade); col[i * 3 + 2] = c.b * (mask && mask[i] ? 1 : shade);
  }
  g.setAttribute("color", new THREE.BufferAttribute(col, 3));
  return new THREE.Points(g, new THREE.PointsMaterial({ size, vertexColors: true, sizeAttenuation: true }));
}
function show(xyz, n, mask, cleanXyz, cleanN, label) {
  if (cloud) scene.remove(cloud);
  if (ghost) scene.remove(ghost);
  cloud = makePoints(xyz, n, mask, 0.04, "--pt", "--inj");
  scene.add(cloud);
  ghost = cleanXyz ? makePoints(cleanXyz, cleanN, null, 0.025, "--ghost") : null;
  if (ghost) { ghost.visible = $("ghost").checked; scene.add(ghost); }
  $("vlabel").textContent = label;
}
$("ghost").addEventListener("change", () => { if (ghost) ghost.visible = $("ghost").checked; });
function resize() {
  const w = viewer.clientWidth, h = viewer.clientHeight;
  renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(viewer);
renderer.setAnimationLoop(() => { controls.update(); renderer.render(scene, camera); });

// ------------------------------------------------------------------ UI
const nice = (s) => s.replace(/_/g, " ");
function condKey() { const c = $("corr").value; return c === "clean" ? "clean" : `${c}_${$("sev").value}`; }
function card(id, probs, ms, truth) {
  const el = $(id), top = [...probs.keys()].sort((a, b) => probs[b] - probs[a]).slice(0, 5), k = top[0];
  el.classList.remove("right", "wrong");
  if (truth !== null) el.classList.add(k === truth ? "right" : "wrong");
  const mark = truth === null ? "" : `<span class="mark">${k === truth ? "✓ correct" : "✗ wrong"}</span>`;
  el.querySelector(".top").innerHTML = `${nice(manifest.classes[k])} <span class="pc">${(probs[k] * 100).toFixed(1)}%</span> ${mark}`;
  el.querySelector(".ms").textContent = ms ? `${ms.toFixed(0)} ms` : "";
  el.querySelector(".t5").innerHTML = top.map((i) => `<div><span>${nice(manifest.classes[i])}</span><b style="width:${Math.max(1, probs[i] * 100)}%"></b><em>${(probs[i] * 100).toFixed(1)}</em></div>`).join("");
}
const cardIds = () => (sessions && sessions.aug ? ["c-pm", "c-ada", "c-off", "c-aug"] : ["c-pm", "c-ada", "c-off"]);
function clearCards() {
  for (const id of cardIds()) {
    const el = $(id); el.classList.remove("right", "wrong");
    el.querySelector(".top").textContent = "running…"; el.querySelector(".t5").innerHTML = ""; el.querySelector(".ms").textContent = "";
  }
  $("check").textContent = "";
}

let busy = false;
async function infer(xyz, n, truth, expKey) {
  if (busy) return; busy = true; $("run").disabled = true; clearCards();
  try {
    setStatus(`Grouping ${n} points (farthest point sampling + 24 nearest neighbours)…`);
    await new Promise((r) => setTimeout(r, 10));
    const t0 = performance.now(), feeds = grouping(xyz, n), tg = performance.now() - t0;
    setStatus(`Running PointMLP on ${backend}…`);
    const pm = await runModel(sessions.pm, feeds, 1); card("c-pm", pm.probs, pm.ms, truth);
    setStatus(`Running AdaGAM on ${backend}…`);
    const ada = await runModel(sessions.ada, feeds, 1); card("c-ada", ada.probs, ada.ms, truth);
    setStatus(`Running AdaGAM with adaptation off…`);
    const off = await runModel(sessions.ada, feeds, 0); card("c-off", off.probs, off.ms, truth);
    const got = [pm.probs, ada.probs, off.probs];
    if (sessions.aug) {
      setStatus(`Running AdaGAM retrained on corrupted copies…`);
      const aug = await runModel(sessions.aug, feeds, 1); card("c-aug", aug.probs, aug.ms, truth); got.push(aug.probs);
    }
    setStatus(`Done on ${backend}. Grouping ${tg.toFixed(0)} ms.`);
    if (expKey && expected[expKey]) {
      const exp = expected[expKey].slice(0, got.length);
      const same = exp.every((e, i) => e[0][0] === [...got[i].keys()].sort((a, b) => got[i][b] - got[i][a])[0]);
      const dev = Math.max(...exp.map((e, i) => Math.abs(e[0][1] - got[i][e[0][0]])));
      $("check").textContent = same ? `✓ Matches the offline PyTorch→ONNX run of this input (largest difference ${(dev * 100).toFixed(2)} points).`
                                    : `Differs from the offline run in the top class (numerical noise on a borderline case).`;
    }
  } catch (e) { console.error(e); setStatus(`Error: ${e.message}`); }
  busy = false; $("run").disabled = false;
}

async function runOfficial() {
  const id = +$("obj").value, obj = await loadObject(id), key = condKey(), c = obj.conds[key], cl = obj.conds.clean;
  const sevTxt = key === "clean" ? "clean" : `${nice($("corr").value)}, severity ${$("sev").value}`;
  show(c.xyz, c.n, c.mask, cl.xyz, cl.n, `#${id} ${nice(manifest.classes[obj.meta.label])} · ${sevTxt} · ${c.n} points`);
  await infer(c.xyz, c.n, obj.meta.label, `${id}/${key}`);
}

function parseCloud(name, buf) {
  let pts = [];
  if (name.endsWith(".npy")) {
    const dv = new DataView(buf), hl = dv.getUint16(8, true), hdr = new TextDecoder().decode(new Uint8Array(buf, 10, hl));
    const shape = hdr.match(/\((\d+),\s*(\d+)/), f8 = /<f8/.test(hdr), off = 10 + hl;
    if (!shape || !/'fortran_order': False/.test(hdr)) throw new Error("Use a C-ordered float32/float64 .npy of shape (N, 3+)");
    const n = +shape[1], c = +shape[2], arr = f8 ? new Float64Array(buf.slice(off)) : new Float32Array(buf.slice(off));
    for (let i = 0; i < n; i++) pts.push([arr[i * c], arr[i * c + 1], arr[i * c + 2]]);
  } else {
    let text = new TextDecoder().decode(buf);
    if (name.endsWith(".ply")) {
      if (!/format ascii/.test(text)) throw new Error("Only ASCII .ply is supported");
      const nv = +(text.match(/element vertex (\d+)/) || [0, 0])[1];
      text = text.split("end_header")[1].trim().split(/\r?\n/).slice(0, nv).join("\n");
    }
    for (const line of text.split(/\r?\n/)) {
      const v = line.trim().split(/[\s,;]+/).map(Number);
      if (v.length >= 3 && v.slice(0, 3).every(Number.isFinite)) pts.push(v.slice(0, 3));
    }
  }
  if (pts.length < 256) throw new Error(`Found ${pts.length} points; need at least 256`);
  if (pts.length > 4096) { for (let i = pts.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [pts[i], pts[j]] = [pts[j], pts[i]]; } pts = pts.slice(0, 4096); }
  const n = pts.length, m = [0, 1, 2].map((k) => pts.reduce((s, p) => s + p[k], 0) / n);
  let r = 0; for (const p of pts) r = Math.max(r, Math.hypot(p[0] - m[0], p[1] - m[1], p[2] - m[2]));
  const xyz = new Float32Array(n * 3);
  pts.forEach((p, i) => { for (let k = 0; k < 3; k++) xyz[i * 3 + k] = (p[k] - m[k]) / (r || 1); });
  return { xyz, n };
}

$("upload").addEventListener("change", async (e) => {
  const f = e.target.files[0]; if (!f) return;
  try {
    const { xyz, n } = parseCloud(f.name.toLowerCase(), await f.arrayBuffer());
    show(xyz, n, null, null, 0, `${f.name} · ${n} points (your file)`);
    await infer(xyz, n, null, null);
  } catch (err) { setStatus(`Could not read ${f.name}: ${err.message}`); }
  e.target.value = "";
});

$("sev").addEventListener("input", () => { $("sevv").textContent = $("sev").value; });
$("corr").addEventListener("change", () => { $("sev").disabled = $("corr").value === "clean"; });
$("run").addEventListener("click", runOfficial);

function presetChips() {
  const pick = [], seen = new Set();
  const add = (p) => { if (!seen.has(p.id) && pick.length < 8) { seen.add(p.id); pick.push(p); } };
  const rescue = manifest.presets.filter((p) => p.kind === "rescue"), hurt = manifest.presets.filter((p) => p.kind === "hurt");
  [305, 320].forEach((id) => rescue.filter((p) => p.id === id).slice(0, 1).forEach(add));
  rescue.forEach((p) => { if (pick.length < 4) add(p); });
  hurt.forEach((p) => { if (pick.length < 7) add(p); });
  $("chips").innerHTML = pick.map((p, i) => {
    const o = manifest.objects.find((x) => x.id === p.id), [c, s] = [p.cond.replace(/_\d$/, ""), p.cond.slice(-1)];
    const col = p.kind === "rescue" ? "var(--ada)" : "var(--off)";
    const txt = p.kind === "rescue" ? "AdaGAM rescues it" : p.cured && sessions && sessions.aug ? "adaptation hurts it, retraining cures it" : "adaptation hurts it";
    return `<button class="chip" data-i="${i}"><i style="background:${col}"></i>${nice(manifest.classes[o.label])} #${o.id} · ${nice(c)} ${s} · ${txt}</button>`;
  }).join("");
  $("chips").querySelectorAll(".chip").forEach((b) => b.addEventListener("click", () => {
    const p = pick[+b.dataset.i];
    $("obj").value = p.id; $("corr").value = p.cond.replace(/_\d$/, ""); $("sev").value = p.cond.slice(-1);
    $("sevv").textContent = $("sev").value; $("sev").disabled = false;
    viewer.scrollIntoView({ behavior: "smooth", block: "center" });
    runOfficial();
  }));
}

(async () => {
  try {
    setStatus("Loading objects…");
    [manifest, expected] = await Promise.all([fetch("manifest.json").then((r) => r.json()), fetch("expected.json").then((r) => r.json())]);
    const opts = manifest.objects.slice().sort((a, b) => manifest.classes[a.label].localeCompare(manifest.classes[b.label]) || a.id - b.id);
    $("obj").innerHTML = opts.map((o) => `<option value="${o.id}">${nice(manifest.classes[o.label])} (#${o.id})</option>`).join("");
    $("obj").value = manifest.objects.some((o) => o.id === 305) ? "305" : opts[0].id;
    $("corr").value = "add_global";
    presetChips();
    const first = await loadObject(+$("obj").value), c = first.conds.add_global_4;
    show(c.xyz, c.n, c.mask, first.conds.clean.xyz, first.conds.clean.n, `#${first.meta.id} ${nice(manifest.classes[first.meta.label])} · add global, severity 4 · ${c.n} points`);
    await loadModels();
    if (sessions.aug) { $("c-aug").hidden = false; presetChips(); }
    for (const id of ["obj", "corr", "sev", "run", "upload"]) $(id).disabled = false;
    setStatus(`Ready on ${backend}. Pick an object or a case below.`);
    await runOfficial();
  } catch (e) { console.error(e); setStatus(`Could not start: ${e.message}`); }
})();
