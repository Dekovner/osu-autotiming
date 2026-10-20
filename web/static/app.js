"use strict";
// auneu.space/timing: upload a song, get red lines from the server, check and edit them on a waveform.
// Times are osu! ms (the BASS decode osu! plays). The browser decodes the file itself for playback;
// decodeShift lines that decode up with the server's (browser time = osu time + decodeShift).

const $ = (id) => document.getElementById(id);
const API = "api/analyze";
const SNAP_COLORS = { 1: "#ecebf1", 2: "#f04b3c", 3: "#b07ad8", 4: "#4598e0", 6: "#d79be8", 8: "#e8c547" };
const LINE_HIT_PX = 7;

let actx = null, buffer = null, musicGain = null, clickGain = null;
let duration = 0, decodeShift = 0;
let mip = [];                      // [{step, min, max}] waveform levels, 1 ms .. 1 s per bin
let lines = [], sel = -1;          // red lines {time, beat_length, meter}, sorted by time
let past = [], future = [];
let view = { start: 0, span: 8000 };
let playing = false, playCtx0 = 0, playPos0 = 0, source = null, rate = 1, cursor = 0;
let metroNext = 0, phaseUncertain = false, fileName = "";

// ---------- time & grid helpers ----------

const fmtClock = (ms) => {
  const neg = ms < 0, a = Math.abs(Math.round(ms));
  const m = Math.floor(a / 60000), s = Math.floor(a / 1000) % 60, r = a % 1000;
  return `${neg ? "-" : ""}${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}:${String(r).padStart(3, "0")}`;
};
const bpmOf = (l) => 60000 / l.beat_length;
const fmtBpm = (b) => String(Math.round(b * 1000) / 1000);
const g12 = (x) => String(Number(x.toPrecision(12)));
const now = () => (playing ? playPos0 + (actx.currentTime - playCtx0) * 1000 * rate : cursor);

function lineAt(t) {
  let i = 0;
  while (i + 1 < lines.length && lines[i + 1].time <= t) i++;
  return i;
}

// grid ticks in [a, b): {t, level} level = 1 (bar), 2 (beat) or the divisor index of a sub-beat tick
function ticks(a, b, div) {
  const out = [];
  for (let i = 0; i < lines.length; i++) {
    const l = lines[i], end = i + 1 < lines.length ? lines[i + 1].time : Infinity;
    const lo = Math.max(a, i === 0 ? -Infinity : l.time), hi = Math.min(b, end);
    if (lo >= hi) continue;
    const step = l.beat_length / div;
    let k = Math.ceil((lo - l.time) / step - 1e-9);
    for (let t = l.time + k * step; t < hi; k++, t = l.time + k * step) {
      const beat = k / div;
      let level;
      if (k % div === 0) level = (Math.round(beat) % l.meter + l.meter) % l.meter === 0 ? 0 : 1;
      else level = div / gcd(k % div + div, div);
      out.push({ t, level, meter: l.meter });
    }
  }
  return out;
}
const gcd = (a, b) => (b ? gcd(b, a % b) : Math.abs(a));

// how far the next line sits off this line's grid (a double barline in the editor if not 0)
function joint(i) {
  if (i < 0 || i + 1 >= lines.length) return null;
  const a = lines[i], b = lines[i + 1];
  const n = Math.max(1, Math.round((b.time - a.time) / a.beat_length));
  const land = a.time + n * a.beat_length;
  return { n, land, off: b.time - land };
}

// ---------- state changes ----------

function commit() {
  past.push(JSON.stringify(lines));
  if (past.length > 300) past.shift();
  future = [];
}
function setLines(next, keepSel = true) {
  const chosen = keepSel && sel >= 0 ? lines[sel] : null;
  lines = next.map((l) => ({ ...l })).sort((x, y) => x.time - y.time);
  sel = chosen ? lines.indexOf(lines.find((l) => l.time === chosen.time && l.beat_length === chosen.beat_length)) : sel;
  if (sel >= lines.length) sel = lines.length - 1;
  refresh();
}
function undo() { if (past.length) { future.push(JSON.stringify(lines)); setLines(JSON.parse(past.pop()), false); } }
function redo() { if (future.length) { past.push(JSON.stringify(lines)); setLines(JSON.parse(future.pop()), false); } }

function edit(fn) {
  if (sel < 0) return;
  commit();
  const l = { ...lines[sel] };
  fn(l);
  l.time = Math.round(l.time);
  const next = lines.slice();
  next[sel] = l;
  lines = next.sort((x, y) => x.time - y.time);
  sel = lines.indexOf(l);
  refresh();
}

function addLineAt(t) {
  if (!lines.length) return;
  const base = lines[lineAt(t)];
  const k = Math.round((t - base.time) / base.beat_length);
  const time = Math.round(base.time + k * base.beat_length);  // on the current grid: no double barline
  if (lines.some((l) => Math.abs(l.time - time) < 1)) return;
  commit();
  const l = { time, beat_length: base.beat_length, meter: base.meter };
  lines = [...lines, l].sort((x, y) => x.time - y.time);
  sel = lines.indexOf(l);
  refresh();
}

function deleteSel() {
  if (sel < 0 || lines.length <= 1) return;
  commit();
  lines = lines.filter((_, i) => i !== sel);
  sel = Math.min(sel, lines.length - 1);
  refresh();
}

// ---------- output ----------

function timingText() {
  return lines.map((l) => `${Math.round(l.time)},${g12(l.beat_length)},${l.meter},1,0,100,1,0`).join("\n") + "\n";
}

function refresh() {
  $("text").value = timingText();
  renderList();
  renderSel();
  $("undo").disabled = !past.length;
  $("redo").disabled = !future.length;
  draw();
}

function renderList() {
  const ol = $("lines");
  ol.innerHTML = "";
  lines.forEach((l, i) => {
    const li = document.createElement("li");
    if (i === sel) li.className = "on";
    const j = joint(i);
    const warn = j && Math.abs(j.off) >= 0.5 ? `next line ${j.off > 0 ? "+" : ""}${j.off.toFixed(1)} ms off grid` : "";
    li.innerHTML = `<span class="n">${i + 1}</span><span class="bpm">${fmtBpm(bpmOf(l))}</span><span class="t">${fmtClock(l.time)}</span><span class="warn">${warn}</span>`;
    li.onclick = () => { select(i); reveal(l.time); };
    ol.appendChild(li);
  });
}

function renderSel() {
  $("sel").hidden = sel < 0;
  if (sel < 0) return;
  const l = lines[sel];
  if (document.activeElement !== $("f-time")) $("f-time").value = Math.round(l.time);
  if (document.activeElement !== $("f-bpm")) $("f-bpm").value = fmtBpm(bpmOf(l));
  if (document.activeElement !== $("f-meter")) $("f-meter").value = l.meter;
  const j = joint(sel), p = $("joint");
  p.hidden = !j || Math.abs(j.off) < 0.5;
  if (!p.hidden) {
    p.innerHTML = `next red line is <b>${j.off > 0 ? "+" : ""}${j.off.toFixed(1)} ms</b> off this grid (a double barline). ` +
      `<button id="fit">fit this BPM</button> <button id="movenext">move next line</button>`;
    $("fit").onclick = () => edit((x) => { x.beat_length = (lines[sel + 1].time - x.time) / j.n; });
    $("movenext").onclick = () => {
      commit();
      lines[sel + 1] = { ...lines[sel + 1], time: Math.round(j.land) };
      refresh();
    };
  }
}

function select(i) { sel = i; renderList(); renderSel(); draw(); }

function reveal(t) {
  if (t < view.start || t > view.start + view.span) view.start = t - view.span * 0.3;
  if (!playing) cursor = t;
  draw();
}

// ---------- waveform ----------

function buildMip(min, max) {
  mip = [{ step: 1, min, max }];
  while (mip[mip.length - 1].min.length > 1 && mip[mip.length - 1].step < 1024) {
    const p = mip[mip.length - 1], n = Math.ceil(p.min.length / 2);
    const mn = new Int8Array(n), mx = new Int8Array(n);
    for (let i = 0; i < n; i++) {
      const a = 2 * i, b = Math.min(a + 1, p.min.length - 1);
      mn[i] = Math.min(p.min[a], p.min[b]);
      mx[i] = Math.max(p.max[a], p.max[b]);
    }
    mip.push({ step: p.step * 2, min: mn, max: mx });
  }
}

function range(t0, t1) {
  const span = Math.max(1, t1 - t0);
  let lv = mip[0];
  for (const m of mip) if (m.step <= span) lv = m;
  const a = Math.max(0, Math.floor(t0 / lv.step)), b = Math.min(lv.min.length, Math.ceil(t1 / lv.step));
  let mn = 0, mx = 0;
  for (let i = a; i < b; i++) { if (lv.min[i] < mn) mn = lv.min[i]; if (lv.max[i] > mx) mx = lv.max[i]; }
  return [mn / 127, mx / 127];
}

function fitCanvas(c) {
  const dpr = window.devicePixelRatio || 1, w = c.clientWidth, h = c.clientHeight;
  if (c.width !== Math.round(w * dpr) || c.height !== Math.round(h * dpr)) { c.width = Math.round(w * dpr); c.height = Math.round(h * dpr); }
  const g = c.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  return [g, w, h];
}

const div = () => +$("div").value;
const xOf = (t, w) => ((t - view.start) / view.span) * w;
const tOf = (x, w) => view.start + (x / w) * view.span;

function draw() {
  if (!mip.length) return;
  const [g, w, h] = fitCanvas($("wave"));
  g.clearRect(0, 0, w, h);
  const t0 = view.start, t1 = view.start + view.span, mid = h / 2;
  // grid under the waveform
  const pxPerBeat = lines.length ? (lines[lineAt(t0)].beat_length / view.span) * w : 0;
  const d = pxPerBeat / div() < 5 ? (pxPerBeat < 6 ? 0 : 1) : div();
  if (d) {
    for (const tk of ticks(t0, t1, d)) {
      const x = Math.round(xOf(tk.t, w)) + 0.5;
      if (tk.level === 0) { g.strokeStyle = "#ecebf1"; g.globalAlpha = 0.55; g.lineWidth = 2; }
      else if (tk.level === 1) { g.strokeStyle = "#ecebf1"; g.globalAlpha = 0.3; g.lineWidth = 1; }
      else { g.strokeStyle = SNAP_COLORS[tk.level] || "#85838f"; g.globalAlpha = 0.35; g.lineWidth = 1; }
      g.beginPath(); g.moveTo(x, tk.level <= 1 ? 0 : h * 0.18); g.lineTo(x, tk.level <= 1 ? h : h * 0.82); g.stroke();
    }
    g.globalAlpha = 1;
  }
  // waveform
  g.fillStyle = "#a9a6b6";
  g.globalAlpha = 0.75;
  for (let x = 0; x < w; x++) {
    const [mn, mx] = range(tOf(x, w), tOf(x + 1, w));
    const y0 = mid - mx * mid * 0.92, y1 = mid - mn * mid * 0.92;
    g.fillRect(x, y0, 1, Math.max(1, y1 - y0));
  }
  g.globalAlpha = 1;
  // red lines
  g.font = "11px 'JetBrains Mono', monospace";
  lines.forEach((l, i) => {
    if (l.time < t0 - 200 || l.time > t1) return;
    const x = Math.round(xOf(l.time, w)) + 0.5;
    g.strokeStyle = "#f04b3c"; g.lineWidth = i === sel ? 3 : 2;
    g.beginPath(); g.moveTo(x, 0); g.lineTo(x, h); g.stroke();
    g.fillStyle = i === sel ? "#f04b3c" : "#f04b3cbb";
    g.beginPath(); g.moveTo(x - 6, 0); g.lineTo(x + 6, 0); g.lineTo(x, 9); g.fill();
    g.fillStyle = "#ecebf1";
    g.fillText(`${fmtBpm(bpmOf(l))} · ${l.meter}/4`, x + 5, 22);
    const j = joint(i);
    if (j && Math.abs(j.off) >= 0.5 && j.land > t0 && j.land < t1) {  // where this grid wanted the next line
      const xl = xOf(j.land, w);
      g.strokeStyle = "#e8c547"; g.setLineDash([3, 3]); g.lineWidth = 1;
      g.beginPath(); g.moveTo(xl, 0); g.lineTo(xl, h); g.stroke(); g.setLineDash([]);
    }
  });
  // playhead
  const xp = Math.round(xOf(now(), w)) + 0.5;
  g.strokeStyle = "#ecebf1"; g.lineWidth = 1;
  g.beginPath(); g.moveTo(xp, 0); g.lineTo(xp, h); g.stroke();
  $("clock").textContent = fmtClock(now());
  drawOverview();
}

function drawOverview() {
  const [g, w, h] = fitCanvas($("overview"));
  g.clearRect(0, 0, w, h);
  g.fillStyle = "#85838f";
  for (let x = 0; x < w; x++) {
    const [mn, mx] = range((x / w) * duration, ((x + 1) / w) * duration);
    g.fillRect(x, h / 2 - mx * h * 0.45, 1, Math.max(1, (mx - mn) * h * 0.45));
  }
  g.fillStyle = "#f04b3c";
  for (const l of lines) g.fillRect(Math.round((l.time / duration) * w), 0, 1, h);
  g.strokeStyle = "#ecebf1"; g.globalAlpha = 0.6;
  g.strokeRect((view.start / duration) * w + 0.5, 0.5, Math.max(2, (view.span / duration) * w), h - 1);
  g.globalAlpha = 1;
  g.fillStyle = "#ecebf1";
  g.fillRect(Math.round((now() / duration) * w), 0, 1, h);
}

// ---------- playback & metronome ----------

function audioGraph() {
  if (musicGain) return;
  musicGain = actx.createGain();
  clickGain = actx.createGain();
  musicGain.connect(actx.destination);
  clickGain.connect(actx.destination);
  applyVolumes();
}
// sliders: music 0-100%, click 0-200% (the click can be pushed above a loud master)
function applyVolumes() {
  const m = +$("vol-music").value / 100, c = +$("vol-click").value / 100;
  if (musicGain) { musicGain.gain.value = m * m; clickGain.gain.value = c * c; }  // squared: closer to how loud it sounds
  try { localStorage.setItem("timing-vol", JSON.stringify([$("vol-music").value, $("vol-click").value])); } catch (e) { /* private mode */ }
}

function play() {
  if (!buffer) return;
  actx.resume();
  audioGraph();
  source = actx.createBufferSource();
  source.buffer = buffer;
  source.playbackRate.value = rate;
  source.connect(musicGain);
  const off = (cursor + decodeShift) / 1000;
  playCtx0 = actx.currentTime + 0.03;
  if (off >= 0) source.start(playCtx0, off);
  else source.start(playCtx0 - off / rate, 0);
  playPos0 = cursor;
  playing = true;
  metroNext = cursor;
  source.onended = () => { if (playing && now() >= duration - 5) pause(); };
  $("play").textContent = "pause";
  loop();
}
function pause() {
  if (!playing) return;
  cursor = Math.min(now(), duration);
  playing = false;
  try { source.stop(); } catch (e) { /* already stopped */ }
  $("play").textContent = "play";
  draw();
}
function seek(t) {
  const was = playing;
  if (was) pause();
  cursor = Math.max(0, Math.min(duration, t));
  if (was) play(); else draw();
}
function loop() {
  if (!playing) return;
  const t = now();
  if (t > view.start + view.span * 0.7 || t < view.start) view.start = t - view.span * 0.3;
  scheduleMetronome();
  draw();
  requestAnimationFrame(loop);
}

let noise = null;
function click(when, accent) {
  if (!noise) {  // 20 ms of white noise for the attack: cuts through a dense mix better than a pure tone
    noise = actx.createBuffer(1, Math.round(actx.sampleRate * 0.02), actx.sampleRate);
    const d = noise.getChannelData(0);
    for (let i = 0; i < d.length; i++) d[i] = (Math.random() * 2 - 1) * (1 - i / d.length);
  }
  const o = actx.createOscillator(), gn = actx.createGain();
  o.type = "square";
  o.frequency.value = accent ? 2000 : 1400;
  gn.gain.setValueAtTime(0.0001, when);
  gn.gain.exponentialRampToValueAtTime(accent ? 0.9 : 0.6, when + 0.001);
  gn.gain.exponentialRampToValueAtTime(0.0001, when + 0.045);
  o.connect(gn).connect(clickGain);
  o.start(when);
  o.stop(when + 0.05);
  const n = actx.createBufferSource(), ng = actx.createGain();
  n.buffer = noise;
  ng.gain.value = accent ? 0.7 : 0.45;
  n.connect(ng).connect(clickGain);
  n.start(when);
}
function scheduleMetronome() {
  if (!$("metro").checked) { metroNext = now(); return; }
  const until = now() + 150 * rate;
  for (const tk of ticks(metroNext, until, 1)) {
    const when = playCtx0 + (tk.t - playPos0) / (1000 * rate);
    if (when >= actx.currentTime) click(when, tk.level === 0);
  }
  metroNext = Math.max(metroNext, until);
}

// ---------- browser decode vs osu! (BASS) decode ----------

function alignDecode(serverEnv) {
  // energy per ms of the browser decode, in the server's scale (dB, 0 = loudest), first 60 s
  const sr = buffer.sampleRate, spm = sr / 1000, n = Math.min(serverEnv.length, Math.floor(buffer.length / spm), 60000);
  const ch = [...Array(buffer.numberOfChannels).keys()].map((c) => buffer.getChannelData(c));
  const e = new Float32Array(n);
  let peak = 1e-12;
  for (let i = 0; i < n; i++) {
    let s = 0;
    const a = Math.floor(i * spm), b = Math.floor((i + 1) * spm);
    for (let k = a; k < b; k++) { let v = 0; for (const c of ch) v += c[k]; v /= ch.length; s += v * v; }
    e[i] = s;
    if (s > peak) peak = s;
  }
  const B = e.map((s) => Math.max(-80, 10 * Math.log10(s / peak + 1e-9)));
  const S = Float32Array.from(serverEnv.subarray(0, n), (v) => (v * 80) / 255 - 80);
  const mean = (x) => x.reduce((p, q) => p + q, 0) / x.length;
  const mb = mean(B), ms = mean(S);
  let best = 0, bestScore = -Infinity;
  for (let lag = -250; lag <= 250; lag++) {
    let s = 0;
    for (let i = 300; i < n - 300; i++) s += (S[i] - ms) * (B[i + lag] - mb);
    if (s > bestScore) { bestScore = s; best = lag; }
  }
  return best;
}

// ---------- loading ----------

function b64(s, Type) {
  const bin = atob(s), out = new Type(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = Type === Int8Array ? (bin.charCodeAt(i) << 24) >> 24 : bin.charCodeAt(i);
  return out;
}

function status(msg, err = false) {
  const p = $("status");
  p.hidden = !msg;
  p.textContent = msg || "";
  p.classList.toggle("err", err);
}

async function load(file) {
  if (!file) return;
  fileName = file.name;
  pause();
  status(`analysing ${file.name}… (a few seconds per minute of audio)`);
  $("drop").querySelector(".drop-name").textContent = file.name;
  const form = new FormData();
  form.append("file", file);
  actx = actx || new (window.AudioContext || window.webkitAudioContext)();
  const decoding = file.arrayBuffer().then((ab) => actx.decodeAudioData(ab));
  let res;
  try {
    const r = await fetch(API, { method: "POST", body: form });
    res = await r.json().catch(() => ({ error: `server error ${r.status}` }));
  } catch (e) {
    res = { error: "can't reach the server" };
  }
  if (res.error) { status(res.error, true); return; }
  try {
    buffer = await decoding;
  } catch (e) {
    status("your browser can't play this file (the timing below still works)", true);
    buffer = null;
  }
  duration = res.duration_ms;
  buildMip(b64(res.wave.min, Int8Array), b64(res.wave.max, Int8Array));
  decodeShift = buffer ? alignDecode(b64(res.env.data, Uint8Array)) : 0;
  phaseUncertain = res.phase_uncertain;
  past = []; future = [];
  lines = res.lines.map((l) => ({ time: Math.round(l.time), beat_length: l.beat_length, meter: l.meter }));
  sel = 0;
  cursor = lines[0].time;
  view = { start: Math.max(0, lines[0].time - 1000), span: Math.min(8000, duration) };
  status(`${lines.length} red line${lines.length > 1 ? "s" : ""} in ${res.seconds.toFixed(1)} s`);
  const notes = [];
  if (phaseUncertain) notes.push("<b>beat phase uncertain</b>: if the grid feels on the offbeat, select the first line and press ½ beat.");
  notes.push("red lines sit on guessed downbeats; drag a line if \"1\" is elsewhere.");
  $("notes").innerHTML = notes.join("<br>");
  $("notes").hidden = false;
  $("editor").hidden = false;
  refresh();
}

// ---------- wiring ----------

$("file").onchange = (e) => load(e.target.files[0]);
const dropZone = $("drop");
["dragenter", "dragover"].forEach((ev) => document.addEventListener(ev, (e) => { e.preventDefault(); dropZone.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => document.addEventListener(ev, (e) => { e.preventDefault(); dropZone.classList.remove("over"); }));
document.addEventListener("drop", (e) => load(e.dataTransfer.files[0]));

$("play").onclick = () => (playing ? pause() : play());
$("rate").onchange = (e) => { rate = +e.target.value; if (playing) { pause(); play(); } };
$("div").onchange = draw;
$("metro").onchange = () => { metroNext = now(); };
try {
  const v = JSON.parse(localStorage.getItem("timing-vol") || "null");
  if (v) { $("vol-music").value = v[0]; $("vol-click").value = v[1]; }
} catch (e) { /* nothing saved */ }
$("vol-music").oninput = applyVolumes;
$("vol-click").oninput = applyVolumes;
$("add").onclick = () => addLineAt(now());
$("del").onclick = deleteSel;
$("undo").onclick = undo;
$("redo").onclick = redo;
$("half").onclick = () => edit((l) => { l.beat_length *= 2; });
$("double").onclick = () => edit((l) => { l.beat_length /= 2; });
$("shift").onclick = () => edit((l) => { l.time += l.beat_length / 2; });
document.querySelectorAll("[data-nudge]").forEach((b) => (b.onclick = () => edit((l) => { l.time += +b.dataset.nudge; })));
$("copy").onclick = async () => {
  try { await navigator.clipboard.writeText($("text").value); } catch (e) { $("text").select(); document.execCommand("copy"); }
  $("copy").textContent = "copied";
  setTimeout(() => ($("copy").textContent = "copy"), 1200);
};
$("again").onclick = () => $("file").click();

const onField = (id, apply) => {
  const el = $(id);
  const go = () => { const v = parseFloat(el.value); if (Number.isFinite(v) && v > 0 || id === "f-time" && Number.isFinite(v)) edit((l) => apply(l, v)); };
  el.addEventListener("keydown", (e) => { if (e.key === "Enter") { go(); el.blur(); } e.stopPropagation(); });
  el.addEventListener("change", go);
};
onField("f-time", (l, v) => { l.time = v; });
onField("f-bpm", (l, v) => { l.beat_length = 60000 / v; });
onField("f-meter", (l, v) => { l.meter = Math.max(1, Math.round(v)); });

// waveform: wheel zooms around the pointer, drag pans, click seeks, a red line can be dragged
const wave = $("wave");
let drag = null;
wave.addEventListener("wheel", (e) => {
  e.preventDefault();
  const w = wave.clientWidth, x = e.offsetX;
  if (e.shiftKey || Math.abs(e.deltaX) > Math.abs(e.deltaY)) {
    view.start += ((e.shiftKey ? e.deltaY : e.deltaX) / w) * view.span;
  } else {
    const t = tOf(x, w), f = Math.exp(e.deltaY * 0.0015);
    view.span = Math.max(150, Math.min(duration + 2000, view.span * f));
    view.start = t - (x / w) * view.span;
  }
  draw();
}, { passive: false });
wave.addEventListener("pointerdown", (e) => {
  const w = wave.clientWidth;
  const hit = lines.findIndex((l) => Math.abs(xOf(l.time, w) - e.offsetX) < LINE_HIT_PX);
  wave.setPointerCapture(e.pointerId);
  if (hit >= 0) {
    select(hit);
    commit();
    drag = { kind: "line", line: lines[hit], dx: e.offsetX - xOf(lines[hit].time, w) };
  } else {
    drag = { kind: "pan", x0: e.offsetX, start0: view.start, moved: false };
  }
});
wave.addEventListener("pointermove", (e) => {
  const w = wave.clientWidth;
  if (!drag) {
    wave.style.cursor = lines.some((l) => Math.abs(xOf(l.time, w) - e.offsetX) < LINE_HIT_PX) ? "ew-resize" : "crosshair";
    return;
  }
  if (drag.kind === "line") {
    drag.line.time = Math.round(tOf(e.offsetX - drag.dx, w));
    lines.sort((a, b) => a.time - b.time);
    sel = lines.indexOf(drag.line);
    $("text").value = timingText();
    renderSel();
    draw();
  } else {
    if (Math.abs(e.offsetX - drag.x0) > 3) drag.moved = true;
    view.start = drag.start0 - ((e.offsetX - drag.x0) / w) * view.span;
    draw();
  }
});
wave.addEventListener("pointerup", (e) => {
  if (drag && drag.kind === "pan" && !drag.moved) seek(tOf(e.offsetX, wave.clientWidth));
  if (drag && drag.kind === "line") refresh();
  drag = null;
});
wave.addEventListener("dblclick", (e) => addLineAt(tOf(e.offsetX, wave.clientWidth)));

const ov = $("overview");
const ovSeek = (e) => { const t = (e.offsetX / ov.clientWidth) * duration; view.start = t - view.span / 2; seek(t); };
ov.addEventListener("pointerdown", (e) => { ov.setPointerCapture(e.pointerId); ovSeek(e); ov.onpointermove = ovSeek; });
ov.addEventListener("pointerup", () => { ov.onpointermove = null; });

document.addEventListener("keydown", (e) => {
  if ($("editor").hidden || e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA" || e.target.tagName === "SELECT") return;
  const k = e.key.toLowerCase();
  if (k === " ") { e.preventDefault(); playing ? pause() : play(); }
  else if ((e.ctrlKey || e.metaKey) && k === "z") { e.preventDefault(); e.shiftKey ? redo() : undo(); }
  else if ((e.ctrlKey || e.metaKey) && k === "y") { e.preventDefault(); redo(); }
  else if (k === "delete" || k === "backspace") deleteSel();
  else if (k === "a") addLineAt(now());
  else if (k === "m") { $("metro").checked = !$("metro").checked; metroNext = now(); }
  else if (k === "arrowleft" || k === "arrowright") {
    e.preventDefault();
    if (!lines.length) return;
    const l = lines[lineAt(now())], step = l.beat_length / div(), dir = k === "arrowright" ? 1 : -1;
    const kk = Math.round((now() - l.time) / step) + dir;
    seek(l.time + kk * step);
    reveal(now());
  } else if (k === "[" || k === "]") {
    const opts = [...$("rate").options].map((o) => +o.value);
    const i = Math.max(0, Math.min(opts.length - 1, opts.indexOf(rate) + (k === "]" ? 1 : -1)));
    $("rate").value = String(opts[i]);
    $("rate").onchange({ target: $("rate") });
  }
});
window.addEventListener("resize", draw);
