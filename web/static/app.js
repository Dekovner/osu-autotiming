"use strict";
// auneu.space/timing: upload a song, get red lines from the server, check and edit them on a waveform.
// Times are osu! ms (the BASS decode osu! plays). The browser decodes the file itself for playback;
// decodeShift lines that decode up with the server's (browser time = BASS time + decodeShift).
// osu! time is not the decode's own clock: a red line sits AUDIO_LAG ms before the attack it times
// (the detector's OSU_OFFSET_MS), so the sound at osu time t is the decode at t + AUDIO_LAG. The
// waveform and playback are shifted by it, so a well-timed line sits on the attack, by eye and ear.

const $ = (id) => document.getElementById(id);
const API = "api/analyze";
const SNAP_COLORS = { 1: "#ecebf1", 2: "#f04b3c", 3: "#b07ad8", 4: "#4598e0", 6: "#d79be8", 8: "#e8c547" };
const LINE_HIT_PX = 7;
const MAGNET_PX = 10;  // the grid pulls a dragged line or the playhead only this close

let actx = null, buffer = null, musicGain = null, clickGain = null;
let duration = 0, decodeShift = 0, AUDIO_LAG = 20.4;
let mip = [];                      // [{step, min, max}] waveform levels, 1 ms .. 1 s per bin
let lines = [], sel = -1;          // red lines {time, beat_length, meter}, sorted by time
let past = [], future = [];
let view = { start: 0, span: 8000 };
let playing = false, playCtx0 = 0, playPos0 = 0, source = null, rate = 1, cursor = 0;
let metroNext = 0, phaseUncertain = false, fileName = "";
let hoverT = null;  // pointer position on the waveform (osu ms), for the readout

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

// grid ticks in [a, b): {t, level} level 0 = bar, 1 = beat, else the divisor of a sub-beat tick
function ticks(a, b, div, src = lines) {
  const out = [];
  for (let i = 0; i < src.length; i++) {
    const l = src[i], end = i + 1 < src.length ? src[i + 1].time : Infinity;
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

// magnet: t moves to the nearest grid tick (current divisor) if one is within MAGNET_PX on screen.
// A dragged line snaps to the grid of the other lines, not its own (which moves with it), and not
// back to the tick it was picked up from: lines sit on the previous grid, so that tick would
// swallow every nudge of a few ms.
function snap(t, e, skip = null, from = null) {
  if (!$("magnet").checked || (e && e.altKey) || !lines.length) return t;
  const r = (MAGNET_PX / $("wave").clientWidth) * view.span;
  const src = skip ? lines.filter((l) => l !== skip) : lines;
  let best = t, bd = r;
  for (const tk of ticks(t - r, t + r, div(), src)) {
    if (from !== null && Math.abs(tk.t - from) < 1) continue;
    if (Math.abs(tk.t - t) <= bd) { bd = Math.abs(tk.t - t); best = tk.t; }
  }
  return best;
}

// how far the next line sits off this line's grid (a double barline in the editor if not 0)
function joint(i) {
  if (i < 0 || i + 1 >= lines.length) return null;
  const a = lines[i], b = lines[i + 1];
  const n = Math.max(1, Math.round((b.time - a.time) / a.beat_length));
  const land = a.time + n * a.beat_length;
  return { n, land, off: b.time - land };
}

// 0.5-15 ms off is a slip that shows as a double barline; further off (a quarter, half a beat)
// is a deliberate phase reset, like the detector's CONNECT_MAX_MS
const accidental = (j) => j && Math.abs(j.off) >= 0.5 && Math.abs(j.off) <= 15;

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

function addLineAt(t, e = null) {
  if (!lines.length) return;
  const base = lines[lineAt(t)];
  let time = Math.round(snap(t, e));  // near the grid: on it (no double barline); else where asked
  if (lines.some((l) => Math.abs(l.time - time) < 1)) time = Math.round(t);
  if (lines.some((l) => Math.abs(l.time - time) < 1)) {
    status("there is already a red line at the playhead: click the waveform where the new one should go");
    return;
  }
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
    const warn = accidental(j) ? `next line ${j.off > 0 ? "+" : ""}${j.off.toFixed(1)} ms off grid` : "";
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
  p.hidden = !accidental(j);
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

// time under the pointer (or of the line being dragged), in a fixed-width slot so nothing reflows
function renderHover() {
  const t = drag && drag.kind === "line" && drag.moved ? drag.line.time : hoverT;
  $("hover").textContent = t === null ? "" : fmtClock(t);
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
    const [mn, mx] = range(tOf(x, w) + AUDIO_LAG, tOf(x + 1, w) + AUDIO_LAG);
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
    if (accidental(j) && j.land > t0 && j.land < t1) {  // where this grid wanted the next line
      const xl = xOf(j.land, w);
      g.strokeStyle = "#e8c547"; g.setLineDash([3, 3]); g.lineWidth = 1;
      g.beginPath(); g.moveTo(xl, 0); g.lineTo(xl, h); g.stroke(); g.setLineDash([]);
    }
  });
  // pointer
  if (hoverT !== null) {
    const xh = Math.round(xOf(hoverT, w)) + 0.5;
    g.strokeStyle = "#4598e0"; g.globalAlpha = 0.6; g.lineWidth = 1;
    g.beginPath(); g.moveTo(xh, 0); g.lineTo(xh, h); g.stroke();
    g.globalAlpha = 1;
  }
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
    const [mn, mx] = range((x / w) * duration + AUDIO_LAG, ((x + 1) / w) * duration + AUDIO_LAG);
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

function setPlayButton(on) {
  const b = $("play");
  b.classList.toggle("on", on);
  b.querySelector(".ico").textContent = on ? "❚❚" : "▶";
  b.querySelector(".lbl").textContent = on ? "pause" : "play";
}

function play() {
  if (!buffer) return;
  actx.resume();
  audioGraph();
  source = actx.createBufferSource();
  source.buffer = buffer;
  source.playbackRate.value = rate;
  source.connect(musicGain);
  const off = (cursor + AUDIO_LAG + decodeShift) / 1000;
  playCtx0 = actx.currentTime + 0.03;
  if (off >= 0) source.start(playCtx0, off);
  else source.start(playCtx0 - off / rate, 0);
  playPos0 = cursor;
  playing = true;
  metroNext = cursor;
  source.onended = () => { if (playing && now() >= duration - 5) pause(); };
  setPlayButton(true);
  loop();
}
function pause() {
  if (!playing) return;
  cursor = Math.min(now(), duration);
  playing = false;
  try { source.stop(); } catch (e) { /* already stopped */ }
  for (const n of clicks) { try { n.stop(); } catch (e) { /* not started or done */ } }
  clicks = [];
  setPlayButton(false);
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

let clicks = [];  // scheduled click nodes, stopped on pause
function click(when, accent) {
  // a clean woodblock-like tick: sine with a quick pitch drop and a fast decay
  const o = actx.createOscillator(), gn = actx.createGain();
  const f = accent ? 1660 : 1180;
  o.frequency.setValueAtTime(f * 1.5, when);
  o.frequency.exponentialRampToValueAtTime(f, when + 0.008);
  gn.gain.setValueAtTime(0.0001, when);
  gn.gain.exponentialRampToValueAtTime(accent ? 1.0 : 0.7, when + 0.001);
  gn.gain.exponentialRampToValueAtTime(0.0001, when + 0.07);
  o.connect(gn).connect(clickGain);
  o.start(when);
  o.stop(when + 0.08);
  clicks.push(o);
  o.onended = () => { clicks = clicks.filter((x) => x !== o); };
}
function scheduleMetronome() {
  if (!$("metro").checked) { metroNext = now(); return; }
  const until = now() + 90 * rate;  // look ahead a little: the rAF loop runs every ~16 ms
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

// busy: the server is still on it (queue or analysis), shown with the drifting dots
function status(msg, err = false, busy = false) {
  const p = $("status");
  p.hidden = !msg;
  p.querySelector(".msg").textContent = msg || "";
  p.classList.toggle("err", err);
  p.classList.toggle("busy", busy && !err);
}

// upload, then ask once a second where the song is: in the queue, being analysed, or done
async function analyseOnServer(form, name) {
  const asJson = async (r) => r.json().catch(() => ({ error: r.status === 413 ? TOO_BIG : `server error ${r.status}` }));
  let job = await asJson(await fetch(API, { method: "POST", body: form }));
  const id = job.job;
  while (!job.error && job.status !== "done") {
    if (job.status === "queued") {
      status(job.ahead > 0 ? `in the queue: ${job.ahead} song${job.ahead > 1 ? "s" : ""} ahead of yours`
        : `you're next (${job.running} song${job.running === 1 ? "" : "s"} being analysed)`, false, true);
    } else {
      status(`analysing ${name}… (a few seconds per minute of audio)`, false, true);
    }
    await new Promise((r) => setTimeout(r, 1000));
    job = await asJson(await fetch(`api/job/${id}`));
  }
  return job;
}

const MAX_MB = 30;  // the server's limit (web/api.py MAX_BYTES); nginx turns away anything bigger
const TOO_BIG = `file too big: the limit is ${MAX_MB} MB (about 15 minutes of audio); cut the song or export a smaller mp3`;

async function load(file) {
  if (!file) return;
  if (file.size > MAX_MB * 1024 * 1024) {  // say so at once instead of uploading 100+ MB for a refusal
    $("drop").querySelector(".drop-name").textContent = file.name;
    status(`${file.name} is ${Math.round(file.size / 1048576)} MB; ${TOO_BIG.replace("file too big: ", "")}`, true);
    return;
  }
  fileName = file.name;
  pause();
  status(`uploading ${file.name}…`, false, true);
  $("drop").querySelector(".drop-name").textContent = file.name;
  const form = new FormData();
  form.append("file", file);
  actx = actx || new (window.AudioContext || window.webkitAudioContext)();
  const decoding = file.arrayBuffer().then((ab) => actx.decodeAudioData(ab));
  let res;
  try {
    res = await analyseOnServer(form, file.name);
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
  AUDIO_LAG = res.audio_lag_ms ?? 20.4;
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

// fine BPM steps: buttons, and arrow keys in the field (±0.1, shift ±1, alt ±0.01); rounded to
// 0.001 BPM so repeated steps don't pile up float tails
const stepBpm = (d) => edit((l) => {
  const b = Math.round((bpmOf(l) + d) * 1000) / 1000;
  if (b > 0) l.beat_length = 60000 / b;
});
document.querySelectorAll("[data-bpm]").forEach((b) => (b.onclick = () => stepBpm(+b.dataset.bpm)));
$("f-bpm").addEventListener("keydown", (e) => {
  if (e.key !== "ArrowUp" && e.key !== "ArrowDown") return;
  e.preventDefault();
  stepBpm((e.key === "ArrowUp" ? 1 : -1) * (e.shiftKey ? 1 : e.altKey ? 0.01 : 0.1));
  $("f-bpm").value = fmtBpm(bpmOf(lines[sel]));
});
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
    drag = { kind: "line", line: lines[hit], from: lines[hit].time, dx: e.offsetX - xOf(lines[hit].time, w), x0: e.offsetX, moved: false };
  } else {
    drag = { kind: "pan", x0: e.offsetX, start0: view.start, moved: false };
  }
});
wave.addEventListener("pointermove", (e) => {
  const w = wave.clientWidth;
  hoverT = drag && drag.kind === "line" ? tOf(e.offsetX, w) : snap(tOf(e.offsetX, w), e);  // the hover sticks where a click would
  renderHover();
  if (!drag) draw();
  if (!drag) {
    wave.style.cursor = lines.some((l) => Math.abs(xOf(l.time, w) - e.offsetX) < LINE_HIT_PX) ? "ew-resize" : "crosshair";
    return;
  }
  if (drag.kind === "line") {
    if (Math.abs(e.offsetX - drag.x0) > 3) drag.moved = true;
    if (!drag.moved) return;
    drag.line.time = Math.round(snap(tOf(e.offsetX - drag.dx, w), e, drag.line, drag.from));
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
  if (drag && !drag.moved) {  // a click, not a drag: seek (a click on a red line seeks to it)
    if (drag.kind === "line") { past.pop(); seek(drag.line.time); } else seek(snap(tOf(e.offsetX, wave.clientWidth), e));
  }
  if (drag && drag.kind === "line" && drag.moved) refresh();
  drag = null;
});
wave.addEventListener("pointerleave", () => { if (!drag) { hoverT = null; renderHover(); draw(); } });
wave.addEventListener("dblclick", (e) => addLineAt(tOf(e.offsetX, wave.clientWidth), e));

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
