"use strict";
// Tap mode for auneu.space/timing: the beats a visitor taps (z / x while the song plays) become
// red lines, for the stretches no detector gets right on its own: rubato, fermatas, live drift.
// The visitor decides where the beats are; this only takes their lag out and puts a tap on the
// hit it was aimed at. Plain functions with no page state (app.js holds the taps and the song's
// clear hits), so they run in node as they are.
//
// A tap lands a little off the beat it means: the visitor's own lag (keyboard, ears, habit) plus
// some jitter. The lag is the shift that lines the taps up with the most clear hits at once, the
// jitter the spread around it. A tap then moves onto a clear hit if one is within reach of that
// jitter (stronger hits reach further), unless that bends the beats more than the taps did; a tap
// with no hit stays where it was, pulled halfway toward the middle of its neighbours by at most the
// jitter (less jitter, the rubato kept). Simulated on the mapper's beats in the free-tempo
// stretches of 250 heavy ranked maps (taps = beat + lag + jitter): with the lag known, beats within
// 10 ms of the meant ones go from 80% to 86% at 8 ms of jitter and from 52% to 68% at 15 ms (on a
// clear hit: 91%, 82%). The lag estimate is a few ms off the true one (median) and misses by over
// 15 ms on 7-12% of a visitor's first songs, 6% once their lag on earlier songs is known.

const TAP_LATE_MAX = 150;     // ms of real time a visitor's taps may run late
const TAP_EARLY_MAX = 60;     // or early: taps run ahead of the music by a little at most, and a shift
                              // by a whole subdivision the other way lines up with hits too
const TAP_LAG_SPREAD = 6;     // ms: how far a tap-to-hit gap votes for the shifts around it
const TAP_LAG_PRIOR_SD = 30;  // ms: how far from the visitor's lag on earlier songs this one may be
const TAP_PRIOR_TIE = 0.8;    // a shift near that lag wins if it lines up this well with the hits as the best one
const TAP_LAG_NEAR = 12;      // ms around the best shift: the gaps that make the lag
const TAP_SPREAD_NEAR = 30;   // ms around it: the gaps that make the jitter
const TAP_MIN_GAPS = 6;       // fewer taps near clear hits than this say nothing about the lag
const TAP_SEARCH_MS = 30;     // hits further from a tap than this are never its beat
const TAP_REACH = [0.3, 2];   // a hit of local rank r (0.5..1) is in reach at jitter * (0.3 + 2 r)
const TAP_KINK = 0.5;         // a lock may bend the beats by at most this much jitter more than the taps did
const TAP_RELAX = 0.5;        // a tap with no hit moves this far toward the middle of its neighbours
const TAP_LINE_TOL = 2;       // ms: beats this close to one red line's grid share it
const TAP_TAKEOVER_MS = 40;   // the last tap this close to a beat of the old grid hands over on that beat

const tapMedian = (a) => {
  const s = Array.from(a).sort((x, y) => x - y), n = s.length;
  return n ? (n % 2 ? s[(n - 1) >> 1] : (s[n / 2 - 1] + s[n / 2]) / 2) : NaN;
};

// first index of sorted a with a[i] >= x
function lowerBound(a, x) {
  let lo = 0, hi = a.length;
  while (lo < hi) { const m = (lo + hi) >> 1; if (a[m] < x) lo = m + 1; else hi = m; }
  return lo;
}

// the visitor's lag and jitter (ms of real time: times the playback rate for song ms) from their
// passes [{taps: osu ms, rate}], or null. One lag for all taps: the shift that puts the most
// clear-hit weight under the taps at once (each tap's own nearest hit is often a neighbour of the
// beat it meant, in dense music; strong hits count far more, a beat is rarely on a weak one), then
// the median gap of the taps that sit near a hit at that shift. prior: the visitor's lag on earlier
// songs, if known: a near tie (a shift by a quarter beat lines up nearly as well where every 1/4
// has a hit) goes its way
function tapLag(passes, hits, prior = null) {
  if (!hits || !hits.t.length) return null;
  const n = passes.reduce((s, p) => s + p.taps.length, 0);
  if (n < TAP_MIN_GAPS) return null;
  const lo = -TAP_LATE_MAX, hi = TAP_EARLY_MAX, score = new Float64Array(hi - lo + 1);  // shift = hit - tap
  for (const { taps, rate } of passes) {
    for (const t of taps) {
      // a hit at gap g (song ms) votes for the shift g / rate (real ms), spread over +-TAP_LAG_SPREAD
      for (let j = lowerBound(hits.t, t + (lo - 3 * TAP_LAG_SPREAD) * rate);
        j < hits.t.length && hits.t[j] <= t + (hi + 3 * TAP_LAG_SPREAD) * rate; j++) {
        const g = (hits.t[j] - t) / rate, w = ((hits.r[j] - 0.5) * 2) ** 2;
        for (let s = Math.max(lo, Math.ceil(g - 3 * TAP_LAG_SPREAD)); s <= Math.min(hi, g + 3 * TAP_LAG_SPREAD); s++) {
          score[s - lo] += w * Math.exp(-0.5 * ((g - s) / TAP_LAG_SPREAD) ** 2);
        }
      }
    }
  }
  const argmax = (f) => { let b = 0; for (let i = 1; i < score.length; i++) if (f(i) > f(b)) b = i; return b; };
  let best = argmax((i) => score[i]);
  if (prior !== null) {  // only a near tie: a prior that is wrong (another device, a bad guess) can't win
    const near = argmax((i) => score[i] * Math.exp(-0.5 * ((i + lo - prior) / TAP_LAG_PRIOR_SD) ** 2));
    if (score[near] >= TAP_PRIOR_TIE * score[best]) best = near;
  }
  const shift = best + lo, gaps = [];  // each tap's nearest hit at that shift, real ms off it
  for (const { taps, rate } of passes) {
    for (const t of taps) {
      const j = lowerBound(hits.t, t + shift * rate);
      let d = Infinity;
      if (j < hits.t.length) d = (hits.t[j] - t) / rate - shift;
      if (j > 0 && Math.abs((hits.t[j - 1] - t) / rate - shift) < Math.abs(d)) d = (hits.t[j - 1] - t) / rate - shift;
      if (Math.abs(d) <= TAP_SPREAD_NEAR) gaps.push(d);
    }
  }
  const near = gaps.filter((d) => Math.abs(d) <= TAP_LAG_NEAR);
  if (near.length < TAP_MIN_GAPS) return null;
  const mid = tapMedian(near);
  return { lag: shift + mid, jitter: Math.min(25, Math.max(4, 1.4826 * tapMedian(gaps.map((d) => Math.abs(d - mid))))), n: near.length };
}

// beats from taps (osu ms, sorted): the lag added, taps locked onto clear hits, the rest smoothed.
// hits: {t: sorted osu ms, r: local strength rank 0.5..1} or null. Returns {beats, locked}
function placeTaps(taps, hits, lag, jitter) {
  const t = taps.map((x) => x + lag), n = t.length, out = t.slice();
  const H = hits && hits.t.length ? hits : { t: [], r: [] };
  // each tap's best clear hit in reach; one hit can be one tap's only (the closest keeps it)
  const owner = new Map();
  for (let k = 0; k < n; k++) {
    let best = -1, score = -Infinity;
    for (let j = lowerBound(H.t, t[k] - TAP_SEARCH_MS); j < H.t.length && H.t[j] <= t[k] + TAP_SEARCH_MS; j++) {
      const reach = jitter * (TAP_REACH[0] + TAP_REACH[1] * H.r[j]), d = Math.abs(H.t[j] - t[k]);
      if (d <= reach && -d / reach > score) { score = -d / reach; best = j; }
    }
    if (best < 0) continue;
    const o = owner.get(best);
    if (o === undefined || Math.abs(H.t[best] - t[k]) < Math.abs(H.t[best] - t[o])) owner.set(best, k);
  }
  const locked = new Array(n).fill(false);
  for (const [j, k] of owner) { out[k] = H.t[j]; locked[k] = true; }
  // a lock that bends the beats more than the tap did is no lock
  for (let k = 1; k < n - 1; k++) {
    if (!locked[k]) continue;
    const bentLock = Math.abs(out[k - 1] - 2 * out[k] + out[k + 1]);
    const bentTap = Math.abs(out[k - 1] - 2 * t[k] + out[k + 1]);
    if (bentLock > bentTap + TAP_KINK * jitter) { out[k] = t[k]; locked[k] = false; }
  }
  // by at most the jitter: across a fermata or a sudden change the middle is far off and the tap was right
  const src = out.slice();
  for (let k = 1; k < n - 1; k++) {
    if (locked[k]) continue;
    const move = TAP_RELAX * (0.5 * (src[k - 1] + src[k + 1]) - src[k]);
    out[k] = src[k] + Math.max(-jitter, Math.min(jitter, move));
  }
  return { beats: out, locked };
}

// taps closer together than 40% of the usual gap are one beat (z and x hit at once): the first stays
function dedupeTaps(taps) {
  const s = Array.from(taps).sort((a, b) => a - b);
  if (s.length < 3) return s;
  const gap = tapMedian(s.slice(1).map((x, i) => x - s[i]));
  return s.filter((x, i) => i === 0 || x - s[i - 1] > 0.4 * gap);
}

// as few red lines as keep every beat within TAP_LINE_TOL ms of its line's grid; times in whole
// ms, each line's beat from its own start to the next one's (so no double barlines). The last
// beat is where the next timing takes over: it gets no line here
function tapLines(beats, meter) {
  const b = beats.map(Math.round), out = [];
  let i = 0;
  while (i < b.length - 1) {
    let j = i + 1;
    while (j + 1 < b.length) {
      const L = (b[j + 1] - b[i]) / (j + 1 - i);
      let ok = true;
      for (let k = i + 1; k <= j && ok; k++) ok = Math.abs(b[i] + (k - i) * L - b[k]) <= TAP_LINE_TOL;
      if (!ok) break;
      j++;
    }
    out.push({ time: b[i], beat_length: (b[j] - b[i]) / (j - i), meter });
    i = j;
  }
  return out;
}

// the red lines with the tapped beats in place of the timing from the first beat to the last.
// From the last beat the timing that was in effect there takes over again: on the old grid's own
// beat if the last tap was that close to one, then (a short bar, if need be) on its next "1", so
// the bars after the taps stay as they were; else from the last tap at the old tempo. An old
// line under half a beat before the first tap was the same "1" and goes; a line of its own right
// at the last tap takes over by itself
function spliceTaps(lines, beats) {
  const ls = lines.slice().sort((a, b) => a.time - b.time);
  const b = beats.slice(), n = b.length, first = b[0], lastTap = b[n - 1];
  const inEffect = (t) => { let i = 0; while (i + 1 < ls.length && ls[i + 1].time <= t + 0.5) i++; return ls[i]; };
  const before = ls.length ? inEffect(first) : null, after = ls.length ? inEffect(lastTap) : null;
  const gap = tapMedian(b.slice(1).map((x, i) => x - b[i]));
  const fits = (t) => t > b[n - 2] + 0.25 * gap;  // a handover must not swallow the last tapped beat
  const tail = [];
  let end = Math.round(lastTap);
  const own = ls.find((l) => Math.abs(l.time - lastTap) <= TAP_TAKEOVER_MS && l.time > first + 1 && fits(l.time));
  if (own) {
    end = own.time;  // the taps ran up to a red line of its own: it takes over
  } else if (after) {
    const L = after.beat_length, m = after.meter, k = Math.round((lastTap - after.time) / L), beat = after.time + k * L;
    if (Math.abs(beat - lastTap) <= Math.min(TAP_TAKEOVER_MS, 0.15 * L) && fits(beat)) {
      end = Math.round(beat);
      const left = (m - (((k % m) + m) % m)) % m;  // beats from there to the old grid's next "1"
      const one = Math.round(after.time + (k + left) * L);
      const nextOld = ls.find((l) => l.time > end + 0.5);
      if (left === 0) tail.push({ time: end, beat_length: L, meter: m });
      else {
        tail.push({ time: end, beat_length: (one - end) / left, meter: left });  // a short bar up to it
        if (!(nextOld && nextOld.time <= one + 0.5)) tail.push({ time: one, beat_length: L, meter: m });
      }
    } else {
      tail.push({ time: end, beat_length: L, meter: m });
    }
  }
  b[n - 1] = end;
  const cutFrom = first - (before ? 0.5 * before.beat_length : 0);
  const kept = ls.filter((l) => l === own || l.time < cutFrom || l.time > end + 0.5);
  const out = [...kept, ...tapLines(b, before ? before.meter : 4)];
  for (const x of tail) if (!out.some((l) => Math.abs(l.time - x.time) < 1)) out.push(x);
  return out.sort((a, c) => a.time - c.time);
}

if (typeof module !== "undefined") {
  module.exports = { tapLag, placeTaps, dedupeTaps, tapLines, spliceTaps, lowerBound, tapMedian };
}
