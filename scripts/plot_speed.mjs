// Plot a speed_scaling.sh sweep as a self-contained HTML page (no dependencies).
// usage: node scripts/plot_speed.mjs <results dir or runs.csv> [out.html]
import fs from "node:fs";
import path from "node:path";

const arg = process.argv[2];
if (!arg) {
    console.error("usage: node scripts/plot_speed.mjs <results dir or runs.csv> [out.html]");
    process.exit(1);
}
const csvPath = fs.statSync(arg).isDirectory() ? path.join(arg, "runs.csv") : arg;
const outPath = process.argv[3] ?? path.join(path.dirname(csvPath), "speed.html");
const chiplet = Number(process.env.CHIPLET ?? 8);

const [header, ...lines] = fs.readFileSync(csvPath, "utf8").trim().split(/\r?\n/);
const cols = header.split(",");
const runs = lines.map(l => Object.fromEntries(l.split(",").map((v, i) => [cols[i], v])));

const median = v => {
    const s = [...v].sort((a, b) => a - b);
    const m = s.length >> 1;
    return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};
const tests = [...new Set(runs.map(r => r.test))].sort();
const series = tests.map(test => {
    const by = new Map();
    for (const r of runs.filter(r => r.test === test)) {
        const n = Number(r.threads);
        if (!by.has(n)) by.set(n, []);
        by.get(n).push(Number(r.sims_per_s));
    }
    const points = [...by].sort((a, b) => a[0] - b[0]).map(([n, v]) => ({
        n, med: median(v), min: Math.min(...v), max: Math.max(...v), reps: v.length,
    }));
    const base = points.find(p => p.n === 1)?.med ?? points[0].med / points[0].n;
    for (const p of points) {
        p.speedup = p.med / base;
        p.eff = p.speedup / p.n;
    }
    return { test, base, points };
});

let machine = "";
const machinePath = path.join(path.dirname(csvPath), "machine.txt");
if (fs.existsSync(machinePath)) {
    const m = fs.readFileSync(machinePath, "utf8");
    const model = m.match(/Model name:\s*(.+)/)?.[1]?.trim();
    const commit = m.match(/^commit: (.+)$/m)?.[1];
    const head = m.match(/reps: (\d+)\s+ms: (\d+)/);
    machine = [model, head && `${head[1]} reps × ${head[2]} ms per run`, commit && `commit ${commit}`]
        .filter(Boolean).join(" · ");
}

const data = JSON.stringify({ series, chiplet, machine });

const html = `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>nana thread scaling: simulations / second</title>
<style>
.viz-root {
  color-scheme: light;
  --surface-1: #fcfcfb;
  --text-primary: #0b0b0b;
  --text-secondary: #52514e;
  --muted: #898781;
  --grid: #e1e0d9;
  --border: rgba(11,11,11,0.10);
  --series-1: #2a78d6;
  --series-2: #eb6834;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) .viz-root {
    color-scheme: dark;
    --surface-1: #1a1a19;
    --text-primary: #ffffff;
    --text-secondary: #c3c2b7;
    --muted: #898781;
    --grid: #2c2c2a;
    --border: rgba(255,255,255,0.10);
    --series-1: #3987e5;
    --series-2: #d95926;
  }
}
:root[data-theme="dark"] .viz-root {
  color-scheme: dark;
  --surface-1: #1a1a19;
  --text-primary: #ffffff;
  --text-secondary: #c3c2b7;
  --muted: #898781;
  --grid: #2c2c2a;
  --border: rgba(255,255,255,0.10);
  --series-1: #3987e5;
  --series-2: #d95926;
}
body { margin: 0; }
.viz-root { background: var(--surface-1); color: var(--text-primary); min-height: 100vh;
  font: 14px/1.4 system-ui, -apple-system, "Segoe UI", sans-serif; padding: 24px 32px; box-sizing: border-box; }
h1 { font-size: 20px; font-weight: 600; margin: 0 0 4px; }
.sub { color: var(--text-secondary); margin: 0 0 16px; }
.legend { display: flex; gap: 20px; align-items: center; color: var(--text-secondary); margin: 0 0 8px; flex-wrap: wrap; }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.swatch { width: 14px; height: 3px; border-radius: 2px; display: inline-block; }
.dash { width: 16px; border-top: 2px dashed var(--muted); display: inline-block; }
.chart { position: relative; max-width: 960px; }
.chart h2 { font-size: 15px; font-weight: 600; margin: 16px 0 2px; }
.chart p { color: var(--text-secondary); margin: 0 0 4px; font-size: 13px; }
svg { display: block; width: 100%; height: auto; overflow: visible; }
svg text { fill: var(--muted); font-size: 12px; }
svg .label { fill: var(--text-primary); font-weight: 600; }
.tip { position: absolute; pointer-events: none; background: var(--surface-1); border: 1px solid var(--border);
  box-shadow: 0 2px 8px rgba(0,0,0,0.12); border-radius: 6px; padding: 8px 10px; font-size: 13px;
  white-space: nowrap; display: none; color: var(--text-primary); }
.tip .k { color: var(--text-secondary); }
.tip i { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; vertical-align: -1px; }
details { max-width: 960px; margin-top: 20px; color: var(--text-secondary); }
table { border-collapse: collapse; font-variant-numeric: tabular-nums; margin-top: 8px; color: var(--text-primary); }
th, td { padding: 3px 12px; text-align: right; border-bottom: 1px solid var(--grid); }
th { color: var(--text-secondary); font-weight: 600; }
</style>
</head>
<body>
<div class="viz-root">
<h1>nana thread scaling: simulations / second</h1>
<p class="sub" id="sub"></p>
<div class="legend" id="legend"></div>
<div class="chart" id="c-sims"><h2>Simulations / second</h2><p>Median of reps; whiskers span min–max. Dashed: linear scaling from the 1-thread median.</p></div>
<div class="chart" id="c-eff"><h2>Parallel efficiency</h2><p>Speedup over the 1-thread median, divided by threads.</p></div>
<details><summary>Table view</summary><div id="table"></div></details>
</div>
<script>
const { series, chiplet, machine } = ${data};
const colors = ["var(--series-1)", "var(--series-2)"];
const names = series.map(s => "test " + s.test);
document.getElementById("sub").textContent = machine;
const only = { "#sims": "c-sims", "#eff": "c-eff" }[location.hash];
if (only) {
  for (const id of ["c-sims", "c-eff"]) if (id !== only) document.getElementById(id).remove();
  document.querySelector("details").remove();
}
document.getElementById("legend").innerHTML =
  series.map((s, i) => '<span><i class="swatch" style="background:' + colors[i] + '"></i>' + names[i] + '</span>').join("") +
  (document.getElementById("c-sims") ? '<span><i class="dash"></i>linear from 1 thread</span>' : "");

const NS = "http://www.w3.org/2000/svg";
const el = (tag, attrs, parent) => {
  const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(e);
  return e;
};
const maxN = Math.max(...series.flatMap(s => s.points.map(p => p.n)));
const fmtK = v => v === 0 ? "0" : v >= 1e6 ? (v / 1e6).toFixed(2) + "M" : Math.round(v / 1e3) + "k";

function chart(host, { height, yMin, yMax, yTicks, yFmt, value, range, ideal, endLabel }) {
  if (!host) return;
  const W = 960, H = height, m = { l: 64, r: 96, t: 12, b: 36 };
  const svg = el("svg", { viewBox: "0 0 " + W + " " + H, role: "img" }, host);
  const x = n => m.l + (n - 0.5) / maxN * (W - m.l - m.r);
  const y = v => m.t + (1 - (v - yMin) / (yMax - yMin)) * (H - m.t - m.b);
  for (const t of yTicks) {
    el("line", { x1: m.l, x2: W - m.r, y1: y(t), y2: y(t), stroke: "var(--grid)", "stroke-width": 1 }, svg);
    el("text", { x: m.l - 8, y: y(t) + 4, "text-anchor": "end" }, svg).textContent = yFmt(t);
  }
  for (const n of [1, ...Array.from({ length: Math.floor(maxN / 4) }, (_, i) => 4 * (i + 1))]) {
    el("text", { x: x(n), y: H - m.b + 18, "text-anchor": "middle" }, svg).textContent = n;
  }
  el("text", { x: (m.l + W - m.r) / 2, y: H - 2, "text-anchor": "middle" }, svg).textContent = "threads";
  if (chiplet > 0) {
    for (let b = chiplet; b < maxN; b += chiplet) {
      el("line", { x1: x(b + 0.5), x2: x(b + 0.5), y1: m.t, y2: H - m.b, stroke: "var(--muted)",
        "stroke-width": 1, "stroke-dasharray": "1 3" }, svg);
      el("text", { x: x(b + 0.5) + 4, y: m.t + 10 }, svg).textContent = "L3 group " + (b / chiplet + 1);
    }
  }
  if (ideal) {
    series.forEach(s => {
      el("line", { x1: x(1), y1: y(ideal(s, 1)), x2: x(maxN), y2: y(ideal(s, maxN)), stroke: "var(--muted)",
        "stroke-width": 1.5, "stroke-dasharray": "5 4" }, svg);
    });
  }
  const off = [-0.14, 0.14];
  series.forEach((s, i) => {
    if (range) {
      for (const p of s.points) {
        const [lo, hi] = range(p);
        el("line", { x1: x(p.n) + off[i], x2: x(p.n) + off[i], y1: y(lo), y2: y(hi), stroke: colors[i],
          "stroke-width": 1.5, "stroke-opacity": 0.45, "stroke-linecap": "round" }, svg);
      }
    }
    el("path", { d: s.points.map((p, k) => (k ? "L" : "M") + x(p.n) + " " + y(value(p))).join(""), fill: "none",
      stroke: colors[i], "stroke-width": 2, "stroke-linejoin": "round" }, svg);
  });
  const ends = series.map((s, i) => ({ i, p: s.points[s.points.length - 1] }))
    .map(e => ({ ...e, ly: y(value(e.p)) })).sort((a, b) => a.ly - b.ly);
  for (let k = 1; k < ends.length; k++) ends[k].ly = Math.max(ends[k].ly, ends[k - 1].ly + 15);
  for (const e of ends) {
    const t = el("text", { x: x(e.p.n) + 10, y: e.ly + 4, class: "label" }, svg);
    t.textContent = endLabel(e.p);
    el("tspan", { fill: "var(--text-secondary)", "font-weight": 400 }, t).textContent = " " + names[e.i];
  }
  const cross = el("line", { y1: m.t, y2: H - m.b, stroke: "var(--muted)", "stroke-width": 1, visibility: "hidden" }, svg);
  const dots = series.map((s, i) => el("circle", { r: 4.5, fill: colors[i], stroke: "var(--surface-1)",
    "stroke-width": 2, visibility: "hidden" }, svg));
  const tip = document.createElement("div");
  tip.className = "tip";
  host.appendChild(tip);
  const hit = el("rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: "transparent" }, svg);
  hit.addEventListener("pointermove", ev => {
    const r = svg.getBoundingClientRect();
    const sx = (ev.clientX - r.left) * W / r.width;
    const n = Math.min(maxN, Math.max(1, Math.round((sx - m.l) / (W - m.l - m.r) * maxN + 0.5)));
    cross.setAttribute("x1", x(n)); cross.setAttribute("x2", x(n)); cross.setAttribute("visibility", "visible");
    let rows = "<div><b>" + n + " thread" + (n > 1 ? "s" : "") + "</b></div>";
    series.forEach((s, i) => {
      const p = s.points.find(q => q.n === n);
      if (!p) { dots[i].setAttribute("visibility", "hidden"); return; }
      dots[i].setAttribute("cx", x(n)); dots[i].setAttribute("cy", y(value(p))); dots[i].setAttribute("visibility", "visible");
      rows += '<div><i style="background:' + colors[i] + '"></i>' + names[i] + ': <b>' + fmtK(p.med) + '</b> sims/s' +
        ' <span class="k">(' + fmtK(p.min) + '–' + fmtK(p.max) + ')</span> · ' + p.speedup.toFixed(2) + '× · ' +
        Math.round(100 * p.eff) + '%</div>';
    });
    tip.innerHTML = rows;
    tip.style.display = "block";
    const px = x(n) * r.width / W;
    tip.style.left = (px > r.width * 0.6 ? px - tip.offsetWidth - 12 : px + 12) + "px";
    tip.style.top = (svg.offsetTop + 16) + "px";
  });
  hit.addEventListener("pointerleave", () => {
    tip.style.display = "none"; cross.setAttribute("visibility", "hidden");
    dots.forEach(d => d.setAttribute("visibility", "hidden"));
  });
}

const peak = Math.max(...series.flatMap(s => s.points.map(p => Math.max(p.max, s.base * p.n))));
const step = peak > 2e6 ? 500e3 : peak > 1e6 ? 250e3 : 100e3;
const yMax = Math.ceil(peak / step) * step;
chart(document.getElementById("c-sims"), {
  height: 420, yMin: 0, yMax, yTicks: Array.from({ length: yMax / step + 1 }, (_, i) => i * step), yFmt: fmtK,
  value: p => p.med, range: p => [p.min, p.max], ideal: (s, n) => s.base * n,
  endLabel: p => fmtK(p.med) + " (" + p.speedup.toFixed(1) + "×)",
});
const effMin = Math.min(50, Math.floor(Math.min(...series.flatMap(s => s.points.map(p => 100 * p.eff))) / 10) * 10);
chart(document.getElementById("c-eff"), {
  height: 260, yMin: effMin / 100, yMax: 1, yTicks: Array.from({ length: (100 - effMin) / 10 + 1 }, (_, i) => (effMin + 10 * i) / 100),
  yFmt: v => Math.round(100 * v) + "%", value: p => p.eff, endLabel: p => Math.round(100 * p.eff) + "%",
});

let t = "<table><tr><th>threads</th>" + series.map(s => "<th>test " + s.test + " sims/s</th><th>min–max</th><th>speedup</th><th>eff</th>").join("") + "</tr>";
for (const p0 of series[0].points) {
  t += "<tr><td>" + p0.n + "</td>";
  for (const s of series) {
    const p = s.points.find(q => q.n === p0.n);
    t += p ? "<td>" + fmtK(p.med) + "</td><td>" + fmtK(p.min) + "–" + fmtK(p.max) + "</td><td>" + p.speedup.toFixed(2) + "×</td><td>" + Math.round(100 * p.eff) + "%</td>" : "<td></td><td></td><td></td><td></td>";
  }
  t += "</tr>";
}
document.getElementById("table").innerHTML = t + "</table>";
</script>
</body>
</html>
`;
fs.writeFileSync(outPath, html);
console.log("wrote " + outPath);
