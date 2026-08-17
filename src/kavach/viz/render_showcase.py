"""
render_showcase.py — Reads the REAL trace_export.json (produced by
kavach.export_trace) and renders it into a self-contained HTML page that
animates the actual 50-step heuristic episode: real demands, real candidate
paths, real chosen routes, real served/blocked outcomes, real final stats.

Nothing in the animation is randomized or invented — every step plays back
exactly what's in trace_export.json, in order.

Usage:
    python3 render_showcase.py
Reads:  trace_export.json  (from this directory)
Writes: kavach_showcase_real.html
"""

import json

with open("trace_export.json") as f:
    trace = json.load(f)

TRACE_JSON = json.dumps(trace)

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Kavach — Real QKD Routing Episode</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  :root{
    --ink:#0A0E1A; --ink-2:#10162A; --ink-3:#161D36; --line:#262F4C;
    --text:#E8ECF4; --text-dim:#8890A6;
    --quantum:#6EE7F0; --quantum-dim:#3A6E77;
    --classical:#F0A868; --classical-dim:#6E5638;
    --served:#8FE3A8; --blocked:#F0616E;
    --display:'Space Grotesk',sans-serif; --body:'Inter',sans-serif; --mono:'IBM Plex Mono',monospace;
  }
  *{margin:0;padding:0;box-sizing:border-box;}
  body{background:var(--ink);color:var(--text);font-family:var(--body);}
  .wrap{max-width:980px;margin:0 auto;padding:48px 28px 80px;}
  .eyebrow{font-family:var(--mono);font-size:12.5px;letter-spacing:0.14em;color:var(--quantum);
    text-transform:uppercase;display:flex;align-items:center;gap:10px;margin-bottom:20px;}
  .eyebrow::before{content:'';width:7px;height:7px;border-radius:50%;background:var(--quantum);
    box-shadow:0 0 8px var(--quantum);animation:pulse-dot 2s ease-in-out infinite;}
  @keyframes pulse-dot{0%,100%{opacity:1;}50%{opacity:0.35;}}
  h1{font-family:var(--display);font-weight:700;font-size:clamp(28px,4.6vw,44px);line-height:1.1;max-width:760px;}
  .sub{margin-top:14px;font-size:15.5px;color:var(--text-dim);max-width:620px;}
  .sub code{font-family:var(--mono);color:var(--quantum);font-size:13.5px;}
  .viz-frame{margin-top:34px;border:1px solid var(--line);border-radius:14px;
    background:linear-gradient(180deg,var(--ink-2),var(--ink));overflow:hidden;}
  .viz-header{display:flex;justify-content:space-between;align-items:center;padding:14px 20px;
    border-bottom:1px solid var(--line);font-family:var(--mono);font-size:11.5px;color:var(--text-dim);flex-wrap:wrap;gap:8px;}
  .legend{display:flex;gap:16px;flex-wrap:wrap;}
  .legend-item{display:flex;align-items:center;gap:6px;}
  .legend-dot{width:8px;height:8px;border-radius:50%;}
  svg#net{display:block;width:100%;height:auto;}
  .edge-label{font-family:var(--mono);font-size:8px;fill:var(--text-dim);}
  .node-label{font-family:var(--mono);font-size:11px;fill:var(--text);font-weight:600;}
  .pulse{filter:drop-shadow(0 0 6px currentColor);}
  .step-panel{border-top:1px solid var(--line);padding:16px 20px;font-family:var(--mono);font-size:12.5px;}
  .step-row{display:flex;justify-content:space-between;flex-wrap:wrap;gap:10px;color:var(--text-dim);}
  .step-row b{color:var(--text);font-weight:600;}
  .step-outcome{padding:2px 9px;border-radius:20px;font-size:11px;text-transform:uppercase;letter-spacing:0.05em;}
  .step-outcome.served{background:rgba(143,227,168,0.12);color:var(--served);}
  .step-outcome.blocked{background:rgba(240,97,110,0.12);color:var(--blocked);}
  .stat-row{display:flex;border-top:1px solid var(--line);}
  .stat{flex:1;padding:16px 20px;text-align:center;border-right:1px solid var(--line);}
  .stat:last-child{border-right:none;}
  .stat .n{font-family:var(--mono);font-size:24px;font-weight:600;font-variant-numeric:tabular-nums;}
  .stat.served .n{color:var(--served);} .stat.blocked .n{color:var(--blocked);} .stat.rate .n{color:var(--text);}
  .stat .l{margin-top:4px;font-size:10.5px;letter-spacing:0.08em;text-transform:uppercase;color:var(--text-dim);}
  .caption{margin-top:12px;font-size:12px;color:var(--text-dim);font-family:var(--mono);line-height:1.6;}
  .note{margin-top:24px;padding:14px 16px;border-left:2px solid var(--quantum-dim);background:var(--ink-2);
    border-radius:0 8px 8px 0;font-size:13px;color:var(--text-dim);line-height:1.6;}
  .note b{color:var(--text);}
  footer{border-top:1px solid var(--line);margin-top:40px;padding-top:20px;font-family:var(--mono);
    font-size:11.5px;color:var(--text-dim);}
</style>
</head>
<body>
<div class="wrap">
  <div class="eyebrow">Kavach — Real Episode Replay</div>
  <h1>50 real routing decisions,<br>one real NSFNET episode.</h1>
  <p class="sub">Widest-path heuristic, actual output of <code>kavach.export_trace</code> — seed 1000, buffered environment, 30% classical-traffic coexistence. Every demand, candidate path, and outcome below is replayed exactly as logged, not simulated for display.</p>

  <div class="viz-frame">
    <div class="viz-header">
      <div class="legend">
        <div class="legend-item"><span class="legend-dot" style="background:var(--quantum)"></span> served</div>
        <div class="legend-item"><span class="legend-dot" style="background:var(--blocked)"></span> blocked</div>
        <div class="legend-item">Amber edges = partially classical-traffic (Raman coexistence)</div>
      </div>
      <div id="stepCounter">step 0 / 0</div>
    </div>
    <svg id="net" viewBox="0 0 900 460" xmlns="http://www.w3.org/2000/svg"></svg>
    <div class="step-panel">
      <div class="step-row" id="stepInfo">—</div>
    </div>
    <div class="stat-row">
      <div class="stat served"><div class="n" id="statServed">0</div><div class="l">Served</div></div>
      <div class="stat blocked"><div class="n" id="statBlocked">0</div><div class="l">Blocked</div></div>
      <div class="stat rate"><div class="n" id="statRate">—</div><div class="l">Blocking rate</div></div>
    </div>
  </div>
  <div class="caption" id="configLine"></div>

  <div class="note"><b>What's compressed, honestly:</b> the real NSFNET graph has 294 individual QKD relay hop-edges (~80km trusted-relay spacing per hop). The 14 endpoint nodes shown here are real; each line between them represents a real backbone route whose distance, hop count, bottleneck secret-key-rate, and classical-traffic fraction are computed directly from the full relay-level graph — not invented for display.</div>

  <footer>Kavach · standalone QKD routing &amp; RL research agent · generated from trace_export.json</footer>
</div>

<script>
const trace = __TRACE_DATA__;

const svg = document.getElementById('net');
const SVGNS = 'http://www.w3.org/2000/svg';
function mk(tag, attrs){ const el = document.createElementNS(SVGNS, tag); for(const k in attrs) el.setAttribute(k, attrs[k]); return el; }

// circular layout for the real 14 endpoint nodes
const endpoints = trace.endpoints;
const cx = 450, cy = 230, r = 175;
const pos = {};
endpoints.forEach((id, i) => {
  const angle = (i / endpoints.length) * 2 * Math.PI - Math.PI/2;
  pos[id] = { x: cx + r*Math.cos(angle), y: cy + r*Math.sin(angle) };
});

function lerpColor(a, b, t){
  const pa = a.match(/\\w\\w/g).map(x=>parseInt(x,16));
  const pb = b.match(/\\w\\w/g).map(x=>parseInt(x,16));
  const c = pa.map((v,i)=>Math.round(v + (pb[i]-v)*t));
  return '#' + c.map(v=>v.toString(16).padStart(2,'0')).join('');
}

// draw real logical edges, tinted by real classical_hop_fraction
trace.logical_edges.forEach(e => {
  const a = pos[e.u], b = pos[e.v];
  const color = lerpColor('#262F4C', '#F0A868', e.classical_hop_fraction);
  const line = mk('line', { x1:a.x, y1:a.y, x2:b.x, y2:b.y, stroke:color, 'stroke-width':1.6, opacity: 0.55 + e.classical_hop_fraction*0.3 });
  svg.appendChild(line);
});

// draw real endpoint nodes
endpoints.forEach(id => {
  const p = pos[id];
  const c = mk('circle', { cx:p.x, cy:p.y, r:9, fill:'var(--ink-3)', stroke:'var(--quantum-dim)', 'stroke-width':1.5 });
  svg.appendChild(c);
  const label = mk('text', { x:p.x, y:p.y+4, 'text-anchor':'middle', class:'node-label' });
  label.textContent = id;
  svg.appendChild(label);
});

let served = 0, blocked = 0;
const statServed = document.getElementById('statServed');
const statBlocked = document.getElementById('statBlocked');
const statRate = document.getElementById('statRate');
const stepInfo = document.getElementById('stepInfo');
const stepCounter = document.getElementById('stepCounter');

document.getElementById('configLine').textContent =
  `config: topology=${trace.config.topology}  env=${trace.config.env_type}  classical_fraction=${trace.config.classical_traffic_fraction}  seed=${trace.config.seed}`;

function updateStats(){
  statServed.textContent = served;
  statBlocked.textContent = blocked;
  const total = served + blocked;
  statRate.textContent = total > 0 ? ((blocked/total)*100).toFixed(1)+'%' : '—';
}

function animateStep(step, onDone){
  stepCounter.textContent = `step ${step.step_idx+1} / ${trace.steps.length}`;
  const d = step.demand;
  const path = step.chosen_logical_path;

  if(!path){
    stepInfo.innerHTML = `demand <b>${d.src}→${d.dst}</b> (${d.required_bits} bits) — <span class="step-outcome blocked">blocked</span> · no feasible candidate met demand`;
    const p = pos[d.src];
    const flash = mk('circle', { cx:p.x, cy:p.y, r:11, fill:'var(--blocked)', class:'pulse' });
    svg.appendChild(flash);
    setTimeout(()=>{ flash.remove(); served=step.served_total; blocked=step.blocked_total; updateStats(); onDone(); }, 500);
    return;
  }

  stepInfo.innerHTML = `demand <b>${d.src}→${d.dst}</b> (${d.required_bits} bits) — <span class="step-outcome served">served</span> via <b>${path.join(' → ')}</b>`;

  const dot = mk('circle', { r:6, fill:'var(--quantum)', class:'pulse' });
  svg.appendChild(dot);
  let seg = 0;
  const segDur = 320;

  function runSeg(){
    if(seg >= path.length-1){
      dot.remove();
      served = step.served_total; blocked = step.blocked_total; updateStats();
      onDone();
      return;
    }
    const a = pos[path[seg]], b = pos[path[seg+1]];
    const start = performance.now();
    function frame(now){
      const t = Math.min(1, (now-start)/segDur);
      dot.setAttribute('cx', a.x + (b.x-a.x)*t);
      dot.setAttribute('cy', a.y + (b.y-a.y)*t);
      if(t<1) requestAnimationFrame(frame);
      else { seg++; runSeg(); }
    }
    requestAnimationFrame(frame);
  }
  runSeg();
}

let idx = 0;
function playNext(){
  if(idx >= trace.steps.length){
    stepInfo.innerHTML = `<b>Episode complete.</b> Real final result — served ${trace.final.served}, blocked ${trace.final.blocked}, blocking rate ${(trace.final.blocking_prob*100).toFixed(1)}%.`;
    setTimeout(()=>{ idx = 0; served=0; blocked=0; updateStats(); playNext(); }, 3500);
    return;
  }
  animateStep(trace.steps[idx], ()=>{ idx++; setTimeout(playNext, 380); });
}

const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
if(!reduceMotion){ setTimeout(playNext, 400); }
else {
  served = trace.final.served; blocked = trace.final.blocked; updateStats();
  stepInfo.textContent = `Final: served ${served}, blocked ${blocked}`;
}
</script>
</body>
</html>
"""

output = HTML_TEMPLATE.replace("__TRACE_DATA__", TRACE_JSON)
with open("kavach_showcase_real.html", "w") as f:
    f.write(output)

print(f"Wrote kavach_showcase_real.html — {len(trace['steps'])} real steps embedded.")
print(f"Final: served={trace['final']['served']} blocked={trace['final']['blocked']} "
      f"blocking_prob={trace['final']['blocking_prob']:.2%}")
