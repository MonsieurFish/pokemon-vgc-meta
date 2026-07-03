"""Build a static snapshot of the meta-projection view for GitHub Pages.

GitHub Pages only serves static files, so the Flask backend (team rater, team
completer, live scraping, on-demand model inference) cannot run there. This
script pre-computes the forecast for a range of glide-lambda values from the
cached current-meta data and bakes them into a single self-contained
``docs/index.html``. The Plotly charts (current-vs-predicted scatters, the
matchup-advantage graph, the rotatable 3D PCA) all run client-side, so the page
stays interactive (including the lambda slider) without any server.

    python scripts/build_static_site.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from vgc_team.app.service import AppState
from vgc_team.config import PROJECT_ROOT

REPO_URL = "https://github.com/MonsieurFish/pokemon-vgc-meta"
LAMBDAS = [round(0.05 * i, 2) for i in range(0, 13)]  # 0.00 .. 0.60
DEFAULT_LAM = "0.30"


def build() -> Path:
    state = AppState()
    state.refresh(source="cached")
    forecasts = {f"{lam:.2f}": state.forecast(lam) for lam in LAMBDAS}
    label = forecasts[DEFAULT_LAM]["label"]
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    html = (
        TEMPLATE
        .replace("__FORECASTS__", json.dumps(forecasts))
        .replace("__DEFAULT_LAM__", DEFAULT_LAM)
        .replace("__LABEL__", json.dumps(label))
        .replace("__GENERATED__", generated)
        .replace("__REPO__", REPO_URL)
    )
    out_dir = PROJECT_ROOT / "docs"
    out_dir.mkdir(exist_ok=True)
    (out_dir / ".nojekyll").write_text("")  # serve files verbatim (underscores etc.)
    out = out_dir / "index.html"
    out.write_text(html)
    print(f"wrote {out} ({out.stat().st_size // 1024} KB) — {len(LAMBDAS)} lambda snapshots, {label}")
    return out


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>VGC Meta Forecaster</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>
  :root { --bg:#0f1420; --panel:#171e2e; --ink:#e6ebf5; --muted:#8b98b3; --accent:#5b8def; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--ink);
         font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }
  header { padding:18px 24px; border-bottom:1px solid #232c42; }
  h1 { margin:0; font-size:20px; } .sub { color:var(--muted); font-size:13px; margin-top:2px; }
  a { color:#a9c2ff; }
  .wrap { max-width:1280px; margin:0 auto; padding:18px 24px; }
  .banner { background:#1b2333; border:1px solid #2b3752; border-radius:10px; padding:12px 14px;
            font-size:13px; color:var(--muted); margin-bottom:14px; }
  .controls { display:flex; gap:18px; align-items:center; flex-wrap:wrap;
              background:var(--panel); padding:14px 16px; border-radius:10px; margin-bottom:16px; }
  select { background:#0f1420; color:var(--ink); border:1px solid #2b3752; border-radius:6px; padding:5px 8px; }
  .status { color:var(--muted); font-size:12.5px; }
  .grid2 { display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-bottom:16px; }
  @media (max-width:960px){ .grid2 { grid-template-columns:1fr; } }
  .card { background:var(--panel); border-radius:10px; padding:14px; }
  .card h2 { margin:0 0 4px; font-size:14px; display:flex; justify-content:space-between; align-items:center; }
  .card .cap { color:var(--muted); font-size:12px; margin-bottom:8px; }
  table.t5 { width:100%; border-collapse:collapse; font-size:13px; margin-top:10px; }
  table.t5 th { text-align:right; color:var(--muted); font-weight:600; font-size:11px;
                text-transform:uppercase; border-bottom:1px solid #232c42; padding:4px 6px; }
  table.t5 td { text-align:right; padding:5px 6px; border-bottom:1px solid #1d2536; }
  table.t5 td:first-child, table.t5 th:first-child { text-align:left; }
  .up { color:#5fd39a; } .down { color:#e77; }
  .movers { font-size:12.5px; color:var(--muted); margin-top:10px; }
  input[type=range]{ width:170px; } code { color:#a9c2ff; }
</style>
</head>
<body>
<header>
  <h1>VGC Meta Forecaster <span style="color:var(--muted);font-size:13px;font-weight:400;">&middot; static snapshot</span></h1>
  <div class="sub">Recent tournament teams &rarr; frozen encoder &rarr; glide-to-anchor forecast + matchup model.
     Points above the diagonal are predicted to <b style="color:#5fd39a">expand</b>.</div>
</header>
<div class="wrap">
  <div class="banner">
    This is a static snapshot generated on <b>__GENERATED__</b>. The interactive
    <b>Team rater</b>, <b>Team completer</b>, and live tournament scraping run model inference on a Python
    backend and are not available on GitHub Pages &mdash; run the full app locally
    (<code>python scripts/run_meta_app.py</code>). Source &amp; instructions:
    <a href="__REPO__">__REPO__</a>.
  </div>

  <div class="controls">
    <label>Glide &lambda; <input id="lam" type="range" min="0" max="0.6" step="0.05" value="0.3" />
      <span id="lamval"><code>0.30</code></span></label>
    <span class="status" id="status"></span>
  </div>

  <div class="grid2">
    <div class="card">
      <h2>Pok&eacute;mon cores</h2>
      <div class="cap">specific team cores &middot; current vs predicted usage share</div>
      <div id="scatterCores" style="height:400px;"></div>
      <table class="t5"><thead id="topCoresHead"></thead><tbody id="topCores"></tbody></table>
      <div class="movers" id="moversCores"></div>
    </div>
    <div class="card">
      <h2>Team archetypes</h2>
      <div class="cap">coarse families &middot; current vs predicted share</div>
      <div id="scatterArch" style="height:400px;"></div>
      <table class="t5"><thead id="topArchHead"></thead><tbody id="topArch"></tbody></table>
      <div class="movers" id="moversArch"></div>
    </div>
  </div>

  <div class="grid2">
    <div class="card">
      <h2>Matchup advantage
        <select id="graphLevel"><option value="families">Team archetypes</option>
          <option value="cores">Pok&eacute;mon cores</option></select>
      </h2>
      <div class="cap">bubble size = share &middot; colour = win vs field &middot; arrow A&rarr;B = A favoured over B</div>
      <div id="graph" style="height:460px;"></div>
    </div>
    <div class="card">
      <h2>Meta space (3D &mdash; drag to rotate)</h2>
      <div class="cap">bubble size = share &middot; colour = predicted growth (green) / decline (red)</div>
      <div id="space3d" style="height:460px;"></div>
    </div>
  </div>
</div>

<script>
const FORECASTS = __FORECASTS__;
const $ = (id) => document.getElementById(id);
const pct = (x) => (x*100).toFixed(1) + "%";
const pt  = (x) => (x*100>=0?"+":"") + (x*100).toFixed(2);
const dark = { paper_bgcolor:"#171e2e", plot_bgcolor:"#171e2e", font:{color:"#e6ebf5"}, margin:{l:52,r:12,t:12,b:44} };
const coreName = (p)=> p.label || ("c"+p.id);
const archName = (p)=> p.name;
let LAST = null;

function renderScatter(div, pts, nameFn, labelThresh){
  const vals = pts.flatMap(p=>[p.current,p.predicted]);
  const mx = Math.max(0.01, ...vals) * 100 * 1.08;
  const maxAbs = Math.max(1e-6, ...pts.map(p=>Math.abs(p.delta)));
  Plotly.react(div,[{
    x:pts.map(p=>p.current*100), y:pts.map(p=>p.predicted*100), mode:"markers+text", type:"scatter",
    marker:{ size:12, color:pts.map(p=>p.delta), colorscale:"RdYlGn", cmin:-maxAbs, cmax:maxAbs, cmid:0,
             line:{width:1,color:"#0f1420"}, colorbar:{title:"Δ", thickness:8} },
    text:pts.map(p=> Math.max(p.current,p.predicted)>=labelThresh ? nameFn(p) : ""),
    textposition:"top center", textfont:{size:9,color:"#c7d2ee"},
    customdata:pts.map(p=>[nameFn(p), pct(p.current), pct(p.predicted), pt(p.delta)]),
    hovertemplate:"<b>%{customdata[0]}</b><br>%{customdata[1]} → %{customdata[2]} (%{customdata[3]} pt)<extra></extra>",
  }], Object.assign({}, dark, {
    xaxis:{title:"current share %", rangemode:"tozero", gridcolor:"#232c42", range:[0,mx]},
    yaxis:{title:"predicted next-week share %", rangemode:"tozero", gridcolor:"#232c42", range:[0,mx]},
    shapes:[{type:"line", x0:0,y0:0,x1:mx,y1:mx, line:{dash:"dot",color:"#5b8def",width:1}}],
    showlegend:false }), {displayModeBar:false});
}
function renderTop5(head, body, pts, nameFn, unit){
  head.innerHTML = `<tr><th>${unit}</th><th>now</th><th>next</th><th>Δpt</th></tr>`;
  body.innerHTML = pts.map(p=>
    `<tr><td>${nameFn(p)}</td><td>${pct(p.current)}</td><td>${pct(p.predicted)}</td>`+
    `<td class="${p.delta>=0?'up':'down'}">${pt(p.delta)}</td></tr>`).join("");
}
function renderMovers(div, pts, nameFn){
  const s = [...pts].sort((a,b)=>b.delta-a.delta);
  const line = (p)=>`<span class="${p.delta>=0?'up':'down'}">${pt(p.delta)}</span> ${nameFn(p)}`;
  div.innerHTML = "▲ " + s.slice(0,3).map(line).join(" · ") + "<br>▼ " + s.slice(-3).reverse().map(line).join(" · ");
}
function renderGraph(g){
  const div = $("graph");
  if(!g || !g.nodes.length){ Plotly.purge(div); div.innerHTML="<div class='cap'>matchup model unavailable</div>"; return; }
  const N = g.nodes; const short=(s)=> s.length>16 ? s.slice(0,15)+"…" : s;
  const trace = { x:N.map(n=>n.x), y:N.map(n=>n.y), mode:"markers+text", type:"scatter",
    marker:{ size:N.map(n=>16+64*Math.sqrt(n.share)), color:N.map(n=>n.field_winrate),
             colorscale:"RdYlGn", cmin:0.45, cmax:0.55, cmid:0.5, line:{width:1.5,color:"#0f1420"},
             colorbar:{title:"vs field", thickness:8} },
    text:N.map(n=>short(n.label)), textposition:"bottom center", textfont:{size:9,color:"#c7d2ee"},
    customdata:N.map(n=>[n.label, pct(n.share), n.field_winrate.toFixed(3)]),
    hovertemplate:"<b>%{customdata[0]}</b><br>share %{customdata[1]} · win vs field %{customdata[2]}<extra></extra>" };
  const ann = g.edges.map(e=>{ const s=e.advantage-0.5; const op=Math.min(1,0.7+9*s);
    return { ax:N[e.source].x, ay:N[e.source].y, x:N[e.target].x, y:N[e.target].y,
      xref:"x",yref:"y",axref:"x",ayref:"y", showarrow:true, arrowhead:3, arrowsize:1.6, arrowwidth:2+13*s,
      arrowcolor:"rgba(125,195,255,"+op.toFixed(2)+")", standoff:16, startstandoff:9 }; });
  Plotly.react(div,[trace], Object.assign({}, dark, { showlegend:false,
    xaxis:{visible:false, range:[-1.4,1.4]}, yaxis:{visible:false, range:[-1.4,1.4], scaleanchor:"x"},
    annotations:ann, margin:{l:10,r:10,t:10,b:10} }), {displayModeBar:false});
}
function render3D(anchors, clusters){
  const maxCur = Math.max(1e-6, ...anchors.map(a=>a.current));
  const maxAbs = Math.max(1e-6, ...anchors.map(a=>Math.abs(a.delta)));
  const A = { type:"scatter3d", mode:"markers", x:anchors.map(a=>a.x), y:anchors.map(a=>a.y), z:anchors.map(a=>a.z),
    marker:{ size:anchors.map(a=>3+22*Math.sqrt(a.current/maxCur)), color:anchors.map(a=>a.delta),
             colorscale:"RdYlGn", cmin:-maxAbs, cmax:maxAbs, cmid:0, opacity:0.85, colorbar:{title:"Δ", thickness:8} },
    text:anchors.map(a=> a.label ? a.label.split(" | ").slice(0,2).join(" / ") : ("anchor "+a.id)), hoverinfo:"text" };
  const C = { type:"scatter3d", mode:"text", x:clusters.map(c=>c.x), y:clusters.map(c=>c.y), z:clusters.map(c=>c.z),
    text:clusters.map(c=>c.label), textfont:{size:9,color:"#c7d2ee"}, hoverinfo:"skip" };
  Plotly.react($("space3d"), [A,C], Object.assign({}, dark, { showlegend:false,
    scene:{ bgcolor:"#171e2e", xaxis:{visible:false}, yaxis:{visible:false}, zaxis:{visible:false} },
    margin:{l:0,r:0,t:0,b:0} }), {displayModeBar:false, responsive:true});
}
function render(d){
  LAST = d;
  renderScatter($("scatterCores"), d.cores, coreName, 0.03);
  renderScatter($("scatterArch"),  d.archetypes, archName, 0);
  renderTop5($("topCoresHead"), $("topCores"), d.top_cores, coreName, "core");
  renderTop5($("topArchHead"),  $("topArch"),  d.top_archetypes, archName, "archetype");
  renderMovers($("moversCores"), d.cores.filter(c=>c.current>=0.01), coreName);
  renderMovers($("moversArch"),  d.archetypes, archName);
  renderGraph($("graphLevel").value === "cores" ? d.matchup_cores : d.matchup_families);
  render3D(d.anchors3d, d.clusters3d);
  const last = d.week_labels[d.week_labels.length-1] || "?";
  $("status").textContent = `${d.n_teams.toLocaleString()} teams · ${d.n_weeks} weeks · forecasting after ${last} · ${__LABEL__}`;
}
function show(){ const v=(+$("lam").value).toFixed(2); $("lamval").innerHTML=`<code>${v}</code>`;
  render(FORECASTS[v] || FORECASTS["__DEFAULT_LAM__"]); }
$("lam").addEventListener("input", show);
$("graphLevel").addEventListener("change", ()=>{ if(LAST) renderGraph(
  $("graphLevel").value==="cores" ? LAST.matchup_cores : LAST.matchup_families); });
show();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    build()
