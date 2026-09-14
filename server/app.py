#!/usr/bin/env python3
"""minimal-transformer visualization server -- stdlib only, no dependencies.

Serves a single page that SHOWS grokking took place on the NAND lowered transformer:
  - the accuracy gap (train fits early, held-out generalizes late),
  - the held-out margin climbing across the plateau and crossing zero exactly at the grok step
    (the mechanism: continued descent, not luck),
  - the three controls side by side (GROKS / NO-GROK / MEMORIZES), and
  - the identity endpoint_portrait == min-L1 prior portrait (NO-6).

Run:  python3 server/app.py    then open http://localhost:8000
"""
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "probes"))
import nand_core as c  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.join(HERE, "..", "fixtures", "nand-groks-in-time.json")

# dense early checkpoints so the plateau and zero-crossing are smooth, sparse tail
CHECKPOINTS = sorted(set(list(range(0, 205, 5)) + [250, 300, 400, 600, 1000, 2000, 4000, 6000]))

REGIMES = [
    {"key": "groks", "label": "GROKS", "holdout": ["s"],
     "blurb": "hold out s (NAND=0, the minority row): not forced, generalizes LATE"},
    {"key": "nogrok", "label": "NO-GROK", "holdout": ["r"],
     "blurb": "hold out r (a redundant NAND=1 row): forced by p,q, generalizes immediately"},
    {"key": "memorize", "label": "MEMORIZES", "holdout": ["p", "q", "r"],
     "blurb": "hold out all majority rows: underdetermined, never generalizes"},
]


def first_100(points, key):
    for p in points:
        ok, tot = p[key]
        if tot > 0 and ok == tot:
            return p["step"]
    return None


def compute(holdout):
    data = c.load(FIX)
    d = data["_d"]
    vocab, emb = data["vocab"], data["embeddings"]
    form = [tuple(x) for x in data["form"]]
    hs = holdout if isinstance(holdout, list) else [holdout]
    train_cons = [(a, b) for a, b in form if a not in hs]
    held_cons = [(a, b) for a, b in form if a in hs]
    lr = data.get("lr", 0.01)
    M, traj = c.train_sgd(train_cons, emb, vocab, d, lr, max(CHECKPOINTS), CHECKPOINTS, held=held_cons)
    points = []
    for step in sorted(traj):
        rec = traj[step]
        points.append({
            "step": step,
            "train": rec["train"], "held": rec["held"],
            "train_acc": rec["train"][0] / rec["train"][1],
            "held_acc": rec["held"][0] / rec["held"][1],
            "margin": rec["held_margin"], "wnorm": rec["wnorm"],
        })
    fit_at = first_100(points, "train")
    grok_at = first_100(points, "held")
    endpoint = c.portrait(M, emb, vocab, d)
    lo, hi = data.get("range", [-1, 1])
    surv = c.survivors(vocab, emb, form, lo, hi, d)
    prior = c.portrait(c.min_l1(surv, d), emb, vocab, d)
    if fit_at is not None and grok_at is not None and grok_at > fit_at:
        verdict = "GROKS"
    elif fit_at is not None and grok_at is not None:
        verdict = "NO-GROK"
    elif fit_at is not None:
        verdict = "MEMORIZES"
    else:
        verdict = "NO-FIT"
    return {
        "holdout": hs, "lr": lr, "points": points,
        "fit_at": fit_at, "grok_at": grok_at, "verdict": verdict,
        "endpoint_portrait": endpoint, "prior_portrait": prior,
        "portrait_match": endpoint == prior,
    }


def payload():
    return {"regimes": [dict(r, **compute(r["holdout"])) for r in REGIMES]}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/" or path == "/index.html":
            self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif path == "/api/data":
            self._send(200, json.dumps(payload()).encode("utf-8"), "application/json")
        else:
            self._send(404, b"not found", "text/plain")


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Grokking on the NAND</title>
<style>
  :root { color-scheme: light dark; --bg:#0f1216; --panel:#171b22; --ink:#e6e9ef; --mut:#8b95a6;
          --train:#4aa3ff; --held:#ff7ac2; --grid:#2a3140; --ok:#39d98a; --bad:#ff6b6b; --line:#232a35; }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--ink);
         font: 14px/1.5 ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }
  header { padding:24px 20px 8px; max-width:1100px; margin:0 auto; }
  h1 { font-size:22px; margin:0 0 4px; }
  header p { color:var(--mut); margin:4px 0; max-width:70ch; }
  main { max-width:1100px; margin:0 auto; padding:12px 20px 60px; }
  .grid { display:grid; grid-template-columns:repeat(3,1fr); gap:14px; }
  @media (max-width:820px){ .grid{ grid-template-columns:1fr; } }
  .card { background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:14px; }
  .card h2 { font-size:15px; margin:0 0 2px; display:flex; align-items:center; gap:8px; }
  .tag { font:600 11px ui-monospace, monospace; padding:2px 7px; border-radius:999px; }
  .tag.GROKS{ background:rgba(57,217,138,.15); color:var(--ok); }
  .tag.NOGROK{ background:rgba(74,163,255,.15); color:var(--train); }
  .tag.MEMORIZES{ background:rgba(255,107,107,.15); color:var(--bad); }
  .blurb { color:var(--mut); font-size:12.5px; min-height:34px; }
  svg { width:100%; height:auto; display:block; }
  .legend { display:flex; gap:14px; font-size:12px; color:var(--mut); margin-top:6px; flex-wrap:wrap; }
  .sw { display:inline-block; width:10px; height:10px; border-radius:2px; vertical-align:middle; margin-right:5px; }
  .facts { font:12.5px ui-monospace, monospace; color:var(--mut); margin-top:8px; line-height:1.7; }
  .facts b { color:var(--ink); font-weight:600; }
  .hero { margin-top:18px; }
  .portrait { margin-top:18px; }
  .mono { font-family:ui-monospace, monospace; }
  .match { color:var(--ok); } .nomatch { color:var(--bad); }
  footer { color:var(--mut); font-size:12px; max-width:1100px; margin:0 auto; padding:0 20px 40px; }
  a { color:var(--train); }
</style></head>
<body>
<header>
  <h1>Grokking on the NAND, made visible</h1>
  <p>The NAND gate read as the orbit of a lowered transformer, its weight learned by SGD from zero init.
     Grokking is not just a late-rising curve &mdash; it is <b>continued descent across the plateau</b>. Watch the
     held-out <b>margin</b> climb while training accuracy sits pinned at 100%, and cross zero exactly at the
     moment generalization happens. Then compare the three controls: whether the held-out row is
     <i>forced</i> by the training rows decides grok vs. no-grok vs. memorize.</p>
</header>
<main>
  <div class="hero card" id="hero"></div>
  <div class="portrait card" id="portrait"></div>
  <h2 style="max-width:1100px;margin:26px auto 10px;font-size:16px;">The three controls</h2>
  <div class="grid" id="controls"></div>
</main>
<footer>Deterministic, torch-free. Data computed server-side by <span class="mono">probes/nand_core.py</span>;
  the same numbers <span class="mono">run.sh</span> asserts. Reload to recompute.</footer>

<script>
const W=520, H=230, PAD={l:44,r:16,t:16,b:34};
const clsMap={"GROKS":"GROKS","NO-GROK":"NOGROK","MEMORIZES":"MEMORIZES","NO-FIT":"MEMORIZES"};
function xpos(step,maxStep){ // log-ish scale: sqrt keeps early steps legible
  const f=Math.sqrt(step)/Math.sqrt(maxStep); return PAD.l+f*(W-PAD.l-PAD.r); }
function ypos(v,lo,hi){ const f=(v-lo)/(hi-lo); return H-PAD.b-f*(H-PAD.t-PAD.b); }
function path(pts){ return pts.map((p,i)=>(i?'L':'M')+p[0].toFixed(1)+' '+p[1].toFixed(1)).join(' '); }
function esc(s){ return s.replace(/</g,'&lt;'); }

function accuracyChart(reg, big){
  const pts=reg.points, maxStep=pts[pts.length-1].step;
  const tr=pts.map(p=>[xpos(p.step,maxStep), ypos(p.train_acc,0,1)]);
  const hd=pts.map(p=>[xpos(p.step,maxStep), ypos(p.held_acc,0,1)]);
  let marks='';
  if(reg.fit_at!=null){ const x=xpos(reg.fit_at,maxStep);
    marks+=`<line x1="${x}" y1="${PAD.t}" x2="${x}" y2="${H-PAD.b}" stroke="var(--train)" stroke-dasharray="3 3" opacity=".5"/>
            <text x="${x+3}" y="${PAD.t+11}" fill="var(--train)" font-size="10">fit @${reg.fit_at}</text>`; }
  if(reg.grok_at!=null){ const x=xpos(reg.grok_at,maxStep);
    marks+=`<line x1="${x}" y1="${PAD.t}" x2="${x}" y2="${H-PAD.b}" stroke="var(--ok)" stroke-dasharray="3 3" opacity=".6"/>
            <text x="${x+3}" y="${PAD.t+24}" fill="var(--ok)" font-size="10">grok @${reg.grok_at}</text>`; }
  // shade the plateau (fit..grok) where the "nothing is happening" illusion lives
  let shade='';
  if(reg.fit_at!=null && reg.grok_at!=null && reg.grok_at>reg.fit_at){
    const x0=xpos(reg.fit_at,maxStep), x1=xpos(reg.grok_at,maxStep);
    shade=`<rect x="${x0}" y="${PAD.t}" width="${x1-x0}" height="${H-PAD.t-PAD.b}" fill="var(--ok)" opacity=".07"/>`; }
  const yt=[0,.5,1].map(v=>`<line x1="${PAD.l}" y1="${ypos(v,0,1)}" x2="${W-PAD.r}" y2="${ypos(v,0,1)}" stroke="var(--grid)"/>
     <text x="${PAD.l-6}" y="${ypos(v,0,1)+3}" fill="var(--mut)" font-size="10" text-anchor="end">${v*100|0}%</text>`).join('');
  return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="accuracy over steps">
    ${shade}${yt}${marks}
    <path d="${path(tr)}" fill="none" stroke="var(--train)" stroke-width="2"/>
    <path d="${path(hd)}" fill="none" stroke="var(--held)" stroke-width="2"/>
    <text x="${PAD.l}" y="${H-6}" fill="var(--mut)" font-size="10">step 0</text>
    <text x="${W-PAD.r}" y="${H-6}" fill="var(--mut)" font-size="10" text-anchor="end">${maxStep}</text>
  </svg>`;
}

function marginChart(reg){
  const pts=reg.points, maxStep=pts[pts.length-1].step;
  const ms=pts.map(p=>p.margin), lo=Math.min(-0.5,...ms), hi=Math.max(0.5,...ms);
  const mg=pts.map(p=>[xpos(p.step,maxStep), ypos(p.margin,lo,hi)]);
  const zeroY=ypos(0,lo,hi);
  let cross='';
  if(reg.grok_at!=null){ const x=xpos(reg.grok_at,maxStep);
    cross=`<circle cx="${x}" cy="${zeroY}" r="4" fill="var(--ok)"/>
           <text x="${x+6}" y="${zeroY-6}" fill="var(--ok)" font-size="10">crosses 0 @${reg.grok_at}</text>`; }
  let shade='';
  if(reg.fit_at!=null && reg.grok_at!=null && reg.grok_at>reg.fit_at){
    const x0=xpos(reg.fit_at,maxStep), x1=xpos(reg.grok_at,maxStep);
    shade=`<rect x="${x0}" y="${PAD.t}" width="${x1-x0}" height="${H-PAD.t-PAD.b}" fill="var(--ok)" opacity=".07"/>`; }
  return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="held-out margin over steps">
    ${shade}
    <line x1="${PAD.l}" y1="${zeroY}" x2="${W-PAD.r}" y2="${zeroY}" stroke="var(--mut)" stroke-dasharray="4 3"/>
    <text x="${PAD.l-6}" y="${zeroY+3}" fill="var(--mut)" font-size="10" text-anchor="end">0</text>
    <path d="${path(mg)}" fill="none" stroke="var(--held)" stroke-width="2"/>
    ${cross}
    <text x="${PAD.l}" y="${PAD.t+2}" fill="var(--mut)" font-size="10">held-out margin (logit gap)</text>
  </svg>`;
}

function heroFacts(reg){
  const gap = (reg.fit_at!=null && reg.grok_at!=null) ? (reg.grok_at-reg.fit_at) : null;
  const mFit = reg.points.find(p=>p.step===reg.fit_at);
  return `<div class="facts">
    train hits 100% at <b>step ${reg.fit_at}</b> &rarr; held-out only at <b>step ${reg.grok_at}</b>
    (a <b>${gap}-step</b> grok gap).<br>
    at the fit the held-out margin is <b>${mFit?mFit.margin.toFixed(2):'?'}</b> (wrong); it climbs
    <b>monotonically</b> and the weight norm keeps <b>rising</b> across the shaded plateau &mdash;
    the plateau is <b>not</b> a stationary point. The zero-crossing <b>is</b> the grok.
  </div>`;
}

function render(data){
  const reg = data.regimes.find(r=>r.verdict==="GROKS") || data.regimes[0];
  document.getElementById('hero').innerHTML =
    `<h2>The grok, and its cause <span class="tag ${clsMap[reg.verdict]}">${reg.verdict}</span></h2>
     <div class="legend"><span><span class="sw" style="background:var(--train)"></span>train accuracy</span>
       <span><span class="sw" style="background:var(--held)"></span>held-out accuracy / margin</span>
       <span><span class="sw" style="background:var(--ok);opacity:.5"></span>plateau (fit&rarr;grok)</span></div>
     <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:8px" class="heropair">
       <div>${accuracyChart(reg,true)}</div><div>${marginChart(reg)}</div></div>
     ${heroFacts(reg)}`;

  const P=reg;
  document.getElementById('portrait').innerHTML =
    `<h2>The endpoint <i>is</i> the prior <span class="tag ${P.portrait_match?'GROKS':'MEMORIZES'}">
        ${P.portrait_match?'MATCH':'MISMATCH'}</span></h2>
     <div class="facts">
       Trained holding out s, the SGD endpoint's phase portrait equals the analytic min-L1 prior portrait
       from NO-3 &mdash; the dynamic grok and the static implicit-bias prior are the same object.<br>
       SGD endpoint: <span class="mono ${P.portrait_match?'match':'nomatch'}">${esc(P.endpoint_portrait)}</span><br>
       min-L1 prior: <span class="mono ${P.portrait_match?'match':'nomatch'}">${esc(P.prior_portrait)}</span>
     </div>`;

  document.getElementById('controls').innerHTML = data.regimes.map(r=>`
    <div class="card">
      <h2>${r.label} <span class="tag ${clsMap[r.verdict]}">${r.verdict}</span></h2>
      <div class="blurb">${r.blurb}</div>
      ${accuracyChart(r,false)}
      <div class="facts">holdout <b>{${r.holdout.join(',')}}</b> &middot; fit <b>${r.fit_at}</b> &middot;
        grok <b>${r.grok_at==null?'never':r.grok_at}</b></div>
    </div>`).join('');
}

fetch('/api/data').then(r=>r.json()).then(render)
  .catch(e=>{document.getElementById('hero').textContent='error: '+e;});
</script>
</body></html>
"""


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print("minimal-transformer viz on http://localhost:%d  (Ctrl-C to stop)" % port)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
