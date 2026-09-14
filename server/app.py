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
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "probes"))
import nand_core as c  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.join(HERE, "..", "fixtures", "nand-groks-in-time.json")

# record EVERY step so the scrubber moves one training step at a time
VIZ_STEPS = 400
CHECKPOINTS = list(range(0, VIZ_STEPS + 1))

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


def compute(holdout, seed=None):
    data = c.load(FIX)
    d = data["_d"]
    vocab, emb = data["vocab"], data["embeddings"]
    form = [tuple(x) for x in data["form"]]
    hs = holdout if isinstance(holdout, list) else [holdout]
    train_cons = [(a, b) for a, b in form if a not in hs]
    held_cons = [(a, b) for a, b in form if a in hs]
    lr = data.get("lr", 0.01)
    # honest wall-clock cost of the actual learning: a bare run (no per-step recording), same seed
    t0 = time.perf_counter()
    c.train_sgd(train_cons, emb, vocab, d, lr, max(CHECKPOINTS), [max(CHECKPOINTS)], held=None, seed=seed)
    train_ms = (time.perf_counter() - t0) * 1000.0
    M, traj = c.train_sgd(train_cons, emb, vocab, d, lr, max(CHECKPOINTS), CHECKPOINTS,
                          held=held_cons, seed=seed)
    points = []
    for step in sorted(traj):
        rec = traj[step]
        Mi = rec["M"]
        held_pred = [[s, c.predict(Mi, s, emb, vocab, d), t] for s, t in held_cons]
        points.append({
            "step": step,
            "train": rec["train"], "held": rec["held"],
            "train_acc": rec["train"][0] / rec["train"][1],
            "held_acc": rec["held"][0] / rec["held"][1],
            "margin": rec["held_margin"], "wnorm": rec["wnorm"],
            "portrait": c.portrait(Mi, emb, vocab, d),
            "held_pred": held_pred, "M": Mi,
        })
    wmax = max((abs(v) for pt in points for row in pt["M"] for v in row), default=1.0) or 1.0
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
        "holdout": hs, "lr": lr, "points": points, "wmax": wmax,
        "train_ms": round(train_ms, 1), "steps": max(CHECKPOINTS),
        "fit_at": fit_at, "grok_at": grok_at, "verdict": verdict,
        "endpoint_portrait": endpoint, "prior_portrait": prior,
        "portrait_match": endpoint == prior,
    }


def payload(seed=None):
    return {"seed": seed, "init": "zero" if seed is None else "random",
            "regimes": [dict(r, **compute(r["holdout"], seed=seed)) for r in REGIMES]}


def selfcheck():
    """Reproduce the EXACT run.sh NO-4 verdict on the zero-init default, so a freshly started server is
    self-evidently the same deterministic machine the CLI asserts -- without needing to run run.sh. Uses the
    coarse CLI checkpoint grid from the fixture (not the server's per-step grid), so the numbers match the
    asserted string. Returns the summary and whether it matches the expected zero-init default."""
    data = c.load(FIX)
    d = data["_d"]
    vocab, emb = data["vocab"], data["embeddings"]
    form = [tuple(x) for x in data["form"]]
    hs = [data.get("holdout", "s")] if not isinstance(data.get("holdout"), list) else data["holdout"]
    train_cons = [(a, b) for a, b in form if a not in hs]
    held_cons = [(a, b) for a, b in form if a in hs]
    cps = sorted(set(data.get("checkpoints", [0, 10, 25, 50, 100, 200, 400, 800, 1600, 3200, 6000])))
    M, traj = c.train_sgd(train_cons, emb, vocab, d, data.get("lr", 0.01), max(cps), cps,
                          held=held_cons, seed=None)

    def first(key):
        for s in cps:
            ok, tot = traj[s][key]
            if tot > 0 and ok == tot:
                return s
        return None
    fit, grok = first("train"), first("held")
    endpoint = c.portrait(M, emb, vocab, d)
    lo, hi = data.get("range", [-1, 1])
    prior = c.portrait(c.min_l1(c.survivors(vocab, emb, form, lo, hi, d), d), emb, vocab, d)
    verdict = "GROKS" if fit is not None and grok is not None and grok > fit else "OTHER"
    # the exact expectation the CLI (run.sh NO-4a) asserts for the zero-init default
    expected = (verdict == "GROKS" and fit == 50 and grok == 100 and endpoint == prior)
    return {
        "verdict": verdict, "fit_at": fit, "grok_at": grok,
        "endpoint_portrait": endpoint, "prior_portrait": prior,
        "endpoint_is_prior": endpoint == prior,
        "matches_runsh_no4": expected,
    }


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
        elif path == "/explain" or path == "/explain.html":
            self._send(200, PAGE_EXPLAIN.encode("utf-8"), "text/html; charset=utf-8")
        elif path == "/api/data":
            qs = parse_qs(urlparse(self.path).query)
            seed = None
            if "seed" in qs:
                try:
                    seed = int(qs["seed"][0])
                except ValueError:
                    seed = None
            self._send(200, json.dumps(payload(seed)).encode("utf-8"), "application/json")
        elif path == "/api/health":
            self._send(200, json.dumps(selfcheck()).encode("utf-8"), "application/json")
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
  .herohint { color:var(--mut); font-size:13px; margin:4px 0 12px; }
  .seg { display:inline-flex; gap:4px; padding:4px; background:var(--bg); border:1px solid var(--line);
         border-radius:10px; margin-bottom:6px; flex-wrap:wrap; }
  .segbtn { background:transparent; border:none; color:var(--mut); font:600 12.5px ui-monospace, monospace;
            padding:6px 14px; border-radius:7px; cursor:pointer; transition:background .12s, color .12s; }
  .segbtn:hover { color:var(--ink); }
  .segbtn.on { background:var(--held); color:#111; }
  .card[data-k] { cursor:pointer; transition:border-color .12s, transform .12s; }
  .card[data-k]:hover { border-color:var(--held); transform:translateY(-1px); }
  .card.sel { border-color:var(--held); box-shadow:0 0 0 1px var(--held); }
  .portrait { margin-top:18px; }
  .mono { font-family:ui-monospace, monospace; }
  .match { color:var(--ok); } .nomatch { color:var(--bad); } .mut { color:var(--mut); }
  .scrubwrap { display:flex; align-items:center; gap:12px; margin-top:14px; }
  .scrubwrap input[type=range]{ flex:1; accent-color:var(--held); }
  .btn { background:var(--held); color:#111; border:none; border-radius:8px; padding:6px 12px;
         font:600 13px system-ui; cursor:pointer; }
  .phase { font:600 12px ui-monospace, monospace; color:var(--ink); white-space:nowrap; }
  .initline { font-size:11.5px; margin-top:6px; }
  .timenote { font-size:12.5px; margin-top:6px; background:var(--bg); border:1px solid var(--line);
              border-left:3px solid var(--ok); border-radius:0 8px 8px 0; padding:8px 12px; line-height:1.5; }
  .btn:disabled { opacity:.6; cursor:default; }
  .state { margin-top:12px; display:grid; grid-template-columns:repeat(5,1fr);
           gap:6px 18px; background:var(--bg); border:1px solid var(--line); border-radius:10px; padding:12px; }
  @media (max-width:640px){ .state{ grid-template-columns:repeat(2,1fr); } }
  .staterow { display:flex; justify-content:space-between; gap:10px; align-items:baseline; min-height:22px;
              border-bottom:1px dashed var(--line); padding-bottom:3px; white-space:nowrap; overflow:hidden; }
  .staterow.wide { grid-column:1/-1; }
  .staterow span:first-child { color:var(--mut); font-size:12px; }
  .weights { margin-top:12px; }
  .wgrid-wrap { display:flex; gap:18px; align-items:center; flex-wrap:wrap; background:var(--bg);
                border:1px solid var(--line); border-radius:10px; padding:14px; }
  .wtitle { font-size:12px; color:var(--mut); margin-bottom:8px; }
  .wgrid { display:grid; grid-template-columns:repeat(3,48px); grid-auto-rows:48px; gap:5px; }
  .wcell { display:grid; place-items:center; border:1px solid var(--line); border-radius:6px;
           font:600 12px ui-monospace, monospace; color:var(--ink); transition:background .08s linear; }
  .wcap { flex:1; min-width:230px; color:var(--mut); font-size:12.5px; line-height:1.55; }
  .pnote { color:var(--mut); font-size:12px; margin-top:10px; line-height:1.5;
           border-top:1px dashed var(--line); padding-top:8px; }
  .terms { font-size:12.5px; margin-top:10px; }
  .terms a { margin:0 2px; }
  footer { color:var(--mut); font-size:12px; max-width:1100px; margin:0 auto; padding:0 20px 40px; }
  a { color:var(--train); }
  .topbar { display:flex; align-items:flex-start; justify-content:space-between; gap:16px; }
  .learnbtn { flex:none; background:var(--panel); border:1px solid var(--line); color:var(--ink);
              border-radius:999px; padding:8px 16px; font:600 13px system-ui; text-decoration:none;
              white-space:nowrap; transition:border-color .15s, transform .15s; }
  .learnbtn:hover { border-color:var(--held); transform:translateY(-1px); }
  a.term { color:var(--held); text-decoration:none; border-bottom:1px dotted var(--held); cursor:pointer; }
  a.term:hover { border-bottom-style:solid; }
  .learnbtn.sm { display:inline-block; margin-top:10px; padding:5px 12px; font-size:12px; }
  /* tutorial */
  #learn { max-width:760px; margin:56px auto 0; padding:0 20px; }
  .learnintro { border-top:1px solid var(--line); padding-top:40px; }
  .learnintro h2 { font-size:24px; margin:0 0 6px; }
  .learnintro p { color:var(--mut); font-size:15px; max-width:60ch; }
  .lesson { position:relative; padding:26px 0 26px 54px; border-bottom:1px solid var(--line);
            opacity:0; transform:translateY(16px); transition:opacity .5s ease, transform .5s ease; }
  .lesson.in { opacity:1; transform:none; }
  .lesson .num { position:absolute; left:0; top:26px; width:36px; height:36px; border-radius:50%;
                 display:grid; place-items:center; background:var(--held); color:#111;
                 font:700 16px ui-monospace, monospace; }
  .lesson h3 { font-size:19px; margin:2px 0 10px; }
  .lesson p, .lesson li { font-size:15px; line-height:1.65; color:var(--ink); }
  .lesson ol, .lesson ul { padding-left:20px; } .lesson li { margin:6px 0; }
  .lesson i { color:var(--mut); font-style:italic; }
  .callout { background:var(--bg); border:1px solid var(--line); border-left:3px solid var(--held);
             border-radius:0 8px 8px 0; padding:12px 14px; margin-top:14px; font-size:14px; color:var(--mut); }
  .callout b:first-child { color:var(--held); }
  table.map { width:100%; border-collapse:collapse; margin:12px 0; font-size:13.5px; }
  table.map th, table.map td { text-align:left; padding:8px 10px; border-bottom:1px solid var(--line);
                               vertical-align:top; }
  table.map th { color:var(--mut); font-weight:600; }
  table.map tr td:first-child { color:var(--mut); width:46%; }
  .controls-list { list-style:none; padding-left:0; }
  .controls-list li { border-bottom:1px dashed var(--line); padding:6px 0; }
  .c-grok{ color:var(--ok);} .c-no{ color:var(--train);} .c-mem{ color:var(--bad); }
  .closer { font-size:16px; font-weight:600; color:var(--ink); margin-top:20px; }
  .diagram { margin:18px 0 4px; }
  .diagram svg { width:100%; height:auto; display:block; background:var(--bg);
                 border:1px solid var(--line); border-radius:10px; }
  .diagram figcaption { color:var(--mut); font-size:12.5px; line-height:1.5; margin-top:8px; }
  #learn .lesson:last-child { border-bottom:none; }
  #learn .lesson .learnbtn { display:inline-block; margin-top:16px; }
</style></head>
<body>
<header>
  <div class="topbar">
    <h1>Watch a tiny AI suddenly learn</h1>
    <a href="#learn" class="learnbtn" id="learnbtn">Learn more &darr;</a>
  </div>
  <p>Below is a very small artificial &ldquo;brain&rdquo;, simple enough that every part of it is visible, learning a basic rule from examples. For a long stretch it looks stuck: it has memorized its practice
     set but fails anything new. Then, all at once, it <b>clicks</b> and starts getting new cases right. That sudden
     click has a name: <b>grokking</b>, and this page lets you watch it happen.</p>
  <p>As the machine practices, a hidden measure of its <i>confidence</i> on a new case keeps climbing even while
     its visible score sits frozen at 100%, and the exact instant that confidence tips over is the instant it
     starts to understand. You can also change which example is kept hidden from practice, and see why the machine
     sometimes learns the general rule, sometimes learns it instantly, and sometimes only memorizes and never
     really gets it.</p>
</header>
<main>
  <div class="hero card" id="hero"></div>
  <div class="portrait card" id="portrait"></div>
  <h2 style="max-width:1100px;margin:26px auto 10px;font-size:16px;">The three controls</h2>
  <div class="grid" id="controls"></div>
</main>
<section id="learn">
  <div class="learnintro">
    <h2>What am I looking at?</h2>
    <p>A guided tour, from zero. No math background needed, each step builds on the last, and every idea
       points back to the live demo above. Scroll on.</p>
  </div>

  <article class="lesson"><span class="num">1</span>
    <h3>What is a transformer?</h3>
    <p>A transformer is the kind of model behind today's AI language systems. Its one job is simple to state:
       <b>given a sequence so far, predict what comes next.</b> It does this in three moves:</p>
    <ol>
      <li><b>Embed</b>, turn each token (a word, a symbol) into a list of numbers, a <i>vector</i>, so the
          machine can do arithmetic with meaning.</li>
      <li><b>Transform</b>, let those vectors mix and update through a stack of layers. Each layer reads the
          running state and adds to it (the &ldquo;residual stream&rdquo;), so information accumulates.</li>
      <li><b>Unembed</b>, turn the final vector back into a score for every possible next token, and pick the
          top one.</li>
    </ol>
    <p>Feed the pick back in and repeat, and you get generation. That's the whole skeleton: <b>embed &rarr;
       transform &rarr; unembed &rarr; repeat.</b> A frontier model does this with billions of numbers you cannot
       read; the parts, though, are exactly these.</p>
  </article>

  <article class="lesson"><span class="num">2</span>
    <h3>What is a &ldquo;lowered&rdquo; transformer?</h3>
    <p><b>Representational lowering</b> means compiling a behavior down into the <i>smallest</i> representation that
       reproduces it, and keeping every part legible. The <b>minimal (lowered) transformer</b> here has the
       <i>same skeleton</i> as a frontier model, shrunk until you can see all of it: 6 tokens and one small weight
       matrix <span class="mono">M</span>. Each part lines up one-to-one:</p>
    <table class="map">
      <tr><th>Frontier transformer</th><th>This minimal transformer</th></tr>
      <tr><td>token &rarr; embedding vector</td><td><span class="mono">emb[x]</span>, a short list of integers</td></tr>
      <tr><td>a stack of layers updating a hidden state</td><td>one legible update <span class="mono">(I + M)</span> on the vector</td></tr>
      <tr><td>unembed &rarr; softmax &rarr; sample</td><td>score every token, take the <span class="mono">argmax</span></td></tr>
      <tr><td>billions of opaque weights</td><td>a 3&times;3 matrix you can read</td></tr>
    </table>
    <p>The step is: <span class="mono">next = argmax over v of  emb[v] &middot; (I + M) &middot; emb[x]</span>. Same
       architecture, same idea of depth, but <b>no hidden layers</b>. Nothing is buried; the mechanism is on
       the table.</p>
    <figure class="diagram">
      <svg viewBox="0 0 720 172" role="img" aria-label="the lowered-transformer step, end to end">
        <defs><marker id="a2" markerWidth="9" markerHeight="9" refX="6.5" refY="3" orient="auto">
          <path d="M0 0 L7 3 L0 6 z" fill="var(--mut)"/></marker></defs>
        <circle cx="42" cy="58" r="22" fill="var(--panel)" stroke="var(--line)"/>
        <text x="42" y="63" text-anchor="middle" fill="var(--ink)" font-size="15" font-family="ui-monospace,monospace">x</text>
        <text x="42" y="98" text-anchor="middle" fill="var(--mut)" font-size="11">token</text>
        <line x1="68" y1="58" x2="112" y2="58" stroke="var(--mut)" marker-end="url(#a2)"/>
        <rect x="116" y="36" width="118" height="44" rx="8" fill="var(--panel)" stroke="var(--line)"/>
        <text x="175" y="55" text-anchor="middle" fill="var(--ink)" font-size="13" font-family="ui-monospace,monospace">emb[x]</text>
        <text x="175" y="71" text-anchor="middle" fill="var(--mut)" font-size="11">vector</text>
        <line x1="238" y1="58" x2="282" y2="58" stroke="var(--mut)" marker-end="url(#a2)"/>
        <rect x="286" y="36" width="118" height="44" rx="8" fill="var(--panel)" stroke="var(--line)"/>
        <text x="345" y="55" text-anchor="middle" fill="var(--ink)" font-size="13" font-family="ui-monospace,monospace">(I + M)&middot;</text>
        <text x="345" y="71" text-anchor="middle" fill="var(--mut)" font-size="11">update</text>
        <line x1="408" y1="58" x2="452" y2="58" stroke="var(--mut)" marker-end="url(#a2)"/>
        <rect x="456" y="36" width="118" height="44" rx="8" fill="var(--panel)" stroke="var(--line)"/>
        <text x="515" y="54" text-anchor="middle" fill="var(--ink)" font-size="12">score every</text>
        <text x="515" y="70" text-anchor="middle" fill="var(--mut)" font-size="11">token</text>
        <line x1="578" y1="58" x2="620" y2="58" stroke="var(--mut)" marker-end="url(#a2)"/>
        <circle cx="668" cy="58" r="28" fill="var(--held)"/>
        <text x="668" y="55" text-anchor="middle" fill="#111" font-size="11" font-weight="700">argmax</text>
        <text x="668" y="69" text-anchor="middle" fill="#111" font-size="10">next token</text>
        <path d="M668 88 L668 144 L42 144 L42 82" fill="none" stroke="var(--mut)" stroke-dasharray="4 3" marker-end="url(#a2)"/>
        <text x="355" y="162" text-anchor="middle" fill="var(--mut)" font-size="11">repeat, feed the pick back in (this loop is the &ldquo;orbit&rdquo; of lesson 3)</text>
      </svg>
      <figcaption>The whole machine, left to right: a token becomes a vector, one legible update
        <span class="mono">(I+M)</span> transforms it, every token is scored, and the top-scoring one is emitted, then it repeats. A frontier transformer has this exact shape, with many opaque update layers in the
        middle instead of one readable matrix.</figcaption>
    </figure>
  </article>

  <article class="lesson"><span class="num">3</span>
    <h3>Generation is an &ldquo;orbit&rdquo;</h3>
    <p>Apply the step to its own output, over and over, and you trace a path through the space of tokens, a
       <b>trajectory</b>, or <i>orbit</i>. Because the map is a fixed, finite, deterministic function, every orbit
       eventually settles into a cycle. A resting point it never leaves is a <b>fixed-point attractor</b>; the set
       of starting tokens that flow into it is that attractor's <b>basin</b>. So a lowered transformer is a tiny
       <b>dynamical system</b>: seed it, and watch where it falls.</p>
    <figure class="diagram">
      <svg viewBox="0 0 720 150" role="img" aria-label="an orbit running through a transient into a fixed-point attractor">
        <defs><marker id="a3" markerWidth="9" markerHeight="9" refX="6.5" refY="3" orient="auto">
          <path d="M0 0 L7 3 L0 6 z" fill="var(--mut)"/></marker></defs>
        <circle cx="80" cy="62" r="24" fill="var(--panel)" stroke="var(--line)"/>
        <text x="80" y="67" text-anchor="middle" fill="var(--ink)" font-size="14" font-family="ui-monospace,monospace">s&#8320;</text>
        <line x1="106" y1="62" x2="200" y2="62" stroke="var(--mut)" marker-end="url(#a3)"/>
        <circle cx="228" cy="62" r="24" fill="var(--panel)" stroke="var(--line)"/>
        <text x="228" y="67" text-anchor="middle" fill="var(--ink)" font-size="14" font-family="ui-monospace,monospace">s&#8321;</text>
        <line x1="254" y1="62" x2="348" y2="62" stroke="var(--mut)" marker-end="url(#a3)"/>
        <circle cx="376" cy="62" r="24" fill="var(--panel)" stroke="var(--line)"/>
        <text x="376" y="67" text-anchor="middle" fill="var(--ink)" font-size="14" font-family="ui-monospace,monospace">s&#8322;</text>
        <line x1="402" y1="62" x2="520" y2="62" stroke="var(--mut)" marker-end="url(#a3)"/>
        <circle cx="556" cy="62" r="30" fill="var(--held)"/>
        <text x="556" y="67" text-anchor="middle" fill="#111" font-size="15" font-weight="700">A</text>
        <path d="M544 35 C526 -3 586 -3 568 35" fill="none" stroke="var(--held)" stroke-width="2" marker-end="url(#a3)"/>
        <line x1="66" y1="100" x2="402" y2="100" stroke="var(--mut)" opacity=".5"/>
        <text x="234" y="118" text-anchor="middle" fill="var(--mut)" font-size="11">transient (a one-time run-in)</text>
        <text x="556" y="118" text-anchor="middle" fill="var(--mut)" font-size="11">attractor</text>
        <text x="556" y="132" text-anchor="middle" fill="var(--mut)" font-size="11">(fixed point, it stays)</text>
      </svg>
      <figcaption>Start anywhere, follow the arrows: after a short run-in the orbit enters a cycle it never leaves.
        A <b>fixed point</b> (the loop on <span class="mono">A</span>) is a cycle of length one, a stable
        answer. Where you end up depends only on where you started: that is the attractor's basin.</figcaption>
    </figure>
  </article>

  <article class="lesson"><span class="num">4</span>
    <h3>The task: NAND, the universal gate</h3>
    <p>We teach this machine <b>NAND</b>, a logic gate that outputs 0 only when both inputs are 1, else 1.
       NAND matters because <b>everything a computer can compute can be built out of NAND gates alone.</b> We encode
       its truth table as a next-token map over six tokens: the four input rows <span class="mono">p,q,r,s</span> =
       (00, 01, 10, 11), plus the two output bits <span class="mono">O</span>=1 and <span class="mono">Z</span>=0
       (which are fixed points, they map to themselves). A correct weight makes each input row step in one
       move to its right answer.</p>
    <figure class="diagram">
      <svg viewBox="0 0 720 250" role="img" aria-label="the NAND phase portrait as two basins of attraction">
        <defs><marker id="a4" markerWidth="9" markerHeight="9" refX="6.5" refY="3" orient="auto">
          <path d="M0 0 L7 3 L0 6 z" fill="var(--mut)"/></marker></defs>
        <ellipse cx="235" cy="128" rx="205" ry="104" fill="var(--train)" opacity="0.09"/>
        <ellipse cx="235" cy="128" rx="205" ry="104" fill="none" stroke="var(--train)" stroke-dasharray="5 4" opacity="0.55"/>
        <text x="235" y="34" text-anchor="middle" fill="var(--train)" font-size="12.5" font-weight="700">basin of O, NAND = 1</text>
        <ellipse cx="580" cy="128" rx="118" ry="104" fill="var(--held)" opacity="0.10"/>
        <ellipse cx="580" cy="128" rx="118" ry="104" fill="none" stroke="var(--held)" stroke-dasharray="5 4" opacity="0.6"/>
        <text x="580" y="34" text-anchor="middle" fill="var(--held)" font-size="12.5" font-weight="700">basin of Z, NAND = 0</text>
        <g font-family="ui-monospace,monospace" font-size="13">
          <circle cx="120" cy="80" r="18" fill="var(--panel)" stroke="var(--line)"/><text x="120" y="85" text-anchor="middle" fill="var(--ink)">p</text>
          <circle cx="120" cy="128" r="18" fill="var(--panel)" stroke="var(--line)"/><text x="120" y="133" text-anchor="middle" fill="var(--ink)">q</text>
          <circle cx="120" cy="176" r="18" fill="var(--panel)" stroke="var(--line)"/><text x="120" y="181" text-anchor="middle" fill="var(--ink)">r</text>
          <line x1="140" y1="86" x2="300" y2="120" stroke="var(--mut)" marker-end="url(#a4)"/>
          <line x1="140" y1="128" x2="298" y2="128" stroke="var(--mut)" marker-end="url(#a4)"/>
          <line x1="140" y1="170" x2="300" y2="136" stroke="var(--mut)" marker-end="url(#a4)"/>
          <circle cx="332" cy="128" r="30" fill="var(--train)"/><text x="332" y="133" text-anchor="middle" fill="#111" font-weight="700" font-size="15">O</text>
          <path d="M313 110 C304 74 360 74 351 110" fill="none" stroke="var(--train)" stroke-width="2" marker-end="url(#a4)"/>
          <circle cx="512" cy="128" r="18" fill="var(--panel)" stroke="var(--line)"/><text x="512" y="133" text-anchor="middle" fill="var(--ink)">s</text>
          <line x1="532" y1="128" x2="600" y2="128" stroke="var(--mut)" marker-end="url(#a4)"/>
          <circle cx="632" cy="128" r="30" fill="var(--held)"/><text x="632" y="133" text-anchor="middle" fill="#111" font-weight="700" font-size="15">Z</text>
          <path d="M613 110 C604 74 660 74 651 110" fill="none" stroke="var(--held)" stroke-width="2" marker-end="url(#a4)"/>
        </g>
      </svg>
      <figcaption>The learned map, drawn as a picture: three input rows (<span class="mono">p, q, r</span>) flow to
        the output <b>1</b>, and the single row <span class="mono">s</span> flows to <b>0</b>. The two outputs are
        fixed points (the little self-loops). This 3-to-1 split of the basins <i>is</i> the NAND truth table.</figcaption>
    </figure>
    <div class="callout"><b>In the demo:</b> when it has learned NAND, the phase portrait reads
       <span class="mono">O&larr;Opqr | Z&larr;Zs</span>, three input rows fall into &ldquo;1&rdquo;, one row
       (<span class="mono">s</span>) falls into &ldquo;0&rdquo;. <b>The basins of attraction <i>are</i> the truth
       table.</b> The 3-to-1 split is NAND.</p>
  </article>

  <article class="lesson"><span class="num">5</span>
    <h3>What is training?</h3>
    <p>We don't hand-set the weight. We <b>learn</b> it from examples by <b>gradient descent</b>: show the machine
       the rows, measure how wrong it is, and nudge every number in <span class="mono">M</span> a little in the
       direction that reduces the error. Repeat thousands of times. Starting from all-zeros, the weight slowly grows
       into one that gets the training rows right.</p>
  </article>

  <article class="lesson"><span class="num">6</span>
    <h3>What is grokking?</h3>
    <p>Here is the strange part. A model can fit its <i>training</i> data quickly, then sit on a long flat plateau
       where nothing visible improves, and only <i>much later</i> suddenly start getting <b>held-out</b>
       (never-trained) cases right. That delayed click is <b>grokking</b>.</p>
    <figure class="diagram">
      <svg viewBox="0 0 720 200" role="img" aria-label="the grokking gap: train fits early, held-out generalizes late">
        <rect x="210" y="28" width="141" height="128" fill="var(--ok)" opacity="0.08"/>
        <line x1="50" y1="30" x2="690" y2="30" stroke="var(--grid)"/>
        <line x1="50" y1="93" x2="690" y2="93" stroke="var(--grid)"/>
        <line x1="50" y1="156" x2="690" y2="156" stroke="var(--grid)"/>
        <text x="44" y="34" text-anchor="end" fill="var(--mut)" font-size="10">100%</text>
        <text x="44" y="160" text-anchor="end" fill="var(--mut)" font-size="10">0%</text>
        <line x1="210" y1="28" x2="210" y2="156" stroke="var(--train)" stroke-dasharray="3 3" opacity="0.6"/>
        <text x="214" y="40" fill="var(--train)" font-size="10">train fits</text>
        <line x1="351" y1="28" x2="351" y2="156" stroke="var(--ok)" stroke-dasharray="3 3" opacity="0.7"/>
        <text x="355" y="52" fill="var(--ok)" font-size="10">held-out groks</text>
        <path d="M50 156 Q150 30 210 30 L690 30" fill="none" stroke="var(--train)" stroke-width="2.5"/>
        <path d="M50 156 L345 156 C349 156 349 30 351 30 L690 30" fill="none" stroke="var(--held)" stroke-width="2.5"/>
        <text x="50" y="178" fill="var(--mut)" font-size="10">step 0</text>
        <text x="690" y="178" text-anchor="end" fill="var(--mut)" font-size="10">training steps &rarr;</text>
        <g font-size="11">
          <rect x="250" y="168" width="10" height="10" rx="2" fill="var(--train)"/><text x="266" y="177" fill="var(--mut)">train accuracy</text>
          <rect x="400" y="168" width="10" height="10" rx="2" fill="var(--held)"/><text x="416" y="177" fill="var(--mut)">held-out accuracy</text>
        </g>
      </svg>
      <figcaption>Train accuracy (blue) shoots to 100% early and flatlines. Held-out accuracy (pink) sits at 0
        through the shaded plateau, then jumps to 100% much later. That horizontal gap between fitting and
        generalizing is grokking. (Schematic; scroll up for the live curves.)</figcaption>
    </figure>
    <div class="callout"><b>In the demo:</b> we hold out one row, <span class="mono">s</span>. Training fits the
       other rows by <b>step 50</b>, but <span class="mono">s</span> only becomes correct around <b>step 94</b>.
       Drag the slider through that gap, train accuracy is pinned at 100% the whole time, yet something is
       clearly still happening.</p>
  </article>

  <article class="lesson"><span class="num">7</span>
    <h3>Why the delay? (the plateau isn't idle)</h3>
    <p>The flat stretch only <i>looks</i> idle. Watch the <b>held-out margin</b>, how confidently the machine
       gets <span class="mono">s</span> right (negative = wrong, positive = right). At the fit it is
       <b>&minus;0.9</b> (confidently wrong); across the plateau it climbs steadily; and it <b>crosses zero at the
       exact step grokking happens.</b> Meanwhile the weight's size keeps growing. The model is still learning on
       the plateau, sharpening its margin, not resting. The late jump is a smooth threshold crossing, not a
       lucky accident.</p>
    <figure class="diagram">
      <svg viewBox="0 0 720 200" role="img" aria-label="the held-out margin climbing across the plateau and crossing zero at the grok">
        <rect x="210" y="24" width="141" height="140" fill="var(--ok)" opacity="0.08"/>
        <line x1="50" y1="94" x2="690" y2="94" stroke="var(--mut)" stroke-dasharray="4 3"/>
        <text x="44" y="98" text-anchor="end" fill="var(--mut)" font-size="10">0</text>
        <text x="56" y="40" fill="var(--mut)" font-size="10">margin &gt; 0 &rarr; correct</text>
        <text x="56" y="158" fill="var(--mut)" font-size="10">margin &lt; 0 &rarr; wrong</text>
        <path d="M50 150 C170 132 210 118 351 94 C470 74 600 56 690 46" fill="none" stroke="var(--mut)" stroke-width="1.5" stroke-dasharray="5 4"/>
        <text x="614" y="44" fill="var(--mut)" font-size="10">&#8214;M&#8214; rising</text>
        <path d="M50 150 C150 146 205 132 351 94 C470 66 600 50 690 40" fill="none" stroke="var(--held)" stroke-width="2.5"/>
        <circle cx="351" cy="94" r="5" fill="var(--ok)"/>
        <line x1="351" y1="24" x2="351" y2="164" stroke="var(--ok)" stroke-dasharray="3 3" opacity="0.7"/>
        <text x="357" y="112" fill="var(--ok)" font-size="10.5">crosses 0 = grok</text>
        <text x="214" y="180" fill="var(--mut)" font-size="10">plateau (train already 100%)</text>
      </svg>
      <figcaption>The held-out margin (pink) starts negative, confidently wrong, climbs steadily
        across the plateau, and crosses zero at the exact step it groks, while the weight size
        <span class="mono">&#8214;M&#8214;</span> (dashed) keeps rising. The flat region isn't idle; it's the model
        sharpening until the answer tips over.</figcaption>
    </figure>
    <div class="callout"><b>In the demo:</b> the right-hand chart is the margin; scrub to the shaded band and
       watch the phase portrait <i>flip</i>: <span class="mono">s</span> starts as its own dead-end
       (<span class="mono">s&larr;s</span>), and at the crossing its basin is swallowed into
       <span class="mono">Z</span> (<span class="mono">Z&larr;Zs</span>). That reorganization <i>is</i> the grok.</p>
  </article>

  <article class="lesson"><span class="num">8</span>
    <h3>Why does it generalize at all?</h3>
    <p>The held-out row was never trained, so why does the machine ever get it right? Because of <b>implicit
       bias</b>. Among all the weights that fit the training rows, gradient descent quietly drifts toward the
       <i>simplest</i> one (smallest, largest-margin). And that simplest solution happens to also get the held-out
       row right. Generalization is the bias <b>choosing which attractor</b> an undetermined case falls into.</p>
    <figure class="diagram">
      <svg viewBox="0 0 720 200" role="img" aria-label="the analytic prior and the trained endpoint are the same weight">
        <defs><marker id="a8" markerWidth="9" markerHeight="9" refX="6.5" refY="3" orient="auto">
          <path d="M0 0 L7 3 L0 6 z" fill="var(--mut)"/></marker></defs>
        <rect x="40" y="28" width="250" height="52" rx="8" fill="var(--panel)" stroke="var(--line)"/>
        <text x="165" y="50" text-anchor="middle" fill="var(--ink)" font-size="13" font-weight="600">min-L1 prior</text>
        <text x="165" y="68" text-anchor="middle" fill="var(--mut)" font-size="11">the simplest weight, computed directly</text>
        <rect x="430" y="28" width="250" height="52" rx="8" fill="var(--panel)" stroke="var(--line)"/>
        <text x="555" y="50" text-anchor="middle" fill="var(--ink)" font-size="13" font-weight="600">SGD endpoint</text>
        <text x="555" y="68" text-anchor="middle" fill="var(--mut)" font-size="11">where grokking lands, by descent</text>
        <path d="M165 82 C165 116 300 118 348 128" fill="none" stroke="var(--mut)" marker-end="url(#a8)"/>
        <path d="M555 82 C555 116 420 118 372 128" fill="none" stroke="var(--mut)" marker-end="url(#a8)"/>
        <text x="360" y="118" text-anchor="middle" fill="var(--ok)" font-size="22" font-weight="700">=</text>
        <rect x="220" y="132" width="280" height="50" rx="8" fill="var(--ok)" opacity="0.10"/>
        <rect x="220" y="132" width="280" height="50" rx="8" fill="none" stroke="var(--ok)" opacity="0.6"/>
        <text x="360" y="156" text-anchor="middle" fill="var(--ink)" font-size="14" font-family="ui-monospace,monospace">O&larr;Opqr | Z&larr;Zs</text>
        <text x="360" y="173" text-anchor="middle" fill="var(--ok)" font-size="11">the same phase portrait</text>
      </svg>
      <figcaption>Two roads, one destination: the simplest weight worked out analytically (left) and the weight
        gradient descent actually reaches (right) produce the <i>identical</i> phase portrait. The static
        implicit-bias prior and the dynamic grok are literally the same object, <span class="mono">endpoint == prior</span> in the demo.</figcaption>
    </figure>
    <div class="callout"><b>We can check this exactly:</b> compute the simplest weight analytically (the
       &ldquo;min-L1 prior&rdquo;). It predicts <span class="mono">s&rarr;Z</span>. The trained network converges to
       the <i>very same phase portrait</i>. <b>The static prior and the dynamic grok are the same object</b>, shown in the demo as <span class="mono">endpoint == prior</span>.</p>
  </article>

  <article class="lesson"><span class="num">9</span>
    <h3>How do we know it's real, not luck?</h3>
    <p>One curve rising late proves little. What makes it a claim is the <b>three controls</b>, same machine,
       same training, differing only in whether the training rows <i>force</i> the held-out answer:</p>
    <ul class="controls-list">
      <li><b class="c-grok">GROKS</b>, hold out <span class="mono">s</span> (not forced): generalizes
          <i>late</i>.</li>
      <li><b class="c-no">NO-GROK</b>, hold out a redundant row (forced by the others): generalizes
          <i>immediately</i>.</li>
      <li><b class="c-mem">MEMORIZES</b>, hold out all three &ldquo;1&rdquo; rows (nothing left to force
          them): <i>never</i> generalizes.</li>
    </ul>
    <figure class="diagram">
      <svg viewBox="0 0 720 176" role="img" aria-label="the three controls: grok, no-grok, memorize">
        <g>
          <text x="120" y="22" text-anchor="middle" fill="var(--ok)" font-size="12.5" font-weight="700">GROKS</text>
          <line x1="30" y1="128" x2="210" y2="128" stroke="var(--grid)"/>
          <path d="M30 128 Q60 44 82 44 L210 44" fill="none" stroke="var(--train)" stroke-width="1.5" opacity="0.55"/>
          <rect x="96" y="40" width="40" height="92" fill="var(--ok)" opacity="0.08"/>
          <path d="M30 128 L118 128 C122 128 122 44 126 44 L210 44" fill="none" stroke="var(--held)" stroke-width="2.5"/>
          <text x="120" y="150" text-anchor="middle" fill="var(--mut)" font-size="10.5">hold out s (not forced)</text>
          <text x="120" y="165" text-anchor="middle" fill="var(--mut)" font-size="10.5">&rarr; generalizes late</text>
        </g>
        <g>
          <text x="360" y="22" text-anchor="middle" fill="var(--train)" font-size="12.5" font-weight="700">NO-GROK</text>
          <line x1="270" y1="128" x2="450" y2="128" stroke="var(--grid)"/>
          <path d="M270 128 Q300 44 322 44 L450 44" fill="none" stroke="var(--train)" stroke-width="1.5" opacity="0.55"/>
          <path d="M270 128 C288 128 300 44 320 44 L450 44" fill="none" stroke="var(--held)" stroke-width="2.5"/>
          <text x="360" y="150" text-anchor="middle" fill="var(--mut)" font-size="10.5">hold out a forced row</text>
          <text x="360" y="165" text-anchor="middle" fill="var(--mut)" font-size="10.5">&rarr; generalizes at once</text>
        </g>
        <g>
          <text x="600" y="22" text-anchor="middle" fill="var(--bad)" font-size="12.5" font-weight="700">MEMORIZES</text>
          <line x1="510" y1="128" x2="690" y2="128" stroke="var(--grid)"/>
          <path d="M510 128 Q540 44 562 44 L690 44" fill="none" stroke="var(--train)" stroke-width="1.5" opacity="0.55"/>
          <path d="M510 128 L690 128" fill="none" stroke="var(--held)" stroke-width="2.5"/>
          <text x="600" y="150" text-anchor="middle" fill="var(--mut)" font-size="10.5">hold out all &ldquo;1&rdquo; rows</text>
          <text x="600" y="165" text-anchor="middle" fill="var(--mut)" font-size="10.5">&rarr; never generalizes</text>
        </g>
      </svg>
      <figcaption>Same machine, same training, only the held-out choice changes. Faint blue is train
        accuracy (always fits); pink is held-out accuracy. Whether the held-out row is <i>forced</i> by the
        rest decides everything: late (grok), immediate (no-grok), or never (memorize).</figcaption>
    </figure>
    <p>Grokking happens in exactly one case: when the data doesn't force the answer but the implicit bias still
       points at it. And it isn't a fluke of the starting point, press <b>&#8635; train again</b> and it
       groks every time from a fresh random start, always landing on the same answer.</p>
  </article>

  <article class="lesson"><span class="num">10</span>
    <h3>What value does this have for machine learning?</h3>
    <p>Grokking, generalization, and implicit bias are debated in models far too large to inspect. This minimal
       transformer is a <b>microscope</b> for the same phenomena:</p>
    <ul>
      <li><b>Fully interpretable.</b> Every weight and every intermediate is legible. You <i>watch</i> the
          mechanism instead of guessing, the opposite of reverse-engineering a billion-parameter net.</li>
      <li><b>A controlled lab for grokking.</b> The exact effect seen in big models, reduced to something you can
          compute end-to-end, falsify, and replay.</li>
      <li><b>It bridges theory and practice.</b> The analytic implicit-bias prior and the real SGD run land on the
          same attractor, theory and dynamics, shown to be one thing.</li>
      <li><b>It teaches what &ldquo;it works&rdquo; means.</b> Generalization is not magic: it is the implicit bias
          selecting an attractor for a case the data left open. Here you can see that, step by step.</li>
    </ul>
    <figure class="diagram">
      <svg viewBox="0 0 720 200" role="img" aria-label="the minimal transformer as a microscope on a frontier model">
        <defs><marker id="a10" markerWidth="9" markerHeight="9" refX="6.5" refY="3" orient="auto">
          <path d="M0 0 L7 3 L0 6 z" fill="var(--mut)"/></marker></defs>
        <rect x="34" y="46" width="250" height="108" rx="12" fill="var(--ink)" opacity="0.10"/>
        <rect x="34" y="46" width="250" height="108" rx="12" fill="none" stroke="var(--line)" stroke-dasharray="5 4"/>
        <text x="159" y="38" text-anchor="middle" fill="var(--mut)" font-size="12" font-weight="600">frontier transformer</text>
        <g fill="var(--mut)" font-size="17" opacity="0.55" text-anchor="middle" font-family="ui-monospace,monospace">
          <text x="80" y="86">?</text><text x="130" y="112">?</text><text x="185" y="80">?</text>
          <text x="230" y="118">?</text><text x="110" y="140">?</text><text x="205" y="140">?</text><text x="159" y="98">?</text>
        </g>
        <text x="159" y="172" text-anchor="middle" fill="var(--mut)" font-size="11">billions of weights, opaque</text>
        <line x1="300" y1="100" x2="392" y2="100" stroke="var(--mut)" marker-end="url(#a10)"/>
        <text x="346" y="90" text-anchor="middle" fill="var(--mut)" font-size="10.5">lower</text>
        <text x="346" y="116" text-anchor="middle" fill="var(--mut)" font-size="10.5">&amp; make legible</text>
        <g>
          <rect x="470" y="62" width="120" height="76" rx="8" fill="var(--panel)" stroke="var(--line)"/>
          <g stroke="var(--line)"><line x1="510" y1="62" x2="510" y2="138"/><line x1="550" y1="62" x2="550" y2="138"/>
            <line x1="470" y1="87" x2="590" y2="87"/><line x1="470" y1="113" x2="590" y2="113"/></g>
          <g fill="var(--train)" opacity="0.8"><rect x="474" y="66" width="32" height="17"/><rect x="554" y="117" width="32" height="17"/></g>
          <g fill="var(--held)" opacity="0.8"><rect x="514" y="91" width="32" height="17"/></g>
          <text x="530" y="156" text-anchor="middle" fill="var(--ink)" font-size="12" font-weight="600">minimal transformer</text>
          <text x="530" y="172" text-anchor="middle" fill="var(--mut)" font-size="11">every weight legible</text>
        </g>
        <circle cx="560" cy="100" r="46" fill="none" stroke="var(--ok)" stroke-width="3" opacity="0.85"/>
        <line x1="593" y1="133" x2="622" y2="162" stroke="var(--ok)" stroke-width="5" opacity="0.85" stroke-linecap="round"/>
      </svg>
      <figcaption>Same phenomena, two ways to look. A frontier model hides its mechanism in billions of weights you
        can only probe at; the minimal transformer shrinks the <i>same</i> architecture until every weight and
        every step is on the table, a microscope for grokking, generalization, and implicit bias.</figcaption>
    </figure>
    <p class="closer">Same architecture as a frontier transformer, small enough to hold in your head, and
       grokking, made visible.</p>
    <a href="#top" class="learnbtn" onclick="window.scrollTo({top:0,behavior:'smooth'});return false;">&uarr; Back to the demo</a>
  </article>
</section>

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

function cursor(step,maxStep,id){
  const x=(step==null)?PAD.l:xpos(step,maxStep);
  return `<line id="${id}" x1="${x}" y1="${PAD.t}" x2="${x}" y2="${H-PAD.b}" stroke="var(--ink)" stroke-width="1.5" opacity=".85"/>`;
}
function accuracyChart(reg, big, curStep){
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
    ${shade}${yt}${marks}${big?cursor(curStep,maxStep,'accCursor'):''}
    <path d="${path(tr)}" fill="none" stroke="var(--train)" stroke-width="2"/>
    <path d="${path(hd)}" fill="none" stroke="var(--held)" stroke-width="2"/>
    <text x="${PAD.l}" y="${H-6}" fill="var(--mut)" font-size="10">step 0</text>
    <text x="${W-PAD.r}" y="${H-6}" fill="var(--mut)" font-size="10" text-anchor="end">${maxStep}</text>
  </svg>`;
}

function marginChart(reg, curStep){
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
    ${shade}${cursor(curStep,maxStep,'marCursor')}
    <line x1="${PAD.l}" y1="${zeroY}" x2="${W-PAD.r}" y2="${zeroY}" stroke="var(--mut)" stroke-dasharray="4 3"/>
    <text x="${PAD.l-6}" y="${zeroY+3}" fill="var(--mut)" font-size="10" text-anchor="end">0</text>
    <path d="${path(mg)}" fill="none" stroke="var(--held)" stroke-width="2"/>
    ${cross}
    <text x="${PAD.l}" y="${PAD.t+2}" fill="var(--mut)" font-size="10">held-out margin (logit gap)</text>
  </svg>`;
}

function gl(text, anchor){
  return `<a class="term" href="/explain#${anchor}" target="_blank" rel="noopener">${text}</a>`;
}
function learnLink(href,label){
  return `<a class="learnbtn sm" href="${href}" target="_blank" rel="noopener">${label} &#8599;</a>`;
}
function heroFacts(reg){
  const mFit = reg.points.find(p=>p.step===reg.fit_at);
  if(reg.verdict==="GROKS"){
    const gap = reg.grok_at-reg.fit_at;
    return `train hits 100% at <b>step ${reg.fit_at}</b> &rarr; held-out only at <b>step ${reg.grok_at}</b>
      (a <b>${gap}-step</b> ${gl('grok','groks')} gap). Held out <b>{${reg.holdout.join(',')}}</b>, which the other
      rows do <b>not</b> force, yet the ${gl('implicit bias','implicit-bias')} generalizes it. At the fit the
      held-out margin is <b>${mFit?mFit.margin.toFixed(2):'?'}</b> (wrong); it climbs across the shaded plateau and
      the zero-crossing <b>is</b> the grok.`;
  }
  if(reg.verdict==="NO-GROK"){
    return `held out <b>{${reg.holdout.join(',')}}</b>, a row the others already <b>force</b>. So held-out
      generalizes at <b>step ${reg.grok_at}</b>, as soon as (here before) train even fits at
      <b>step ${reg.fit_at}</b>. No plateau, no ${gl('implicit-bias','implicit-bias')} phase,
      <b>no ${gl('grok','groks')} gap</b>.`;
  }
  return `held out <b>{${reg.holdout.join(',')}}</b>, every &ldquo;1&rdquo; row, so nothing is left to
    <b>force</b> them. Train still fits at <b>step ${reg.fit_at}</b>, but held-out <b>never</b> reaches 100%:
    with no forcing and no ${gl('implicit-bias','implicit-bias')} target, gradient descent just
    ${gl('memorizes','memorize')} the training rows.`;
}

let HERO=null, IDX=0, PLAY=null, MAXSTEP=1;

function buildHeroCharts(){
  const reg=HERO; MAXSTEP=reg.points[reg.points.length-1].step;
  // built ONCE; scrubbing only moves the cursor lines, never re-parses the SVG
  document.getElementById('heroCharts').innerHTML =
    `<div>${accuracyChart(reg,true,reg.points[0].step)}</div><div>${marginChart(reg,reg.points[0].step)}</div>`;
}
function moveCursor(id,step){
  const el=document.getElementById(id); if(!el) return;
  const x=xpos(step,MAXSTEP).toFixed(1); el.setAttribute('x1',x); el.setAttribute('x2',x);
}
function wcolor(v,wmax){
  const a=Math.min(1,Math.abs(v)/(wmax||1));
  return v<0 ? `rgba(74,163,255,${a.toFixed(3)})` : `rgba(255,122,194,${a.toFixed(3)})`;
}
function weightsHTML(p,wmax){
  const cells = p.M.map(row=>row.map(v=>
    `<div class="wcell" style="background:${wcolor(v,wmax)}">${v>=0?'+':''}${v.toFixed(2)}</div>`
  ).join('')).join('');
  return `<div class="wgrid-wrap">
    <div>
      <div class="wtitle">the model's weights, learning live <span class="mono">(M)</span></div>
      <div class="wgrid">${cells}</div>
    </div>
    <div class="wcap">These nine numbers <b>are</b> the whole model. They start at <b>zero</b> and grow as it
      learns (blue = negative, pink = positive, brighter = larger). The readout uses <span class="mono">(I + M)</span>,
      so the diagonal effectively adds 1. When the numbers stop moving, the phase portrait is locked in.</div>
  </div>`;
}
function drawHero(){
  const reg=HERO, p=reg.points[IDX], correct=p.held_acc===1;
  moveCursor('accCursor',p.step); moveCursor('marCursor',p.step);
  const predRows = p.held_pred.map(([s,pred,t])=>{
    const good = pred===t;
    return `<span class="mono">${s} &rarr; <b class="${good?'match':'nomatch'}">${esc(pred)}</b></span>
            <span class="mut">(want ${esc(t)})</span>`;
  }).join(' &nbsp; ');
  document.getElementById('state').innerHTML =
    `<div class="staterow"><span>step</span><b class="mono">${p.step}</b></div>
     <div class="staterow"><span>train acc</span><b class="mono">${(p.train_acc*100).toFixed(0)}%</b></div>
     <div class="staterow"><span>held-out acc</span><b class="mono ${correct?'match':'nomatch'}">${(p.held_acc*100).toFixed(0)}%</b></div>
     <div class="staterow"><span>held margin</span><b class="mono ${p.margin>=0?'match':'nomatch'}">${p.margin>=0?'+':''}${p.margin.toFixed(2)}</b></div>
     <div class="staterow"><span>&#8214;M&#8214;</span><b class="mono">${p.wnorm.toFixed(2)}</b></div>
     <div class="staterow wide"><span>held map</span><span>${predRows}</span></div>
     <div class="staterow wide"><span>portrait</span><b class="mono ${p.portrait===reg.prior_portrait?'match':''}">${esc(p.portrait)}</b></div>`;
  let phase='before the fit';
  if(reg.fit_at!=null && p.step>=reg.fit_at){
    if(reg.grok_at!=null && p.step>=reg.grok_at)
      phase = (reg.verdict==="GROKS") ? 'GROKKED, held-out generalized' : 'GENERALIZED, held-out correct';
    else if(reg.grok_at==null)
      phase = 'MEMORIZING, train fit, held-out never generalizes';
    else
      phase = 'ON THE PLATEAU, train fit, margin still climbing';
  }
  document.getElementById('phase').innerHTML = phase;
  document.getElementById('weights').innerHTML = weightsHTML(p, HERO.wmax);
  document.getElementById('scrub').value = IDX;
}
function setIdx(i){ IDX=Math.max(0,Math.min(HERO.points.length-1,i)); drawHero(); }
function animate(){
  // sweep the whole trajectory in ~2.5s regardless of step count, so you watch it train
  if(PLAY) clearInterval(PLAY);
  IDX=0; drawHero();
  const N=HERO.points.length, inc=Math.max(1,Math.floor(N/150));
  PLAY=setInterval(()=>{ if(IDX>=N-1){ clearInterval(PLAY); PLAY=null; return; } setIdx(IDX+inc); }, 16);
}
function redo(){
  const btn=document.getElementById('redo');
  btn.disabled=true; btn.textContent='↻ training…';
  const seed=Math.floor(Math.random()*1e9);
  fetch('/api/data?seed='+seed).then(r=>r.json()).then(data=>{
    render(data); btn.disabled=false; btn.textContent='↻ train again';
    animate();
  }).catch(e=>{ btn.disabled=false; btn.textContent='↻ train again'; });
}

let DATA=null, CURKEY='groks';

function portraitCard(reg){
  const m = reg.portrait_match;
  const body = m
    ? `Trained holding out <b>{${reg.holdout.join(',')}}</b>, the SGD endpoint's ${gl('phase portrait','portrait')}
       equals the analytic min-L1 prior, the dynamic result and the static ${gl('implicit-bias','implicit-bias')}
       prior are the same object.`
    : `Holding out every &ldquo;1&rdquo; row, SGD ${gl('memorizes','memorize')}: its endpoint does <b>not</b> reach
       the prior ${gl('portrait','portrait')}, because nothing forced it there. The identity holds only when it
       generalizes.`;
  return `<h2>The endpoint vs. the prior <span class="tag ${m?'GROKS':'MEMORIZES'}">${m?'MATCH':'MISMATCH'}</span></h2>
     <div class="facts">${body}<br>
       SGD endpoint: <span class="mono ${m?'match':'nomatch'}">${esc(reg.endpoint_portrait)}</span><br>
       min-L1 prior: <span class="mono ${m?'match':'nomatch'}">${esc(reg.prior_portrait)}</span></div>
     <div class="pnote">Reading it: <span class="mono">X&lt;-Yz</span> means tokens <b>Y</b> and <b>z</b> both settle
       on <b>X</b>. So <span class="mono">O&lt;-Opqr</span> = p, q, r all land on the bit 1.</div>`;
}
function selectRegime(key){
  const reg = DATA.regimes.find(r=>r.key===key) || DATA.regimes[0];
  HERO=reg; IDX=0; CURKEY=reg.key;
  const tag=document.getElementById('heroTag'); tag.className='tag '+clsMap[reg.verdict]; tag.textContent=reg.verdict;
  document.getElementById('scrub').max = reg.points.length-1;
  buildHeroCharts();
  drawHero();
  const ms=reg.train_ms, slow=Math.max(1,Math.round(2500/ms));
  document.getElementById('timenote').innerHTML =
    `&#9201; this run actually learned in about <b>${ms} ms</b> of real compute (${reg.steps} training steps). `
    + `You're watching it in slow motion: the animation stretches that to ~2.5 s, roughly <b>${slow}&times;</b> slower, so the change is visible.`;
  document.getElementById('facts').innerHTML = heroFacts(reg)
    + learnLink('/explain#'+reg.key, 'What does this mean?');
  document.getElementById('portrait').innerHTML = portraitCard(reg)
    + learnLink('/explain#identity', 'What does this mean?');
  document.querySelectorAll('#seg .segbtn').forEach(b=>b.classList.toggle('on', b.dataset.k===reg.key));
  document.querySelectorAll('#controls .card').forEach(c=>c.classList.toggle('sel', c.dataset.k===reg.key));
}

function render(data){
  DATA=data;
  document.getElementById('hero').innerHTML =
    `<h2>Explore the three controls <span id="heroTag" class="tag"></span></h2>
     <p class="herohint">Same machine, same training, pick what to hold out and watch the outcome change.
       Then <b>drag</b> to step through training, or <b>&#8635; train again</b> from a random start.</p>
     <div class="seg" id="seg">
       <button class="segbtn" data-k="groks">GROKS</button>
       <button class="segbtn" data-k="nogrok">NO-GROK</button>
       <button class="segbtn" data-k="memorize">MEMORIZES</button>
     </div>
     <div class="legend"><span><span class="sw" style="background:var(--train)"></span>train accuracy</span>
       <span><span class="sw" style="background:var(--held)"></span>held-out accuracy / margin</span>
       <span><span class="sw" style="background:var(--ok);opacity:.5"></span>plateau</span></div>
     <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:8px" class="heropair" id="heroCharts"></div>
     <div class="scrubwrap">
       <button id="redo" class="btn" title="reinitialize and train the NAND transformer again from a fresh random start">&#8635; train again</button>
       <input type="range" id="scrub" min="0" max="400" value="0" step="1">
       <span id="phase" class="phase"></span>
     </div>
     <div class="initline mut">init: <b>${data.init}</b>${data.seed!=null?' &middot; seed '+data.seed:''} &middot; drag to step one training step at a time</div>
     <div class="timenote mut" id="timenote"></div>
     <div class="state" id="state"></div>
     <div class="weights" id="weights"></div>
     <div id="facts" class="facts"></div>`;
  document.getElementById('scrub').addEventListener('input', e=>{ if(PLAY){clearInterval(PLAY);PLAY=null;} setIdx(+e.target.value); });
  document.getElementById('redo').addEventListener('click', redo);
  document.getElementById('seg').addEventListener('click', e=>{ const b=e.target.closest('.segbtn'); if(b) selectRegime(b.dataset.k); });

  document.getElementById('controls').innerHTML = data.regimes.map(r=>`
    <div class="card" data-k="${r.key}" title="show this control in the explorer above">
      <h2>${r.label} <span class="tag ${clsMap[r.verdict]}">${r.verdict}</span></h2>
      <div class="blurb">${r.blurb}</div>
      ${accuracyChart(r,false)}
      <div class="facts">holdout <b>{${r.holdout.join(',')}}</b> &middot; fit <b>${r.fit_at}</b> &middot;
        grok <b>${r.grok_at==null?'never':r.grok_at}</b></div>
    </div>`).join('');
  document.getElementById('controls').addEventListener('click', e=>{
    const c=e.target.closest('.card'); if(!c) return;
    selectRegime(c.dataset.k);
    document.getElementById('hero').scrollIntoView({behavior:'smooth', block:'start'});
  });

  selectRegime(CURKEY);
}

fetch('/api/data').then(r=>r.json()).then(render)
  .catch(e=>{document.getElementById('hero').textContent='error: '+e;});

// smooth-scroll to the tutorial, and progressively reveal each lesson on scroll
document.getElementById('learnbtn').addEventListener('click', e=>{
  e.preventDefault();
  document.getElementById('learn').scrollIntoView({behavior:'smooth', block:'start'});
});
const io = new IntersectionObserver(entries=>{
  entries.forEach(en=>{ if(en.isIntersecting){ en.target.classList.add('in'); io.unobserve(en.target); } });
}, {threshold:0.15});
document.querySelectorAll('.lesson').forEach(el=>io.observe(el));
</script>
</body></html>
"""


PAGE_EXPLAIN = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>In depth: what the demo means</title>
<style>
  :root { color-scheme: light dark; --bg:#0f1216; --panel:#171b22; --ink:#e6e9ef; --mut:#8b95a6;
          --train:#4aa3ff; --held:#ff7ac2; --grid:#2a3140; --ok:#39d98a; --bad:#ff6b6b; --line:#232a35; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--ink);
         font:15px/1.65 ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }
  a { color:var(--train); }
  .wrap { max-width:820px; margin:0 auto; padding:20px; }
  .backbar { display:flex; justify-content:space-between; align-items:center; padding:16px 0 8px; gap:12px; }
  .backbtn { background:var(--panel); border:1px solid var(--line); color:var(--ink); border-radius:999px;
             padding:8px 15px; font:600 13px system-ui; text-decoration:none; white-space:nowrap; }
  .backbtn:hover { border-color:var(--held); }
  h1 { font-size:27px; margin:12px 0 6px; }
  .lede { color:var(--mut); font-size:15.5px; max-width:66ch; }
  .toc { display:flex; flex-wrap:wrap; gap:8px; margin:18px 0 8px; }
  .toc a { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:6px 11px;
           font-size:12.5px; text-decoration:none; color:var(--mut); }
  .toc a:hover { color:var(--ink); border-color:var(--held); }
  section.topic { border-top:1px solid var(--line); padding-top:26px; margin-top:34px; scroll-margin-top:16px; }
  section.topic > h2 { font-size:21px; margin:0 0 4px; display:flex; align-items:center; gap:10px; flex-wrap:wrap; }
  .tag { font:600 11px ui-monospace, monospace; padding:2px 8px; border-radius:999px; }
  .tag.GROKS{ background:rgba(57,217,138,.15); color:var(--ok); }
  .tag.NOGROK{ background:rgba(74,163,255,.15); color:var(--train); }
  .tag.MEMORIZES{ background:rgba(255,107,107,.15); color:var(--bad); }
  .tag.ID{ background:rgba(255,122,194,.15); color:var(--held); }
  .subtle { color:var(--mut); font-size:14px; margin:4px 0 16px; }
  .layer { border:1px solid var(--line); border-left:3px solid var(--lc,var(--mut)); border-radius:0 10px 10px 0;
           padding:12px 16px; margin:12px 0; background:var(--panel); }
  .layer .lbl { font:700 11px ui-monospace, monospace; letter-spacing:.04em; text-transform:uppercase;
                color:var(--lc,var(--mut)); margin-bottom:4px; }
  .layer.plain { --lc:var(--ink); }
  .layer.ml { --lc:var(--train); }
  .layer.mt { --lc:var(--held); }
  .layer p { margin:0; }
  .mono { font-family:ui-monospace, monospace; }
  .kicker { color:var(--mut); font-size:12.5px; margin-top:44px; border-top:1px solid var(--line); padding:18px 0 40px; }
  b { color:var(--ink); }
</style></head>
<body>
<div class="wrap">
  <div class="backbar">
    <a class="backbtn" href="/">&larr; Back to the demo</a>
    <span class="subtle">in depth</span>
  </div>
  <h1>What the demo is actually showing</h1>
  <p class="lede">The live page lets you <i>watch</i> a tiny AI learn. This page explains what you are seeing, three
    ways: first in everyday language, then in the words a machine-learning researcher would use, then in terms of
    the specific little machine on the demo page (its makers call it a <b>minimal transformer</b>). Read top to
    bottom, or jump to the result you clicked from.</p>
  <div class="toc">
    <a href="#setup">The machine &amp; the task</a>
    <a href="#portrait">Reading the portrait</a>
    <a href="#groks">GROKS</a>
    <a href="#nogrok">NO-GROK</a>
    <a href="#memorize">MEMORIZES</a>
    <a href="#identity">Endpoint vs. prior</a>
    <a href="#implicit-bias">Implicit bias</a>
    <a href="#why">Why it matters</a>
  </div>

  <section class="topic" id="setup">
    <h2>The machine and the task</h2>
    <div class="layer plain"><div class="lbl">In plain terms</div>
      <p>Picture the smallest possible &ldquo;student&rdquo;: a handful of numbers that turn an input into an
        answer. We give it a simple fixed rule to learn from a few worked examples, the way you'd learn a pattern
        from a short answer key. Because the student is so small, we can see <b>every</b> number inside it and
        exactly how it turns each input into an output, which is what makes this a good place to watch learning
        happen.</p></div>
    <div class="layer ml"><div class="lbl">In machine-learning terms</div>
      <p>The task is the <b>NAND</b> truth table, a two-input logic function. It's encoded as a next-token map over
        six symbols (four input rows, two output bits), and the model is a one-layer, tied-embedding transformer
        with a single small weight matrix, trained by gradient descent. NAND is used because it is
        <i>functionally complete</i>, any Boolean circuit can be built from it, so the toy is not a special case
        but a universal primitive.</p></div>
    <div class="layer mt"><div class="lbl">In terms of the minimal transformer</div>
      <p>The step is <span class="mono">next = argmax over v of  emb[v] &middot; (I + M) &middot; emb[x]</span>. Apply
        it repeatedly and each input token flows to an output token and stays there, so the learned weight
        <span class="mono">M</span> is fully described by its <b>phase portrait</b>: which inputs land on
        <span class="mono">O</span> (the bit 1) and which land on <span class="mono">Z</span> (the bit 0). A correct
        NAND weight reads <span class="mono">O&lt;-Opqr | Z&lt;-Zs</span>.</p></div>
  </section>

  <section class="topic" id="portrait">
    <h2>Reading a phase portrait</h2>
    <p class="subtle">Those little strings like <span class="mono">O&lt;-Opqr | Z&lt;-Zs</span> are a compact map
      of the whole trained model. Here is how to read them.</p>
    <div class="layer plain"><div class="lbl">In plain terms</div>
      <p>Read the arrow as <b>&ldquo;lands on.&rdquo;</b> Each group is one destination followed by everything that
        ends up there. <span class="mono">O&lt;-Opqr</span> means the inputs <span class="mono">p, q, r</span> (and
        <span class="mono">O</span> itself) all settle on the answer <b>1</b>; <span class="mono">Z&lt;-Zs</span>
        means <span class="mono">s</span> settles on <b>0</b>. A tidy two-group map like this is the machine getting
        the rule right. If instead you see something like <span class="mono">p&lt;-pq</span>, that means
        <span class="mono">p</span> became its own little dead-end that also swallowed <span class="mono">q</span>,
        a wrong pocket the model got stuck in, a tell-tale sign it memorized rather than learned the rule.</p></div>
    <div class="layer ml"><div class="lbl">In machine-learning terms</div>
      <p>The map is deterministic, so the state space partitions into <b>basins of attraction</b>, each written
        <span class="mono">attractor&lt;-members</span>: the fixed point (or cycle) it converges to, and the states
        that flow into it. The correct NAND function has exactly <b>two</b> basins, one per output bit. Any extra
        basin, or an input token that is its own fixed point, means the learned map is <i>not</i> the target
        function, it is some other function that merely happens to agree on the training rows.</p></div>
    <div class="layer mt"><div class="lbl">In terms of the minimal transformer</div>
      <p>Each token is iterated under <span class="mono">f(x) = argmax emb[v]&middot;(I+M)&middot;emb[x]</span>. The
        target reads <span class="mono">O&lt;-Opqr | Z&lt;-Zs</span>: basin of <span class="mono">O</span> is
        <span class="mono">{O,p,q,r}</span>, basin of <span class="mono">Z</span> is <span class="mono">{Z,s}</span>.
        The memorizing endpoint <span class="mono">O&lt;-O | Z&lt;-Zrs | p&lt;-pq</span> reads: <span class="mono">O
        </span> attracts only itself, <span class="mono">Z</span> attracts <span class="mono">{Z,r,s}</span>, and
        <span class="mono">p</span> is a spurious fixed point that also captured <span class="mono">q</span>. Three
        basins, wrong outputs, no NAND, exactly what memorization looks like as a picture.</p></div>
  </section>

  <section class="topic" id="groks">
    <h2>GROKS <span class="tag GROKS">delayed learning</span></h2>
    <p class="subtle">You held out one example the rest do not pin down, yet the machine eventually gets it, long
      after it aced the practice set.</p>
    <div class="layer plain"><div class="lbl">In plain terms</div>
      <p>The student nailed its practice questions almost immediately, then spent a long time looking completely
        stuck on a new question, getting it wrong over and over. Then, abruptly, it clicked and got the new one
        right. That late &ldquo;aha&rdquo; is called <b>grokking</b>. The surprise is that the stuck period was not
        wasted: underneath, the student was steadily getting more sure of the right answer, and the click is simply
        the moment that quiet confidence finally tipped past the wrong answer.</p></div>
    <div class="layer ml"><div class="lbl">In machine-learning terms</div>
      <p><b>Delayed generalization.</b> Training accuracy saturates at 100% early, but held-out (test) accuracy
        rises much later, during the continued optimization after the data is already fit. The engine of the delay
        is the <b>implicit bias</b> of gradient descent: past zero training loss it keeps growing the margin
        (equivalently, drifts toward the min-norm / max-margin solution). The held-out point isn't logically forced
        by the training set, but the max-margin solution happens to classify it correctly, so generalization
        arrives on the margin-growing timescale, not the fitting timescale.</p></div>
    <div class="layer mt"><div class="lbl">In terms of the minimal transformer</div>
      <p>Hold out row <span class="mono">s</span>. The other rows do not <i>force</i> it: more than one weight fits
        them while sending <span class="mono">s</span> to different places. Yet from a zero start, gradient descent
        converges to the single <b>simplest</b> weight, which sends <span class="mono">s&rarr;Z</span>. On the demo
        you can watch the held-out <b>margin</b> climb across the flat plateau and cross zero at the exact grok
        step, while the phase portrait undergoes a <b>bifurcation</b>: <span class="mono">s</span> starts as its own
        dead-end (<span class="mono">s&lt;-s</span>) and its basin is absorbed into <span class="mono">Z</span>
        (<span class="mono">Z&lt;-Zs</span>) the instant it groks.</p></div>
  </section>

  <section class="topic" id="nogrok">
    <h2>NO-GROK <span class="tag NOGROK">instant learning</span></h2>
    <p class="subtle">You held out an example the others already imply, so there is nothing to delay: it is right as
      soon as (or before) the practice set is fit.</p>
    <div class="layer plain"><div class="lbl">In plain terms</div>
      <p>Sometimes the hidden question is already answered by the others. If you know two sides of a pattern, the
        third is obvious, so the moment the student learns the rest, it gets the hidden one for free. No stuck
        period, no dramatic click, it just knows.</p></div>
    <div class="layer ml"><div class="lbl">In machine-learning terms</div>
      <p>The held-out point is <b>determined</b> by the training constraints (it lies in their span), so the
        generalizing solution and the fitting solution are one and the same. Generalization coincides with, or even
        precedes, convergence on the training set; there is no separate implicit-bias phase to wait through, hence
        no grok gap.</p></div>
    <div class="layer mt"><div class="lbl">In terms of the minimal transformer</div>
      <p>Hold out a redundant &ldquo;1&rdquo; row such as <span class="mono">r</span>. Rows
        <span class="mono">p</span> and <span class="mono">q</span> already force it, so the survivor set agrees on
        <span class="mono">r&rarr;O</span> from the start. The demo shows held-out accuracy reaching 100% at a step
        <i>before</i> training even finishes fitting, and the endpoint still equals the prior,
        <span class="mono">O&lt;-Opqr | Z&lt;-Zs</span>.</p></div>
  </section>

  <section class="topic" id="memorize">
    <h2>MEMORIZES <span class="tag MEMORIZES">no learning of the rule</span></h2>
    <p class="subtle">You held out so much that nothing points at the answer, so the machine just memorizes what it
      saw and never recovers the rule.</p>
    <div class="layer plain"><div class="lbl">In plain terms</div>
      <p>If you hide every example that could teach the underlying pattern, the student has nothing to reason from
        for the hidden cases. It can still ace the questions it was shown, by rote, but it never figures out the
        rule, so it stays wrong on everything new, forever. This is the ordinary failure people mean by
        &ldquo;it just memorized the answers.&rdquo;</p></div>
    <div class="layer ml"><div class="lbl">In machine-learning terms</div>
      <p><b>Underdetermination, i.e. memorization / overfitting.</b> The training set neither logically forces the
        held-out labels nor makes them the implicit-bias-preferred completion, so no amount of continued training
        pulls test accuracy up. Train loss goes to zero; test accuracy is stuck below 100%. Grokking needs a
        generalizing solution to exist and be bias-preferred; remove that and you get pure memorization.</p></div>
    <div class="layer mt"><div class="lbl">In terms of the minimal transformer</div>
      <p>Hold out all three &ldquo;1&rdquo; rows <span class="mono">{p, q, r}</span>. Only the fixed points remain,
        which force nothing about the inputs. Training fits the shown rows, but the endpoint's portrait is something
        like <span class="mono">O&lt;-O | Z&lt;-Zrs | p&lt;-pq</span>, <b>not</b> the NAND prior, and the held-out
        rows never become correct. This is the sharp control: grokking versus memorizing turns entirely on whether
        the training rows force the held-out ones.</p></div>
  </section>

  <section class="topic" id="identity">
    <h2>Endpoint vs. the prior <span class="tag ID">two roads, one answer</span></h2>
    <p class="subtle">Whenever it truly learns, the weight it trains into is identical to the &ldquo;simplest&rdquo;
      weight worked out by hand, evidence the late learning is not luck.</p>
    <div class="layer plain"><div class="lbl">In plain terms</div>
      <p>There are two ways to find the simplest rule that fits the examples: work it out with pen and paper ahead
        of time, or let the student slowly train until it settles. The striking result is that when the student
        groks, it lands on the <b>exact same</b> rule the pen-and-paper method picks. So the sudden click is not the
        machine guessing lucky, it is the machine <i>finding</i> the simplest rule consistent with what it was
        shown.</p></div>
    <div class="layer ml"><div class="lbl">In machine-learning terms</div>
      <p>This is the implicit bias made concrete. The <b>analytic prior</b> (here the minimum-complexity weight,
        min-L1) and the <b>SGD endpoint</b> coincide: the solution gradient descent reaches equals the one a
        min-norm / max-margin criterion selects in closed form. It's a rare case where the dynamical
        characterization (where training goes) and the variational one (what objective the answer minimizes) are
        shown to be the same object, not merely argued to be. When the setup memorizes instead, they diverge, the
        endpoint is not the prior, which is exactly why the identity holds <i>only</i> when it generalizes.</p></div>
    <div class="layer mt"><div class="lbl">In terms of the minimal transformer</div>
      <p>The demo enumerates the min-L1 weight and prints its phase portrait, then trains a weight by SGD and prints
        <i>its</i> portrait. In GROKS and NO-GROK they read identically (<span class="mono">O&lt;-Opqr |
        Z&lt;-Zs</span>): <span class="mono">endpoint == prior</span>. In MEMORIZES they differ:
        <span class="mono">MISMATCH</span>. Same object when it generalizes; different objects when it only
        memorizes.</p></div>
  </section>

  <section class="topic" id="implicit-bias">
    <h2>What &ldquo;implicit bias&rdquo; means</h2>
    <p class="subtle">The phrase runs through this whole story. It is the reason the machine generalizes at all,
      and the reason grokking happens on the schedule it does.</p>
    <div class="layer plain"><div class="lbl">In plain terms</div>
      <p>Often many different answers fit the practice examples equally well. Something still has to decide which
        one the machine actually ends up with, and the training process has a built-in, unspoken preference: it
        drifts toward the <b>simplest, most clear-cut</b> answer, even though nobody wrote that rule down. That
        hidden preference is the <b>implicit bias</b>. It is why, faced with an example the practice set left open,
        the machine tends to land on the sensible general rule instead of some odd answer that only fits what it
        was shown.</p></div>
    <div class="layer ml"><div class="lbl">In machine-learning terms</div>
      <p>An overparameterized model has <i>many</i> zero-loss solutions; the optimizer chooses among them. For
        gradient descent on separable (and homogeneous) problems, the chosen one is the <b>max-margin / min-norm</b>
        solution (Soudry et&nbsp;al.), reached during the margin-growing phase <i>after</i> the loss is already
        near zero. Crucially it is not a term in the loss, it is a property of the optimization trajectory, which is
        why it is called <i>implicit</i>. It is the engine of grokking: the generalizing solution is the
        bias-preferred one, and it is reached late.</p></div>
    <div class="layer mt"><div class="lbl">In terms of the minimal transformer</div>
      <p>Among all the integer weights that fit the shown rows, more than one exists, the <b>survivor set</b>. The
        implicit bias selects the minimum-complexity member (here <b>min-L1</b>), which is exactly the weight whose
        portrait is the NAND prior <span class="mono">O&lt;-Opqr | Z&lt;-Zs</span>. Gradient descent from a zero
        start reaches that <i>same</i> weight, which is why <span class="mono">endpoint == prior</span> when it
        groks. Take the forcing away (the MEMORIZES control) and there is no generalizing target for the bias to
        prefer, so it never gets there.</p></div>
  </section>

  <section class="topic" id="why">
    <h2>Why this matters</h2>
    <div class="layer plain"><div class="lbl">In plain terms</div>
      <p>The same &ldquo;sudden understanding&rdquo; shows up in the huge AI models people use every day, but those
        are far too big to see inside. This tiny version is small enough to watch completely, so you can see that
        the sudden click is not magic: it is the machine settling on the simplest rule that fits, and you can see
        exactly when and why.</p></div>
    <div class="layer ml"><div class="lbl">In machine-learning terms</div>
      <p>It is a fully inspectable model of grokking, implicit bias, and the memorization / generalization boundary,
        where the analytic and dynamical accounts can be checked against each other exactly, deterministically, and
        end to end. A controlled microscope for phenomena usually studied only at scales where they can't be
        resolved.</p></div>
    <div class="layer mt"><div class="lbl">In terms of the minimal transformer</div>
      <p>Same architecture as a frontier transformer (embed, an update, an argmax readout), shrunk until every
        weight and every intermediate is legible, with no hidden layers. That legibility is the whole point: it
        turns &ldquo;it generalized&rdquo; into a picture you can point at, the implicit bias selecting which
        attractor an undetermined input falls into.</p></div>
  </section>

  <div class="kicker">Deterministic and torch-free. The numbers behind every claim here are computed live on the
    demo page and asserted by the project's test suite. <a href="/">&larr; back to the demo</a></div>
</div>
</body></html>
"""


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    # Prove the zero-init default is the same deterministic machine run.sh asserts -- BEFORE serving,
    # and without needing run.sh. Fail loudly if it ever regresses.
    hc = selfcheck()
    tag = "OK" if hc["matches_runsh_no4"] else "MISMATCH"
    print("self-check [%s]: zero-init default -> %s fit@step%s grok@step%s, endpoint==prior:%s "
          "(identical to run.sh NO-4a; deterministic, no run.sh required)"
          % (tag, hc["verdict"], hc["fit_at"], hc["grok_at"],
             "yes" if hc["endpoint_is_prior"] else "no"), flush=True)
    if not hc["matches_runsh_no4"]:
        print("REFUSING TO SERVE: the deterministic default no longer matches the asserted CLI verdict.",
              file=sys.stderr)
        return 1
    try:
        srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    except OSError as exc:
        print("could not bind port %d: %s\n(is a server already running? try: python3 server/app.py <other-port>)"
              % (port, exc), file=sys.stderr)
        return 1
    print("minimal-transformer viz on http://localhost:%d  (Ctrl-C to stop)" % port, flush=True)
    print("  default page load is zero-init and deterministic; the browser step numbers are the EXACT")
    print("  first-crossing steps (per-step grid), while run.sh reports the coarse-grid crossings -- same run.",
          flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
