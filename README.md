# minimal-transformer

**The NAND gate, read as the orbit of a lowered transformer — and grokking made visible, first as a
prior-conditional static and then as a training dynamic under real SGD.**

A lowered transformer is a deterministic dynamical system: the compiled weight is the dynamics, the seed
is the initial condition, generation is the *orbit*, and the reachable behaviours are the *attractors*.
This repository takes the smallest interesting gate — NAND, the lone universal boolean connective — and
shows it is not a circuit but a **phase portrait**, then shows that **grokking** (delayed generalization)
is exactly the implicit-bias prior choosing which attractor an underdetermined seed falls into.

Everything is deterministic, torch-free (pure Python stdlib), and fail-closed: each probe prints one exact
verdict string, and `run.sh` asserts it.

## The encoding

NAND is a next-token map over six tokens: the four input rows `p,q,r,s = (00,01,10,11)` and the two output
bits `O = 1`, `Z = 0`, the outputs as fixed points. A token `x` has embedding `emb[x] ∈ Z^3`; the
transformer step is

```
f(x) = argmax_v  emb[v] · (I + M) · emb[x]        (tied embeddings: unemb == emb)
```

## The two halves

### NO-1..3 — the static orbit (`probes/nand_orbit_probe.py`)

The weight `M` is the **min-L1 integer weight** compiled from the enumerable bench — no gradient descent,
just the implicit-bias prior.

| rung | fact |
|------|------|
| **NO-1** `THE-NAND-ORBIT` | compiled to a weight, NAND's phase portrait is two fixed-point attractors (the output bits). Every input row seeded flows in one step to its output and stays. Generation is the orbit; the reachable behaviours are exactly the attractors. |
| **NO-2** `THE-BASINS-ARE-THE-TRUTH-TABLE` | the basin of `O` is `{p,q,r}` (the rows with NAND=1), the basin of `Z` is `{s}` (the one row with NAND=0). The 3:1 basin asymmetry **is** the gate. |
| **NO-3** `GROK-IS-PRIOR-CONDITIONAL` | hold out row `s`; the other three do **not** force it (the survivor set sends it to more than one place), yet the min-L1 weight — the prior — selects the generalising dynamics `s→Z`. Selection is the prior: grok via implicit bias, as a static property of the portrait. |

### NO-4 — the training dynamic (`probes/nand_grok_sgd_probe.py`)

The same 6-token NAND encoding and the same embeddings, but now `M` is **learned by SGD** (zero init, the
exact enumerable-transformer DYN-1 gradient `(p_u − onehot)·emb[u]·emb[s]`, tied embeddings). We split the
form into a TRAIN subset and a HELD-OUT remainder and record train vs held-out accuracy at fixed
checkpoints. The control is the same as NO-3's: **whether the train subset forces the held-out.**

| rung | holdout | verdict |
|------|---------|---------|
| **NO-4a** `GROKS` | `s` (the minority NAND=0 row) | train fits at **step 50**, held-out `s→Z` reaches 100% only at **step 100** — a genuine gap. `s` is not box-forced, yet SGD's implicit max-margin bias selects `s→Z` anyway. **The dynamic image of NO-3's min-L1 prior.** |
| **NO-4b** `NO-GROK` | `r` (a redundant majority row) | `r` is forced by `p,q`, so it generalizes at step 25 — before the fit. No implicit-bias phase, no grok gap. |
| **NO-4c** `MEMORIZES` | `p,q,r` (all majority rows) | the remainder (only the fixed points) does not force them — underdetermined. Train fits; held-out **never** reaches 100%. |

The three regimes line up exactly with forcing: **forced → immediate (no grok); not forced but reachable by
the implicit bias → delayed (grok); underdetermined → never (memorize).** That is the whole grokking
phenomenon, on the smallest gate, visible in time.

### NO-5..6 — the evidence (`probes/nand_grok_evidence_probe.py`)

An accuracy gap is only the *symptom*. These two rungs prove grokking is the *cause* — that learning kept
happening during the apparent plateau and produced the late jump.

| rung | fact |
|------|------|
| **NO-5** `MARGIN-CROSSES-AT-GROK` | on the plateau where train accuracy is already 100% and nothing seems to be happening, the held-out **margin** (`logit(correct) − max other`) is still negative at the fit (−0.91), climbs **monotonically**, and crosses zero **exactly** at the grok step — while `‖M‖` keeps rising. The flip is a smooth threshold crossing driven by continued descent, not luck. The plateau is not a stationary point. |
| **NO-6** `SGD-ENDPOINT-IS-THE-PRIOR` | trained holding out `s`, the SGD endpoint's phase portrait `O<-Opqr\|Z<-Zs` equals the min-L1 **prior** portrait NO-3 selects analytically. The dynamic grok and the static implicit-bias prior are literally the same object — an independent oracle the SGD run lands on. |

## Visualize (localhost, stdlib only)

```sh
python3 server/app.py            # then open http://localhost:8000  (optional: app.py <port>)
```

You do **not** need to run `run.sh` first. On startup the server runs a self-check that reproduces the exact
`run.sh` NO-4a verdict on the zero-init default and prints it (`self-check [OK]: zero-init default -> GROKS
fit@step50 grok@step100, endpoint==prior:yes`); if that ever fails to match, it refuses to serve. The
default page load is **zero-init and deterministic** — identical bytes every time, the same machine the CLI
asserts. (`GET /api/health` returns the self-check as JSON.) Only **↻ train again** introduces a random
start. Note the browser reports the *exact* first-crossing steps (it records every step), while `run.sh`
reports the coarse-grid crossings — the same run, sampled differently.

A dependency-free Python server (`http.server`) computes the trajectories with `probes/nand_core.py` — the
same numbers `run.sh` asserts — and serves one page that shows: the accuracy gap with the plateau shaded,
the **held-out margin crossing zero exactly at the grok step**, the endpoint-equals-prior identity, and the
three controls (GROKS / NO-GROK / MEMORIZES) side by side.

- **Three-control explorer** — a `GROKS / NO-GROK / MEMORIZES` selector (or click a control card below) drives
  the whole panel from the already-computed regimes: hold out `s` (not forced) → grokks late; hold out a
  forced row → generalizes immediately; hold out all three "1" rows → memorizes, never generalizes. The
  charts, scrubber, live state, and the endpoint-vs-prior card all update to the chosen control.
- **Live weight heatmap** — a 3×3 grid of the model's weight matrix `M`, colored by value (blue negative,
  pink positive, brighter = larger), updating as you scrub. Watch all nine numbers grow from zero as it learns
  and freeze once the phase portrait locks in — the whole model, visible.
- **Step-scrubber** — drag the slider to move **one training step at a time** (every step 0..400). A live
  panel shows train/held accuracy, the held-out margin, `‖M‖`, the held-row map (`s → ?`) and the full
  phase portrait at that step. Watch the grok as a *bifurcation*: `s` starts as its own spurious fixed
  point (`s<-s`), and at the grok step its basin is absorbed into `Z` (`Z<-Zs`) exactly as the margin
  crosses zero.
- **↻ train again** — reinitializes from a fresh **random** start (`/api/data?seed=N`) and retrains, then
  sweeps the trajectory so you watch it grok again. It groks every time and always lands on the same prior
  portrait — the grok is the implicit bias, not an artifact of the zero init. (The CLI probes stay
  zero-init and deterministic; the random restart is a server-only demo.)
- **Quiz the machine** — a game below the three controls: pick which trained brain to test (the grokked, the
  no-grok, or the memorized one) and feed it each of the four NAND inputs. The grokked and no-grok machines
  answer 4/4 (including the input they never practiced); the memorized machine fails on exactly the inputs it
  never saw. Uses the trained endpoint weight's one-step prediction per input.
- **Learn more** (top-right button) — a progressive, layman-friendly tutorial that reveals on scroll: what a
  transformer is, how each part maps onto this lowered transformer, orbits and attractors, the NAND task,
  training, grokking, why the plateau isn't idle, the implicit-bias prior, the three controls, and what the
  minimal transformer is good for. Every step points back to the live demo above.

## Run

```sh
sh run.sh
```

Or a single probe:

```sh
python3 probes/nand_orbit_probe.py   fixtures/grok-is-prior-conditional.json
python3 probes/nand_grok_sgd_probe.py fixtures/nand-groks-in-time.json
```

The SGD probe also prints an **ASCII accuracy curve** (train vs held-out, over checkpoints) to *stderr* —
stdout stays the single machine-readable verdict line, so `run.sh` still asserts it exactly. The `<-fit`
and `<-grok` markers bracket the gap:

```
  step |  train acc            | held-out acc
  -----+-----------------------+-----------------------
     50| #################### 100%| ....................   0% <-fit
    100| #################### 100%| #################### 100% <-grok
  gap: train hits 100% at step50, held-out only at step100 -- delayed generalization (grok)
```

To watch the grok happen, the probe reports the first checkpoint each accuracy crosses 100%; edit a
fixture's `lr` / `steps` / `checkpoints` to widen or narrow the gap. `lr` down widens it (train and grok
both delayed, grok more), `lr` up narrows it.

## Provenance

Extracted and extended from the GCCS corpus (`machines/transformer-nand-orbit`, NO-1..3; grokking-as-a-
dynamic after `machines/transformer-grokking-dynamics`, GRK-1). NO-4 is new here: the SGD-in-time grok on
the NAND encoding itself, so the NAND grok is witnessed as both a prior-conditional static and a training
dynamic controlled the same way.

## License

© Jared Foy 2026. Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — you may share
and adapt this work, including commercially, as long as you give appropriate credit. See [`LICENSE`](LICENSE).
