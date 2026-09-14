# minimal-transformer

[![checks](https://github.com/jaredef/minimal-transformer/actions/workflows/ci.yml/badge.svg)](https://github.com/jaredef/minimal-transformer/actions/workflows/ci.yml)

**A ~9-parameter, one-layer transformer that *groks* the NAND gate — where the solution gradient
descent lands on is found independently by exhaustive search, so every claim can be checked exactly.**

Grokking (delayed generalization) and the implicit bias of gradient descent are usually studied in models
far too large to inspect. This repository shrinks the same phenomena to a model with a single 3×3 weight
matrix and a handful of training examples: small enough to watch every weight, and small enough that the
"simplest solution" the implicit bias is supposed to prefer can be computed by brute force and compared
against what training actually reaches.

- **Zero dependencies.** Pure Python standard library (3.8+). No PyTorch, NumPy, or anything else.
- **Deterministic and self-checking.** Each probe prints one exact verdict string; `run.sh` asserts all of
  them. An interactive visualization (`server/app.py`) reproduces the same numbers live.

> **Scope.** This is a *pedagogical microscope*, not a new result. The phenomena (grokking, implicit bias,
> the memorization/generalization boundary) are established; see **Prior work** below for what is and is not
> new here, and **Scope and caveats** for the honest limits.

## The task and the model

The task is the **NAND** truth table (a two-input logic gate that outputs 0 only when both inputs are 1).
NAND is used because it is functionally complete — any Boolean circuit can be built from it — so the toy is
a universal primitive rather than a special case.

It is encoded as a next-token map over six tokens: the four input rows `p,q,r,s = (00,01,10,11)` and the two
output bits `O = 1`, `Z = 0`, with the outputs as fixed points. A token `x` has a fixed integer embedding
`emb[x] ∈ {-1,0,1}^3`; the (tied-embedding, one-layer) transformer step is

```
f(x) = argmax_v  emb[v] · (I + M) · emb[x]        (tied embeddings: unembed == embed)
```

Only the 3×3 weight `M` is learned; the embeddings are fixed (see caveats). Iterating `f` gives each input's
one-step answer, so a trained `M` is fully described by its **phase portrait**: which inputs land on `O` and
which land on `Z`. The correct NAND weight reads `O<-Opqr | Z<-Zs`.

## What it shows

### The analytic reference: the minimum-L1-norm weight (`probes/nand_orbit_probe.py`, NO-1..3)

With no training at all: enumerate all `3^9 = 19,683` integer weight matrices, keep those whose map fits the
rows, and take the one with the smallest L1 norm `Σ|M_ij|`. L1 is a *computable simplicity measure* (not the
L2/max-margin bias itself — see caveats), used because the small integer weight space can be searched
exhaustively.

| rung | fact |
|------|------|
| **NO-1** `THE-NAND-ORBIT` | the fitted weight's phase portrait is two fixed-point attractors (the output bits); every input row maps in one step to its correct output and stays there. |
| **NO-2** `THE-BASINS-ARE-THE-TRUTH-TABLE` | the basin of `O` is `{p,q,r}` (NAND=1), the basin of `Z` is `{s}` (NAND=0). The 3:1 basin split **is** the gate. |
| **NO-3** `GROK-IS-PRIOR-CONDITIONAL` | hold out row `s`; the other rows do **not** force it (weights consistent with them send it to more than one place), yet the minimum-L1-norm weight still selects the generalizing map `s→Z`. Generalization here comes from the simplicity preference, not from the data forcing the answer. |

### The training dynamic: grokking under gradient descent (`probes/nand_grok_sgd_probe.py`, NO-4)

Same encoding, same embeddings, but now `M` is **learned by (full-batch) gradient descent** from a zero
start — the softmax cross-entropy gradient `(p_u − onehot)·emb[u]·emb[s]`, summed over the training rows each
step (the family is loosely called "SGD," but there is no minibatch sampling here). We split the rows into a
training subset and a held-out remainder and record train vs held-out accuracy over training. The control is
**whether the training subset forces the held-out row.**

| rung | holdout | verdict |
|------|---------|---------|
| **NO-4a** `GROKS` | `s` (the minority NAND=0 row) | train accuracy hits 100% at **step 50**, held-out `s→Z` only at **step 100** — a real gap. `s` is not forced, yet the implicit bias reaches `s→Z` anyway, late. |
| **NO-4b** `NO-GROK` | `q` (a genuinely forced row) | *every* weight consistent with the other rows sends `q→O`, so held-out is right at step 25 — before the fit. No plateau, no gap. |
| **NO-4c** `MEMORIZES` | `p,q,r` (all NAND=1 rows) | the remainder (only the fixed points) forces nothing, so nothing points the held-out rows at the right output. Train fits; held-out **never** reaches 100%. |

The three regimes line up with forcing: **forced → immediate (no grok); not forced but reachable by the
implicit bias → delayed (grok); under-determined → never (memorize).**

### The evidence that it's grokking, not luck (`probes/nand_grok_evidence_probe.py`, NO-5..6)

| rung | fact |
|------|------|
| **NO-5** `MARGIN-CROSSES-AT-GROK` | on the plateau where train accuracy is already 100%, the held-out margin (`logit(correct) − max other logit`) is still negative at the fit (−0.91), increases monotonically, and crosses zero **exactly** at the grok step, while `‖M‖` keeps rising. The flip is a smooth threshold crossing driven by continued optimization: the plateau is not a stationary point. |
| **NO-6** `SGD-ENDPOINT-IS-THE-PRIOR` | trained holding out `s`, the trained weight and the minimum-L1-norm weight **induce the same function** — identical phase portraits `O<-Opqr\|Z<-Zs` (a match of behavior, not of the raw parameters). The solution training reaches equals the one the simplicity search picks. |

## Visualize (localhost, no dependencies)

```sh
python3 server/app.py            # then open http://localhost:8000  (optional: app.py <port>)
```

A standard-library `http.server` computes the trajectories with `probes/nand_core.py` — the same numbers
`run.sh` asserts — and serves one page with: the accuracy gap (plateau shaded), the held-out margin crossing
zero at the grok step, a live 3×3 weight heatmap, a per-step scrubber (watch grokking as a phase-portrait
bifurcation), the three controls side by side, a "quiz the machine" game, a scroll-through tutorial, an
in-depth `/explain` page, and a "Show me the code" section pulled from the live source.

On startup the server runs a self-check that reproduces the exact `run.sh` NO-4a verdict on the zero-init
default and refuses to serve if it ever fails to match (`GET /api/health` returns it as JSON). The default
page load is deterministic; only **↻ train again** introduces a random start (which still groks every time,
showing the effect is not an artifact of the zero initialization).

## Run the checks

```sh
sh run.sh
```

Or a single probe (the training probe also prints an ASCII accuracy curve to stderr; stdout stays the single
verdict line that `run.sh` asserts):

```sh
python3 probes/nand_orbit_probe.py    fixtures/grok-is-prior-conditional.json
python3 probes/nand_grok_sgd_probe.py fixtures/nand-groks-in-time.json
```

## Prior work

None of the phenomena here are new; the contribution, if any, is the scale and the exactness of the exhibit.

- **Grokking / delayed generalization** — Power et al., *Grokking: Generalization Beyond Overfitting* (2022).
- **Implicit bias of gradient descent** toward the max-margin / minimum-norm solution — Soudry et al., *The
  Implicit Bias of Gradient Descent on Separable Data* (2018).
- **Grokking as a phase transition / competition of solutions** — Nanda et al. (progress measures), Varma et
  al. (memorization vs generalization circuits), Liu et al. *Omnigrok* (grokking controlled by weight norm).

**What this adds:** a fully inspectable (~9-parameter), deterministic, dependency-free exhibit where the
simplicity-preferred solution is obtained by *exhaustive enumeration* rather than argued asymptotically, and
shown to induce the **same function** the trained network reaches; and where the generalize / no-grok /
memorize outcomes reduce to an exactly **decidable** condition (whether the training rows force the held-out
one). It is a teaching and sanity-checking microscope for these effects, not a claim of a new phenomenon or a
mechanistic identity with grokking in large models.

## Scope and caveats

- **Optimizer.** This is **full-batch** gradient descent (all training rows every step); there is no
  stochastic minibatching. "SGD" is used loosely for the family.
- **Simplicity measure.** The analytic reference minimizes the **L1** norm over an integer lattice, chosen
  because it is exactly enumerable. The implicit bias of GD is generally characterized as **L2 / max-margin**;
  here the two happen to select the same *function*. Do not read min-L1 as literally the implicit bias.
- **The match is behavioral.** NO-6 compares **phase portraits** (input→output maps), not raw weight matrices.
- **Fixed embeddings.** The token embeddings are fixed integers chosen for this demo; only `M` is learned. The
  qualitative story is robust to the initialization of `M` (the "train again" button), but grokking here does
  depend on the encoding, and no claim is made about robustness across all embeddings.
- **Small-scale analog.** This illustrates the same *kind* of effect studied in large models; it is not
  evidence about the mechanism of grokking at scale.

## Provenance

Extracted and extended from the GCCS research corpus (`machines/transformer-nand-orbit`, NO-1..3, and after
`machines/transformer-grokking-dynamics`). NO-4..6 are added here: the gradient-descent-in-time grok on the
NAND encoding, and the margin-crossing and enumerate-and-match evidence.

## License

Jared Foy © 2026. Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — share and adapt,
including commercially, with attribution. See [`LICENSE`](LICENSE).
