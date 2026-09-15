# grokking — the scaled companion

**Genuine grokking (delayed generalization) on modular addition, grown from the repo's
max-margin core by constraint accumulation, with each constraint shown load-bearing by ablation
and the generalizing solution checked against Gromov's analytic Fourier construction.**

> Requires **numpy** (`pip install -r requirements.txt`). This subdirectory is the only part of
> the repo that is not pure standard library; the NAND core and server stay dependency-free.

## Why this exists

The NAND demo in the repo root is honest but *not grokking in the strong sense* — it is the
implicit **max-margin bias** of gradient descent (Soudry et al.): a smooth threshold crossing, the
norm only grows, and "generalization" is a single held-out bit. Genuine grokking (Power et al.;
Liu et al. *Omnigrok*) is different: train fits early, a long plateau, then a **sudden** late jump
to full generalization, **driven by the weight norm falling** under weight decay.

This companion grows the max-margin base into genuine grokking by **accumulating the constraints
that induce it**, and demonstrates each is necessary by ablation. Grokking is thereby exhibited as
a *threshold-induced property*, not a single lucky setting.

## The constraint ladder

| | constraint | induced property | witnessed by |
|---|---|---|---|
| base | implicit max-margin bias (NAND core) | direction converges, norm grows | repo root |
| C_a | **weight decay** | a norm budget: the norm *turns over* after the fit | `ladder.py` (ablate → vanishes) |
| C_b | **capacity + data ≥ threshold** | a real memorize-vs-generalize competition | `threshold.py` |
| C_c | **structured task** (modular addition) | the generalizing solution has low norm (Fourier) | the task |
| C_d | **quadratic activation** | that solution is reachable by the optimizer | `ladder.py` (ablate → vanishes) |

## What the scripts show (p=11, d=64, H=256, AdamW, seed 0)

- **`modadd.py`** — one grokking run. Train hits 100% at **step ~1000**; held-out stays near zero
  through a plateau, then rises to **100% around step ~13000**; the weight norm **peaks (~20) then
  falls to ~13**. That norm-turnover-then-generalize is the Omnigrok signature, the opposite of the
  NAND core's monotone margin climb.
- **`threshold.py`** — grokking as a threshold. At data fraction **0.5 it memorizes** (held-out
  stuck at chance forever); at **0.7 and above it groks**. Absent, then present: a threshold.
- **`ladder.py`** — the falsifier. At the grokking operating point, **remove weight decay** or
  **swap quadratic → ReLU** and grokking **vanishes** (held-out stays ~0, norm grows instead of
  turning over). Each constraint is load-bearing.
- **`fourier.py`** — the constructed prior. Gromov (2023) derives that the grokking solution is
  periodic; its signature is a **sparse embedding spectrum**. The grokked model puts **~0.94 of its
  embedding energy in two Fourier frequencies**; the memorized models sit at the diffuse baseline
  (~0.43). So the trained network converges onto the *authored* analytic solution — the scaled
  analogue of the NAND core's "trained weight matches a computed reference," with the reference
  constructed (Gromov) rather than enumerated.

## Run

```sh
pip install -r requirements.txt
sh run.sh                 # ~1-2 minutes (trains several small models by AdamW)
# or individually:
python3 modadd.py         # one grokking run + curve
python3 threshold.py      # the data-fraction threshold
python3 ladder.py         # the ablation falsifier
python3 fourier.py        # convergence to Gromov's Fourier solution
```

## Honest scope

Modular-addition grokking is a known result; the reproduction itself is not novel. What this adds
is the **method**: growing it from the max-margin base by explicit constraint accumulation, proving
the minimal load-bearing set by ablation, and recovering an exact-style *check* at scale by
comparing against Gromov's constructed solution. The disciplines are standard ML (weight decay,
quadratic activation, Fourier analysis); the framing organizes and falsifies them. It is a
scaled teaching-and-falsification artifact, not a claim about the mechanism of grokking in large
models.
