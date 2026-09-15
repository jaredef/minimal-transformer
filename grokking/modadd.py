"""Modular addition that genuinely groks -- the scaled companion to the zero-dependency
NAND core, grown by constraint accumulation.

Requires numpy (this subdirectory is the ONLY part of the repo that is not pure standard
library; the core NAND probes and server stay dependency-free).

The NAND demo in the repo root shows the implicit max-margin bias of gradient descent: a
smooth threshold crossing, norm growing, one held-out bit. That is not grokking in the
strong sense. This module grows that base into genuine grokking by accumulating the
constraints that induce it, and `ladder.py` shows each one is load-bearing by ablation:

  base   implicit max-margin bias (the NAND core; no weight decay)
  + C_a  weight decay            -- a norm budget, so the weight norm turns over after the fit
  + C_b  capacity + enough data  -- a real memorize-vs-generalize competition (data >= threshold)
  + C_c  structured task         -- modular addition has a low-norm Fourier solution
  + C_d  quadratic activation    -- makes that Fourier solution reachable by the optimizer

Grokking is then a threshold-induced property: below the data threshold the model memorizes;
above it, training fits early and generalization arrives thousands of steps later as the norm
falls. The generalizing solution is Gromov's (2023) analytic Fourier solution; `fourier.py`
checks the trained embeddings converge onto it.

Task: predict (a + b) mod p. Model: learned token embeddings for a and b, concatenated, one
hidden layer with the chosen activation, softmax readout over the p classes. Optimizer: Adam
with decoupled weight decay (AdamW). Deterministic given the seed.
"""
import numpy as np


def _activation(act):
    if act == "quad":
        return (lambda z: z * z), (lambda z: 2.0 * z)
    return (lambda z: np.maximum(z, 0.0)), (lambda z: (z > 0).astype(np.float64))


def train(act="quad", p=11, d=64, H=256, frac=0.7, lr=1e-3, wd=1.0, steps=20000,
          seed=0, ckpt=1000, on_ckpt=None):
    """Train by AdamW and return (history, params). history is a list of
    (step, train_acc, val_acc, weight_norm) at every `ckpt` steps; params is the final dict.
    Data split and weight init are drawn from a single RNG stream for reproducibility.
    If `on_ckpt` is given, it is called as on_ckpt(step, params, train_acc, val_acc, weight_norm)
    at every checkpoint (used to record extra diagnostics like the embedding spectrum)."""
    rng = np.random.default_rng(seed)
    A, B = np.meshgrid(np.arange(p), np.arange(p), indexing="ij")
    a, b = A.ravel(), B.ravel()
    y = (a + b) % p
    idx = rng.permutation(p * p)
    ntr = int(frac * p * p)
    tr, va = idx[:ntr], idx[ntr:]
    P = {
        "E": rng.normal(0, 0.3, (p, d)),
        "W1": rng.normal(0, 1.0 / np.sqrt(2 * d), (2 * d, H)),
        "b1": np.zeros(H),
        "W2": rng.normal(0, 1.0 / np.sqrt(H), (H, p)),
        "b2": np.zeros(p),
    }
    decay = {"E": True, "W1": True, "b1": False, "W2": True, "b2": False}
    m = {k: np.zeros_like(v) for k, v in P.items()}
    v = {k: np.zeros_like(v) for k, v in P.items()}
    beta1, beta2, eps, t = 0.9, 0.999, 1e-8, 0
    phi, dphi = _activation(act)

    def forward(ix):
        x = np.concatenate([P["E"][a[ix]], P["E"][b[ix]]], axis=1)
        z = x @ P["W1"] + P["b1"]
        h = phi(z)
        return x, z, h, h @ P["W2"] + P["b2"]

    def accuracy(ix):
        return float((forward(ix)[3].argmax(1) == y[ix]).mean())

    def weight_norm():
        return float(np.sqrt(sum((P[k] ** 2).sum() for k in ("E", "W1", "W2"))))

    history = []
    for step in range(steps + 1):
        x, z, h, logits = forward(tr)
        logits = logits - logits.max(1, keepdims=True)
        ex = np.exp(logits)
        probs = ex / ex.sum(1, keepdims=True)
        mt = len(tr)
        dlog = probs.copy()
        dlog[np.arange(mt), y[tr]] -= 1.0
        dlog /= mt
        gW2 = h.T @ dlog
        gb2 = dlog.sum(0)
        dz = (dlog @ P["W2"].T) * dphi(z)
        gW1 = x.T @ dz
        gb1 = dz.sum(0)
        dx = dz @ P["W1"].T
        gE = np.zeros_like(P["E"])
        np.add.at(gE, a[tr], dx[:, :d])
        np.add.at(gE, b[tr], dx[:, d:])
        grads = {"E": gE, "W1": gW1, "b1": gb1, "W2": gW2, "b2": gb2}
        t += 1
        for k in P:
            m[k] = beta1 * m[k] + (1 - beta1) * grads[k]
            v[k] = beta2 * v[k] + (1 - beta2) * grads[k] ** 2
            P[k] -= lr * ((m[k] / (1 - beta1 ** t)) / (np.sqrt(v[k] / (1 - beta2 ** t)) + eps))
            if decay[k]:
                P[k] -= lr * wd * P[k]  # decoupled weight decay (AdamW)
        if step % ckpt == 0:
            ta, vaa, wn = accuracy(tr), accuracy(va), weight_norm()
            history.append((step, ta, vaa, wn))
            if on_ckpt is not None:
                on_ckpt(step, P, ta, vaa, wn)
    return history, P


def first_reaching(history, index, thresh=1.0):
    """First checkpoint step at which history[.][index] >= thresh, or None."""
    for row in history:
        if row[index] >= thresh:
            return row[0]
    return None


def summarize(history):
    fit = first_reaching(history, 1, 1.0)
    grok = first_reaching(history, 2, 0.9)
    vals = [r[2] for r in history]
    norms = [r[3] for r in history]
    return {
        "fit_at": fit, "grok_at": grok,
        "final_val": vals[-1], "max_val": max(vals),
        "norm_start": norms[0], "norm_end": norms[-1],
        "norm_turned_over": norms[-1] < 0.9 * max(norms),
    }


if __name__ == "__main__":
    history, _ = train()
    print("step   train   val    norm")
    for step, tr, va, nm in history:
        bar = "#" * int(va * 30)
        print("%5d  %.2f   %.2f  %6.1f  %s" % (step, tr, va, nm, bar))
    s = summarize(history)
    print("\nfit@%s  grok@%s  final_val=%.2f  norm %.0f->%.0f  turned_over=%s"
          % (s["fit_at"], s["grok_at"], s["final_val"], s["norm_start"], s["norm_end"],
             s["norm_turned_over"]))
