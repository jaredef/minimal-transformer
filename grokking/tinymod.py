"""The tiny grokking model: ~539 parameters, smaller than any published trained-and-grokked model.

The mod-addition model in modadd.py (~36k params) is comfortably in the known small-model grokking
regime. This strips it to the minimum that still groks: a shared additive token embedding, a quadratic
activation, and a linear readout -- no concatenated embeddings, no separate first layer:

    pre    = E[a] + E[b]          # E: (p, H), shared and additive (translation-equivariant)
    h      = pre * pre            # quadratic activation (gives the cosine angle-addition for mod-add)
    logits = h @ Wout + bout      # Wout: (H, p)
    params = 2*p*H + p

At p=11, H=24 this is 539 parameters, and it groks (a+b) mod 11 with a real held-out plateau: train fits
in a few hundred steps, held-out stays at chance for thousands of steps, then jumps. For reference, the
smallest trained-and-grokked model in the published literature we could find is Google PAIR's 3,216-param
(24-neuron, mod-67) MLP; this is ~6x smaller. Requires numpy. Deterministic given the seed.

The memorizing control swaps the quadratic activation for ReLU, which fits just as fast and never groks.
"""
import numpy as np

np = np  # re-exported so the server can reach numpy via tinymod.np


def train(act="quad", p=11, H=24, frac=0.8, lr=1e-3, wd=1.0, steps=20000, seed=0, ckpt=200,
          on_ckpt=None, anneal=False):
    """AdamW training of the minimal additive model. Returns (history, params); params has keys
    E, Wout, bout. history rows are (step, train_acc, val_acc, weight_norm). Calls
    on_ckpt(step, params, train_acc, val_acc, weight_norm) at each checkpoint if given.
    anneal=True cosine-decays the learning rate to a small floor over training, so the solution settles
    instead of oscillating -- held-out accuracy then converges to and holds 100% rather than wobbling."""
    rng = np.random.default_rng(seed)
    A, B = np.meshgrid(np.arange(p), np.arange(p), indexing="ij")
    a, b = A.ravel(), B.ravel()
    y = (a + b) % p
    idx = rng.permutation(p * p)
    ntr = int(frac * p * p)
    tr, va = idx[:ntr], idx[ntr:]
    P = {"E": rng.normal(0, 0.3, (p, H)),
         "Wout": rng.normal(0, 1.0 / np.sqrt(H), (H, p)),
         "bout": np.zeros(p)}
    decay = {"E": True, "Wout": True, "bout": False}
    m = {k: np.zeros_like(v) for k, v in P.items()}
    v = {k: np.zeros_like(v) for k, v in P.items()}
    beta1, beta2, eps, t = 0.9, 0.999, 1e-8, 0
    phi = (lambda z: z * z) if act == "quad" else (lambda z: np.maximum(z, 0.0))
    dphi = (lambda z: 2.0 * z) if act == "quad" else (lambda z: (z > 0).astype(np.float64))

    def forward(ix):
        pre = P["E"][a[ix]] + P["E"][b[ix]]
        h = phi(pre)
        return pre, h, h @ P["Wout"] + P["bout"]

    def accuracy(ix):
        return float((forward(ix)[2].argmax(1) == y[ix]).mean())

    def weight_norm():
        return float(np.sqrt((P["E"] ** 2).sum() + (P["Wout"] ** 2).sum()))

    history = []
    for step in range(steps + 1):
        pre, h, logits = forward(tr)
        logits = logits - logits.max(1, keepdims=True)
        ex = np.exp(logits)
        probs = ex / ex.sum(1, keepdims=True)
        n = len(tr)
        dl = probs.copy()
        dl[np.arange(n), y[tr]] -= 1.0
        dl /= n
        gWout = h.T @ dl
        gbout = dl.sum(0)
        dpre = (dl @ P["Wout"].T) * dphi(pre)
        gE = np.zeros_like(P["E"])
        np.add.at(gE, a[tr], dpre)
        np.add.at(gE, b[tr], dpre)
        grads = {"E": gE, "Wout": gWout, "bout": gbout}
        t += 1
        lr_t = lr * (0.02 + 0.98 * 0.5 * (1 + np.cos(np.pi * step / steps))) if anneal else lr
        for k in P:
            m[k] = beta1 * m[k] + (1 - beta1) * grads[k]
            v[k] = beta2 * v[k] + (1 - beta2) * grads[k] ** 2
            P[k] -= lr_t * ((m[k] / (1 - beta1 ** t)) / (np.sqrt(v[k] / (1 - beta2 ** t)) + eps))
            if decay[k]:
                P[k] -= lr_t * wd * P[k]
        if step % ckpt == 0:
            ta, vaa, wn = accuracy(tr), accuracy(va), weight_norm()
            history.append((step, ta, vaa, wn))
            if on_ckpt is not None:
                on_ckpt(step, P, ta, vaa, wn)
    return history, P


def summarize(history):
    fit = next((s for s, ta, va, nm in history if ta >= 1.0), None)
    grok = next((s for s, ta, va, nm in history if va >= 0.95), None)
    return {"fit_at": fit, "grok_at": grok, "final_val": history[-1][2]}


if __name__ == "__main__":
    hist, _ = train()
    for step, ta, va, nm in hist:
        print("%6d  train %.2f  val %.2f  norm %.1f  %s" % (step, ta, va, nm, "#" * int(va * 30)))
    print(summarize(hist), "params =", 2 * 11 * 24 + 11)
