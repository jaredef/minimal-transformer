"""The constructed prior: does the grokked network converge to Gromov's analytic solution?

For modular addition with a quadratic activation, Gromov (2023) derives that the generalizing
solution is periodic: the token embeddings become sums of a few sinusoids (Fourier modes) in
the token index, so multiplying quadratic features reproduces (a + b) mod p via the cosine
angle-addition identity. Its defining, checkable signature is a *sparse embedding spectrum*:
almost all of the embedding energy lives in a handful of frequencies.

So we do not need to enumerate a weight space (impossible at this scale). We compare the trained
embedding against the *authored* structure of the analytic solution: take the discrete Fourier
transform of the embedding table across the token axis and measure how concentrated the energy is.
A grokked model concentrates on a few frequencies; a memorized model does not. This is the
scaled analogue of the NAND core's "the trained weight matches the computed reference": here the
reference is Gromov's constructed Fourier solution, and the match is spectral concentration.
"""
import numpy as np

import modadd


def embedding_spectrum(E, p):
    """Power per Fourier frequency of the embedding table across the token axis, DC removed and
    conjugate pairs folded, normalized to sum to 1. Returns an array over k = 1..p//2."""
    power = (np.abs(np.fft.fft(E, axis=0)) ** 2).sum(axis=1)  # (p,)
    power[0] = 0.0  # drop the DC component
    folded = np.array([power[k] + power[p - k] for k in range(1, p // 2 + 1)])
    total = folded.sum()
    return folded / total if total > 0 else folded


def concentration(spectrum, top=2):
    """Fraction of embedding energy in the `top` strongest frequencies (1.0 = perfectly sparse,
    ~top/len = diffuse/memorizing)."""
    return float(np.sort(spectrum)[::-1][:top].sum())


def report(p=11, top=2, steps=20000, seed=0):
    configs = [
        ("GROKKED   (quad, wd=1)", dict(act="quad", wd=1.0)),
        ("MEMORIZED (relu, wd=1)", dict(act="relu", wd=1.0)),
        ("MEMORIZED (quad, wd=0)", dict(act="quad", wd=0.0)),
    ]
    diffuse = top / (p // 2)
    print("Fourier alignment to Gromov's analytic solution (p=%d, top-%d energy; diffuse baseline ~%.2f)\n"
          % (p, top, diffuse))
    rows = []
    for tag, kw in configs:
        history, params = modadd.train(p=p, steps=steps, seed=seed, **kw)
        val = modadd.summarize(history)["final_val"]
        spec = embedding_spectrum(params["E"], p)
        conc = concentration(spec, top)
        rows.append((tag, val, conc, spec))
        print("%-24s val=%.2f  top-%d energy=%.2f  spectrum=%s"
              % (tag, val, top, conc, np.round(spec, 2)))
    grok_conc = rows[0][2]
    ok = grok_conc >= 0.8 and all(r[2] < grok_conc for r in rows[1:])
    print("\n%s the grokked model's embedding is sparse in Fourier space (converged to the analytic"
          "\nstructure); the memorized models are not." % ("PASS:" if ok else "CHECK:"))
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if report() else 1)
