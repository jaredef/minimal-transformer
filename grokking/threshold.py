"""Grokking as a threshold-induced property.

Hold the model and optimizer fixed and vary only the fraction of the p*p pairs used for training.
Below a data threshold the model memorizes (held-out accuracy stays near chance forever); above it
the model groks (fits early, then generalizes to 100% thousands of steps later). The transition is
sharp, which is what "threshold-induced" means: the property is not present a little at frac 0.5 and
more at frac 0.7 -- it is absent, then present.
"""
import modadd


def report(fracs=(0.4, 0.5, 0.7, 0.9), steps=20000, seed=0):
    print("Data-fraction threshold (p=11, d=64, H=256, quad, wd=1)\n")
    grokked = {}
    for frac in fracs:
        s = modadd.summarize(modadd.train(act="quad", wd=1.0, frac=frac, steps=steps, seed=seed)[0])
        g = s["grok_at"] is not None
        grokked[frac] = g
        print("frac=%.1f  fit@%-6s grok@%-7s final_val=%.2f  ->  %s"
              % (frac, s["fit_at"], s["grok_at"], s["final_val"], "GROKS" if g else "memorizes"))
    low = [f for f in fracs if not grokked[f]]
    high = [f for f in fracs if grokked[f]]
    ok = bool(low) and bool(high) and max(low) < min(high)
    print("\n%s a sharp transition: memorizes up to frac=%s, groks from frac=%s."
          % ("PASS:" if ok else "CHECK:",
             max(low) if low else "-", min(high) if high else "-"))
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if report() else 1)
