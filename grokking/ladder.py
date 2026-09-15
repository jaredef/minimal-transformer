"""The constraint ladder, with ablation as the falsifier.

Grokking here is not one setting; it is a *threshold-induced property* that appears only once
the right constraints have accumulated. This script demonstrates that by ablation: at a fixed
grokking-scale operating point, remove one constraint at a time and show the property vanishes.

  reference        quadratic activation + weight decay  -> GROKS (val -> 1.0, norm turns over)
  ablate C_a       remove weight decay                  -> memorizes (val ~ 0, norm grows)
  ablate C_d       quadratic -> ReLU                    -> memorizes (val ~ 0, norm grows)

Each ablation that kills grokking proves its constraint is load-bearing. (C_b, enough data, is
shown separately by the data-fraction threshold in threshold.py.) This is Fielding's ablation
test: remove a constraint, assert its induced property is gone, not merely reduced.
"""
import modadd


def _line(tag, history):
    s = modadd.summarize(history)
    print("%-30s fit@%-6s grok@%-7s final_val=%.2f  norm %.0f->%.0f  turned_over=%s"
          % (tag, s["fit_at"], s["grok_at"], s["final_val"],
             s["norm_start"], s["norm_end"], s["norm_turned_over"]))
    return s


def report(steps=20000, seed=0):
    print("Ablation falsifier at the grokking operating point (p=11, d=64, H=256, frac=0.7)\n")
    ref = _line("reference (quad, wd=1)",
                modadd.train(act="quad", wd=1.0, steps=steps, seed=seed)[0])
    abl_a = _line("ablate C_a: no weight decay",
                  modadd.train(act="quad", wd=0.0, steps=steps, seed=seed)[0])
    abl_d = _line("ablate C_d: ReLU not quad",
                  modadd.train(act="relu", wd=1.0, steps=steps, seed=seed)[0])
    grokked = ref["grok_at"] is not None
    vanished = abl_a["grok_at"] is None and abl_d["grok_at"] is None
    ok = grokked and vanished
    print("\n%s reference groks and each ablation removes it -- both constraints are load-bearing."
          % ("PASS:" if ok else "CHECK:"))
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if report() else 1)
