"""Precompute the grokking visualization data for the server.

Trains the grokking model (and a memorized control) once and dumps a compact JSON trajectory the
stdlib server can serve without importing numpy: per-checkpoint train/val accuracy, weight norm,
and the embedding Fourier spectrum (which sharpens from diffuse to a few frequencies as the model
groks). Run this whenever you want to refresh the data:

    python3 grokking/build_viz.py            # writes server/grok_viz.json
"""
import json
import os

import modadd
import fourier

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "server", "grok_viz.json")

P, D, H, FRAC, STEPS, SEED, CKPT = 11, 64, 256, 0.7, 20000, 0, 250


def trajectory(act, wd):
    rows = []

    def rec(step, params, ta, va, wn):
        spec = fourier.embedding_spectrum(params["E"], P).tolist()
        rows.append({
            "step": step, "train_acc": round(ta, 4), "val_acc": round(va, 4),
            "norm": round(wn, 3), "spectrum": [round(x, 4) for x in spec],
            "top2": round(fourier.concentration(fourier.embedding_spectrum(params["E"], P), 2), 4),
        })

    hist, _ = modadd.train(act=act, p=P, d=D, H=H, frac=FRAC, wd=wd, steps=STEPS, seed=SEED,
                           ckpt=CKPT, on_ckpt=rec)
    s = modadd.summarize(hist)
    return rows, s


def main():
    grok_rows, gs = trajectory("quad", 1.0)
    mem_rows, ms = trajectory("relu", 1.0)
    data = {
        "meta": {
            "p": P, "d": D, "H": H, "frac": FRAC, "steps": STEPS, "seed": SEED,
            "n_train": int(FRAC * P * P), "n_val": P * P - int(FRAC * P * P),
            "n_params": P * D + 2 * D * H + H * P,
            "diffuse_baseline": round(2.0 / (P // 2), 4),
            "grok": {"fit_at": gs["fit_at"], "grok_at": gs["grok_at"],
                     "final_val": gs["final_val"]},
        },
        "grokked": grok_rows,
        "memorized": mem_rows,
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(data, fh, separators=(",", ":"))
    print("wrote %s  (%d grokked checkpoints, fit@%s grok@%s final_val=%.2f, %d params)"
          % (os.path.relpath(OUT), len(grok_rows), gs["fit_at"], gs["grok_at"],
             gs["final_val"], data["meta"]["n_params"]))


if __name__ == "__main__":
    main()
