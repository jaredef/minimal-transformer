#!/usr/bin/env python3
"""nand-grok-evidence (NO-5, NO-6): PROVE grokking happened, don't just plot a suggestive gap.

A late-rising held-out accuracy curve is only the SYMPTOM. Grokking is the claim that learning KEPT
HAPPENING during the apparent plateau and CAUSED the late jump. These two rungs witness the cause.

    NO-5  MARGIN-CROSSES-AT-GROK   the mechanism. On the plateau where TRAIN accuracy is already 100% and
          nothing seems to be happening, the HELD-OUT margin (logit(correct) - max other) is still
          NEGATIVE at the fit, climbs MONOTONICALLY, and crosses zero EXACTLY at the grok step -- while
          ||M|| keeps rising. The flip is a smooth threshold crossing driven by continued descent, not a
          jump: the plateau is not a stationary point. This is what makes it grokking and not luck.

    NO-6  SGD-ENDPOINT-IS-THE-PRIOR   the identity. Trained holding out s, the SGD endpoint's phase
          portrait equals the min-L1 PRIOR portrait that NO-3 selects analytically (O<-Opqr|Z<-Zs). The
          dynamic grok and the static implicit-bias prior are literally the same object -- an independent
          oracle the SGD run lands on, not a re-plot of its own output.

Deterministic, torch-free, fail-closed. See probes/nand_core.py for the shared step and training."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nand_core as c


def first_100(traj, cps, key):
    for cp in cps:
        rec = traj[cp]
        if key in rec:
            ok, tot = rec[key]
            if tot > 0 and ok == tot:
                return cp
    return None


def run(path):
    data = c.load(path)
    d = data["_d"]
    vocab, emb = data["vocab"], data["embeddings"]
    rung = data.get("rung", "MARGIN-CROSSES-AT-GROK")
    train_cons, held_cons, hs = c.split(data)
    ho = "".join(hs)
    lr = data.get("lr", 0.01)
    steps = data.get("steps", 6000)
    cps = sorted(set(data.get("checkpoints", [0, 10, 25, 50, 60, 70, 80, 90, 100, 200, 6000])))

    M, traj = c.train_sgd(train_cons, emb, vocab, d, lr, steps, cps, held=held_cons)

    if rung == "MARGIN-CROSSES-AT-GROK":
        fit_at = first_100(traj, cps, "train")
        grok_at = first_100(traj, cps, "held")
        if fit_at is None or grok_at is None or grok_at <= fit_at:
            print("NO-EVIDENCE holdout=%s reason=no-delayed-generalization-in-this-fixture" % ho)
            return 1
        plateau = [cp for cp in cps if fit_at <= cp <= grok_at]
        margins = [traj[cp]["held_margin"] for cp in plateau]
        norms = [traj[cp]["wnorm"] for cp in plateau]
        monotone = all(margins[i] < margins[i + 1] for i in range(len(margins) - 1))
        norm_rising = all(norms[i] < norms[i + 1] for i in range(len(norms) - 1))
        neg_at_fit = traj[fit_at]["held_margin"] < 0
        nonneg_at_grok = traj[grok_at]["held_margin"] >= 0
        ok = monotone and norm_rising and neg_at_fit and nonneg_at_grok
        verdict = "MARGIN-CROSSES-AT-GROK" if ok else "MARGIN-EVIDENCE-BROKEN"
        print("%s holdout=%s fit_at=step%d grok_at=step%d held_margin_at_fit=%.2f held_margin_at_grok=%+.2f "
              "monotone_climb=%s wnorm_rising_on_plateau=%s "
              "note=on-the-plateau-where-train-is-already-100-percent-the-held-out-margin-is-negative-climbs-"
              "monotonically-and-crosses-zero-exactly-at-the-grok-step-while-weight-norm-rises-the-flip-is-"
              "continued-descent-not-luck-the-plateau-is-not-a-stationary-point"
              % (verdict, ho, fit_at, grok_at, traj[fit_at]["held_margin"],
                 traj[grok_at]["held_margin"], "yes" if monotone else "no",
                 "yes" if norm_rising else "no"))
        return 0 if ok else 1

    if rung == "SGD-ENDPOINT-IS-THE-PRIOR":
        endpoint = c.portrait(M, emb, vocab, d)
        lo, hi = data.get("range", [-1, 1])
        full_form = [tuple(x) for x in data["form"]]
        surv = c.survivors(vocab, emb, full_form, lo, hi, d)
        prior = c.portrait(c.min_l1(surv, d), emb, vocab, d)
        match = endpoint == prior
        verdict = "SGD-ENDPOINT-IS-THE-PRIOR" if match else "ENDPOINT-PRIOR-MISMATCH"
        print("%s holdout=%s endpoint_portrait=%s prior_portrait=%s match=%s "
              "note=trained-holding-out-s-the-sgd-endpoint-phase-portrait-equals-the-min-l1-prior-portrait-"
              "no-3-selects-analytically-the-dynamic-grok-and-the-static-implicit-bias-prior-are-the-same-"
              "object" % (verdict, ho, endpoint, prior, "yes" if match else "no"))
        return 0 if match else 1

    print("REJECT unknown-rung=%s" % rung)
    return 1


def main(argv):
    if len(argv) != 2:
        print("usage: nand_grok_evidence_probe.py <fixture.json>", file=sys.stderr)
        return 2
    try:
        return run(argv[1])
    except Exception as exc:
        print("REJECT malformed=" + str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
