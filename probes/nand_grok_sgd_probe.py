#!/usr/bin/env python3
"""nand-grok-sgd (NO-4): grokking on the NAND, as a training DYNAMIC in time.

NO-1..3 (`nand_orbit_probe.py`) read the NAND gate as the ORBIT of a lowered transformer and showed the
grok in the STATICS: hold out the fourth row s=(1,1), the other three do NOT force it (the survivor set
sends s to more than one place), yet the min-L1 weight -- the implicit-bias PRIOR -- selects the
generalising dynamics s->Z (`GROK-IS-PRIOR-CONDITIONAL`). That is grok as a property of the phase portrait.

NO-4 shows the SAME phenomenon as a DYNAMIC of actual gradient descent, on the SAME 6-token NAND encoding
and the SAME embeddings. We split the NAND form into a TRAIN subset and a HELD-OUT remainder and train the
lowered transformer's weight M by SGD -- the exact DYN-1 gradient of the enumerable transformer,
(p_u - onehot) * emb[u][r] * emb[s][c], tied embeddings (unemb == emb), zero init -- on TRAIN ONLY. At
fixed checkpoints we record TRAIN accuracy (fraction of train rows the argmax satisfies) and HELD-OUT
accuracy. The control is exactly NO-3's control and GRK-1's: whether the train subset FORCES the held-out.

    NO-4a  GROKS      hold out s (NAND=0, the minority row): train fits EARLY, held-out reaches 100%
                      LATER -- a genuine gap between fitting and generalising. s is not box-forced by the
                      other rows, yet SGD's implicit (max-margin) bias selects s->Z anyway -- the dynamic
                      image of NO-3's min-L1 prior. Delayed generalisation: grokking.
    NO-4b  NO-GROK    hold out a redundant majority row (r, NAND=1): it IS forced by p,q, so it
                      generalises as soon as (here before) the data is fit -- no delay, no grok gap.
    NO-4c  MEMORIZES  hold out ALL three majority rows (p,q,r): the remainder (only the fixed points
                      O->O, Z->Z) does NOT force them -- underdetermined, nothing pulls the held-out rows
                      to the right output. Train fits; held-out NEVER reaches 100% within the budget.

Deterministic: fixed zero init, fixed learning rate, fixed step budget, fixed checkpoints, tied embeddings.
Accuracies are exact fractions; the "at step" facts are the FIRST checkpoint crossing 100% -- robust to
float roundoff. Torch-free. Companion of the statics NO-1..3 and of the closure-bench grok GRK-1.
"""
import json
import math
import sys


def logits(M, s, emb, vocab, d):
    # lowered transformer, tied embeddings: logit(t) = emb[t] . (I + M) . emb[s]
    im = [[M[r][c] + (1 if r == c else 0) for c in range(d)] for r in range(d)]
    v = [sum(im[r][c] * emb[s][c] for c in range(d)) for r in range(d)]
    return [sum(emb[t][r] * v[r] for r in range(d)) for t in vocab]


def softmax(z):
    m = max(z)
    ex = [math.exp(x - m) for x in z]
    tot = sum(ex)
    return [e / tot for e in ex]


def argmax(z):
    return max(range(len(z)), key=lambda i: (z[i], -i))


def accuracy(M, cons, emb, vocab, d):
    if not cons:
        return (0, 0)
    ok = sum(1 for s, t in cons if vocab[argmax(logits(M, s, emb, vocab, d))] == t)
    return (ok, len(cons))


def train(train_cons, held_cons, emb, vocab, d, lr, steps, checkpoints):
    # REUSE of the enumerable transformer's DYN-1 gradient with tied embeddings (unemb == emb).
    M = [[0.0] * d for _ in range(d)]
    snaps = {}
    for step in range(steps + 1):
        if step in checkpoints:
            snaps[step] = (accuracy(M, train_cons, emb, vocab, d),
                           accuracy(M, held_cons, emb, vocab, d))
        grad = [[0.0] * d for _ in range(d)]
        for s, t in train_cons:
            p = softmax(logits(M, s, emb, vocab, d))
            ti = vocab.index(t)
            for u in range(len(vocab)):
                g = p[u] - (1.0 if u == ti else 0.0)
                for r in range(d):
                    for c in range(d):
                        grad[r][c] += g * emb[vocab[u]][r] * emb[s][c]
        for r in range(d):
            for c in range(d):
                M[r][c] -= lr * grad[r][c]
    return snaps


def first_100(snaps, cps, index):
    for cp in cps:
        ok, tot = snaps[cp][index]
        if tot > 0 and ok == tot:
            return cp
    return None


def render_curve(snaps, cps, train_at, held_at):
    # ASCII accuracy curves (train vs held-out) over checkpoints, on stderr so stdout stays the
    # single machine-readable verdict line. The grok is the horizontal gap between the two 100% marks.
    width = 20  # columns of the accuracy bar (0..100%)
    lines = ["", "  step |  train acc            | held-out acc          ",
             "  -----+-----------------------+-----------------------"]
    for cp in cps:
        (to, tt), (ho, ht) = snaps[cp]
        tf = to / tt if tt else 0.0
        hf = ho / ht if ht else 0.0
        tb = "#" * round(tf * width) + "." * (width - round(tf * width))
        hb = "#" * round(hf * width) + "." * (width - round(hf * width))
        tmark = " <-fit" if cp == train_at else ""
        hmark = " <-grok" if cp == held_at else ""
        lines.append("  %5d| %s %3d%%%s| %s %3d%%%s"
                     % (cp, tb, round(tf * 100), "".ljust(0), hb, round(hf * 100), tmark + hmark))
    if train_at is not None and held_at is not None and held_at > train_at:
        lines.append("  gap: train hits 100%% at step%d, held-out only at step%d -- delayed generalization (grok)"
                     % (train_at, held_at))
    lines.append("")
    print("\n".join(lines), file=sys.stderr)


def run(path):
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    vocab = data["vocab"]
    emb = data["embeddings"]
    d = len(next(iter(emb.values())))
    form = [tuple(x) for x in data["form"]]
    held = data["holdout"]
    held_syms = held if isinstance(held, list) else [held]
    train_cons = [(a, b) for a, b in form if a not in held_syms]
    held_cons = [(a, b) for a, b in form if a in held_syms]
    lr = data.get("lr", 0.01)
    steps = data.get("steps", 6000)
    checkpoints = sorted(set(data.get("checkpoints", [0, 10, 25, 50, 100, 200, 400, 800, 1600, 3200, 6000])))

    snaps = train(train_cons, held_cons, emb, vocab, d, lr, steps, checkpoints)
    cps = sorted(snaps)
    train_at = first_100(snaps, cps, 0)
    held_at = first_100(snaps, cps, 1)
    ho = "".join(held_syms)
    render_curve(snaps, cps, train_at, held_at)

    if train_at is None:
        print("NO-FIT holdout=%s train never fits reason=unsatisfiable" % ho)
        return 1
    train_str = "step%d" % train_at
    if held_at is None:
        print("MEMORIZES holdout=%s train_100_at=%s heldout_100_at=never reason=not-forced-by-train "
              "note=the-remaining-train-rows-do-not-force-the-held-out-underdetermined-nothing-pulls-them-"
              "to-the-right-output-gd-memorizes-and-never-generalizes" % (ho, train_str))
        return 0
    held_str = "step%d" % held_at
    if held_at > train_at:
        print("GROKS holdout=%s train_100_at=%s heldout_100_at=%s gap=yes "
              "note=train-fits-early-held-out-reaches-100-percent-later-s-is-not-box-forced-yet-sgd-implicit-"
              "max-margin-bias-selects-s-to-z-the-dynamic-image-of-no-3-min-l1-prior-delayed-generalization-"
              "is-grokking" % (ho, train_str, held_str))
    else:
        print("NO-GROK holdout=%s train_100_at=%s heldout_100_at=%s gap=no "
              "note=the-held-out-row-is-forced-by-the-remaining-rows-so-it-generalizes-as-soon-as-or-before-"
              "the-data-is-fit-no-implicit-bias-phase-needed-no-grok-gap" % (ho, train_str, held_str))
    return 0


def main(argv):
    if len(argv) != 2:
        print("usage: nand_grok_sgd_probe.py <fixture.json>", file=sys.stderr)
        return 2
    try:
        return run(argv[1])
    except Exception as exc:
        print("REJECT malformed=" + str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
