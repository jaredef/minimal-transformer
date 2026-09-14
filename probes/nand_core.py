#!/usr/bin/env python3
"""Shared core for the NAND minimal-transformer probes and the visualization server.

A minimal one-layer transformer over a 6-token encoding of the NAND truth table: token x has an embedding
emb[x] in Z^d, and one step is
    f(x) = argmax_v  emb[v] . (I + M) . emb[x]        (tied embeddings: unembed == embed)
The weight M is obtained two ways: the minimum-L1-norm weight found by exhaustive search over the small
integer weight space, or a weight learned by gradient descent. This module provides both, plus the training
trajectory (accuracy, held-out margin, ||M||) and the phase portrait, so every probe and the server report
the same numbers. Deterministic, and pure-Python with no machine-learning libraries."""
import itertools
import json
import math
import random


# ---- the minimal-transformer step -------------------------------------------------

def add_id(M, d):
    return [[M[r][c] + (1 if r == c else 0) for c in range(d)] for r in range(d)]


def logits(M, s, emb, vocab, d):
    im = add_id(M, d)
    v = [sum(im[r][c] * emb[s][c] for c in range(d)) for r in range(d)]
    return [sum(emb[t][r] * v[r] for r in range(d)) for t in vocab]


def softmax(z):
    m = max(z)
    ex = [math.exp(x - m) for x in z]
    tot = sum(ex)
    return [e / tot for e in ex]


def argmax(z):
    return max(range(len(z)), key=lambda i: (z[i], -i))


def predict(M, s, emb, vocab, d):
    return vocab[argmax(logits(M, s, emb, vocab, d))]


def accuracy(M, cons, emb, vocab, d):
    if not cons:
        return (0, 0)
    ok = sum(1 for s, t in cons if predict(M, s, emb, vocab, d) == t)
    return (ok, len(cons))


def margin(M, s, t, emb, vocab, d):
    # logit(correct) - max other logit; > 0 iff the argmax is correct. The held-out margin crossing zero
    # is the exact moment generalization happens.
    z = logits(M, s, emb, vocab, d)
    ti = vocab.index(t)
    others = [z[i] for i in range(len(vocab)) if i != ti]
    return z[ti] - max(others)


def fro_norm(M, d):
    return math.sqrt(sum(M[r][c] * M[r][c] for r in range(d) for c in range(d)))


# ---- the two ways to get M --------------------------------------------------------

def train_sgd(train_cons, emb, vocab, d, lr, steps, checkpoints, held=None, seed=None, init_scale=0.15):
    """Gradient descent with the softmax cross-entropy gradient (p_u - onehot)*emb[u]*emb[s], tied
    embeddings. seed=None gives a zero initialization (the deterministic default the tests assert); an
    integer seed gives a small deterministic random initialization, so 'train again' starts from a fresh
    point and still generalizes. Returns (final_M, trajectory) recording the state at each checkpoint."""
    if seed is None:
        M = [[0.0] * d for _ in range(d)]
    else:
        rng = random.Random(seed)
        M = [[rng.uniform(-init_scale, init_scale) for _ in range(d)] for _ in range(d)]
    cps = sorted(set(checkpoints))
    traj = {}
    for step in range(steps + 1):
        if step in cps:
            rec = {"train": accuracy(M, train_cons, emb, vocab, d), "wnorm": fro_norm(M, d),
                   "M": [row[:] for row in M]}
            if held:
                rec["held"] = accuracy(M, held, emb, vocab, d)
                # worst-case held margin: >= 0 iff every held row is correct
                rec["held_margin"] = min(margin(M, s, t, emb, vocab, d) for s, t in held)
            traj[step] = rec
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
    return M, traj


def survivors(vocab, emb, form, lo, hi, d):
    vals = list(range(lo, hi + 1))
    out = []
    for row in itertools.product(vals, repeat=d * d):
        M = [list(row[r * d:(r + 1) * d]) for r in range(d)]
        if all(predict(M, s, emb, vocab, d) == n for s, n in form):
            out.append(M)
    return out


def l1(M, d):
    return sum(abs(M[r][c]) for r in range(d) for c in range(d))


def min_l1(surv, d):
    return min(surv, key=lambda m: (l1(m, d), tuple(x for row in m for x in row)))


# ---- phase portrait ---------------------------------------------------------------

def orbit(M, seed, emb, vocab, d):
    seen, order, cur = {}, [], seed
    while cur not in seen:
        seen[cur] = len(order)
        order.append(cur)
        cur = predict(M, cur, emb, vocab, d)
    return order[seen[cur]:]


def portrait(M, emb, vocab, d):
    b = {}
    for s in vocab:
        att = "".join(sorted(set(orbit(M, s, emb, vocab, d))))
        b.setdefault(att, []).append(s)
    return "|".join("%s<-%s" % (a, "".join(sorted(b[a]))) for a in sorted(b))


# ---- fixture loading --------------------------------------------------------------

def load(path):
    with open(path, "r", encoding="utf-8") as h:
        data = json.load(h)
    data["_d"] = len(next(iter(data["embeddings"].values())))
    return data


def split(data):
    form = [tuple(x) for x in data["form"]]
    held = data.get("holdout", "s")
    hs = held if isinstance(held, list) else [held]
    train_cons = [(a, b) for a, b in form if a not in hs]
    held_cons = [(a, b) for a, b in form if a in hs]
    return train_cons, held_cons, hs
