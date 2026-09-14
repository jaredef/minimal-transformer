#!/usr/bin/env python3
"""transformer-nand-orbit (NO-1..3): the NAND gate, read as the ORBIT of a lowered transformer. A lowered transformer is a deterministic
dynamical system (enumerable-transformer paper, section 4.12): the compiled weight is the dynamics, the seed is the initial condition, the
orbit is the generation, and the reachable behaviours are the attractors. Encoding NAND as a next-token map over 6 tokens -- the four input
rows p,q,r,s = (00,01,10,11) and the two output bits O=1, Z=0, with the outputs as fixed points -- and compiling the min-L1 integer weight
that realises it (no gradient descent, just the enumerable bench), the PHASE PORTRAIT is exactly the truth table: two fixed-point attractors
(the two output values), whose BASINS are the NAND preimages -- three rows flow to O (NAND=1), the one row s=(1,1) flows to Z (NAND=0), the 3:1
asymmetry that IS NAND. And the grokking is prior-conditional: hold out the fourth row and the other three do not FORCE it (the survivor set
disagrees), yet the min-L1 weight -- the implicit bias -- selects the generalising dynamics (s->Z) anyway: selection is the prior (section 4.10),
the exact dynamical-systems companion of the statistical grok watched on the 30B (transformer-mp-30b-traj). Deterministic, torch-free.

    NO-1  THE-NAND-ORBIT  compiled to a weight, NAND's phase portrait is two fixed-point attractors (the output bits O,Z); every input row seeded flows in one step to its correct output and stays -- generation is the orbit, the reachable behaviours are exactly the attractors.
    NO-2  THE-BASINS-ARE-THE-TRUTH-TABLE  the basin of O is {p,q,r} (the rows with NAND=1) and the basin of Z is {s} (the one row with NAND=0); the dynamical basins reproduce the NAND truth table exactly -- the 3:1 basin asymmetry IS the gate.
    NO-3  GROK-IS-PRIOR-CONDITIONAL  hold out row s; the other three do NOT force it (survivors send s to more than one place), yet the min-L1 weight -- the prior -- selects the generalising dynamics s->Z; selection is the prior, grok via implicit bias, the exact companion of the 30B statistical grok.
"""
import itertools, json, os, sys
def add_id(w): n=len(w); return [[w[r][c]+(1 if r==c else 0) for c in range(n)] for r in range(n)]
def mv(m,v): return [sum(m[r][i]*v[i] for i in range(len(v))) for r in range(len(m))]
def dot(a,b): return sum(x*y for x,y in zip(a,b))
def argmax(vs): return max(range(len(vs)),key=lambda i:(vs[i],-i))
def predict(vocab,emb,M,last): res=mv(add_id(M),emb[last]); return vocab[argmax([dot(emb[t],res) for t in vocab])]
def l1(m): return sum(abs(x) for row in m for x in row)
def survivors(vocab,emb,form,lo,hi):
    w=len(next(iter(emb.values()))); vals=list(range(lo,hi+1)); out=[]
    for row in itertools.product(vals,repeat=w*w):
        M=[list(row[r*w:(r+1)*w]) for r in range(w)]
        if all(predict(vocab,emb,M,c)==n for c,n in form): out.append(M)
    return out
def minL1(surv): return min(surv,key=lambda m:(l1(m),tuple(x for row in m for x in row)))
def orbit(vocab,emb,M,seed):
    seen={}; order=[]; cur=seed
    while cur not in seen: seen[cur]=len(order); order.append(cur); cur=predict(vocab,emb,M,cur)
    return order[seen[cur]:]
def portrait(vocab,emb,M):
    b={}
    for s in vocab:
        att="".join(sorted(set(orbit(vocab,emb,M,s)))); b.setdefault(att,[]).append(s)
    return b
def main(path):
    d=json.load(open(path)); rung=d["rung"]; V=d["vocab"]; E=d["embeddings"]; F=[tuple(x) for x in d["form"]]
    lo,hi=d["range"]; ho=d["holdout"]
    surv=survivors(V,E,F,lo,hi); M=minL1(surv); B=portrait(V,E,M)
    inputs=[t for t in V if t not in ("Z","O")]
    if rung=="THE-NAND-ORBIT":
        atts=sorted(B); pt="|".join("%s<-%s"%(a,"".join(sorted(B[a]))) for a in atts)
        fixed=all(len(orbit(V,E,M,o))==1 for o in ("Z","O"))
        print("THE-NAND-ORBIT survivors=%d attractors=%d fixed_points=%s portrait=%s note=compiled-to-a-weight-nands-phase-portrait-is-two-fixed-point-attractors-the-output-bits-every-input-row-seeded-flows-in-one-step-to-its-correct-output-and-stays-generation-is-the-orbit-the-reachable-behaviours-are-exactly-the-attractors"%(len(surv),len(atts),fixed,pt))
    elif rung=="THE-BASINS-ARE-THE-TRUTH-TABLE":
        def NAND(a,b): return 1-(a&b)
        rowmap={"p":(0,0),"q":(0,1),"r":(1,0),"s":(1,1)}
        basinO=sorted(t for t in inputs if "O" in [k for k in B if t in B[k]][0])
        basinZ=sorted(t for t in inputs if "Z" in [k for k in B if t in B[k]][0])
        ok=all((("O" if t in basinO else "Z")==("O" if NAND(*rowmap[t]) else "Z")) for t in inputs)
        print("THE-BASINS-ARE-THE-TRUTH-TABLE basin_O=%s basin_Z=%s matches_nand_truth_table=%s note=the-basin-of-o-is-the-rows-with-nand-1-and-the-basin-of-z-is-the-one-row-with-nand-0-the-dynamical-basins-reproduce-the-nand-truth-table-exactly-the-3-to-1-basin-asymmetry-is-the-gate"%("".join(basinO),"".join(basinZ),ok))
    elif rung=="GROK-IS-PRIOR-CONDITIONAL":
        F3=[c for c in F if c[0]!=ho]; s3=survivors(V,E,F3,lo,hi)
        dests=sorted(set(predict(V,E,m,ho) for m in s3)); prior=predict(V,E,minL1(s3),ho)
        forced=(len(dests)==1); target="Z" if ho=="s" else "?"
        print("GROK-IS-PRIOR-CONDITIONAL holdout=%s survivors_on_rest=%d held_dests=%s forced=%s prior_selects=%s generalises=%s note=hold-out-row-s-the-other-three-do-not-force-it-survivors-send-it-to-more-than-one-place-yet-the-min-l1-weight-the-prior-selects-the-generalising-dynamics-s-to-z-selection-is-the-prior-grok-via-implicit-bias-the-exact-companion-of-the-30b-statistical-grok"%(ho,len(s3),"".join(dests),forced,prior,prior==target))
    else: raise SystemExit("unknown "+rung)
main(sys.argv[1])
