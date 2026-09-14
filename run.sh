#!/usr/bin/env sh
# minimal-transformer: the NAND gate as the orbit of a lowered transformer, its basins as the truth
# table, and grokking made visible -- first as a prior-conditional static (NO-1..3), then as a
# training dynamic under real SGD (NO-4). Deterministic, torch-free, fail-closed. Run: sh run.sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PY=${PYTHON:-python3}

orbit="$PY $ROOT/probes/nand_orbit_probe.py"
grok="$PY $ROOT/probes/nand_grok_sgd_probe.py"
evid="$PY $ROOT/probes/nand_grok_evidence_probe.py"

check() {
  probe=$1; fixture=$2; expected=$3
  output=$($probe "$fixture")
  printf '%s\n' "$output"
  if [ "$output" != "$expected" ]; then
    printf 'expected: %s\nactual:   %s\n' "$expected" "$output" >&2
    exit 1
  fi
}

echo "== NO-1..3  the static orbit: gate as phase portrait, basins as truth table, prior-conditional grok =="
check "$orbit" "$ROOT/fixtures/the-nand-orbit.json" 'THE-NAND-ORBIT survivors=1 attractors=2 fixed_points=True portrait=O<-Opqr|Z<-Zs note=compiled-to-a-weight-nands-phase-portrait-is-two-fixed-point-attractors-the-output-bits-every-input-row-seeded-flows-in-one-step-to-its-correct-output-and-stays-generation-is-the-orbit-the-reachable-behaviours-are-exactly-the-attractors'
check "$orbit" "$ROOT/fixtures/the-basins-are-the-truth-table.json" 'THE-BASINS-ARE-THE-TRUTH-TABLE basin_O=pqr basin_Z=s matches_nand_truth_table=True note=the-basin-of-o-is-the-rows-with-nand-1-and-the-basin-of-z-is-the-one-row-with-nand-0-the-dynamical-basins-reproduce-the-nand-truth-table-exactly-the-3-to-1-basin-asymmetry-is-the-gate'
check "$orbit" "$ROOT/fixtures/grok-is-prior-conditional.json" 'GROK-IS-PRIOR-CONDITIONAL holdout=s survivors_on_rest=2 held_dests=Zs forced=False prior_selects=Z generalises=True note=hold-out-row-s-the-other-three-do-not-force-it-survivors-send-it-to-more-than-one-place-yet-the-min-l1-weight-the-prior-selects-the-generalising-dynamics-s-to-z-selection-is-the-prior-grok-via-implicit-bias-the-exact-companion-of-the-30b-statistical-grok'

echo
echo "== NO-4  the training dynamic: the same grok, now delayed generalization under real SGD =="
check "$grok" "$ROOT/fixtures/nand-groks-in-time.json" 'GROKS holdout=s train_100_at=step50 heldout_100_at=step100 gap=yes note=train-fits-early-held-out-reaches-100-percent-later-s-is-not-box-forced-yet-sgd-implicit-max-margin-bias-selects-s-to-z-the-dynamic-image-of-no-3-min-l1-prior-delayed-generalization-is-grokking'
check "$grok" "$ROOT/fixtures/nand-generalises-immediately.json" 'NO-GROK holdout=q train_100_at=step50 heldout_100_at=step25 gap=no note=the-held-out-row-is-forced-by-the-remaining-rows-so-it-generalizes-as-soon-as-or-before-the-data-is-fit-no-implicit-bias-phase-needed-no-grok-gap'
check "$grok" "$ROOT/fixtures/nand-memorizes.json" 'MEMORIZES holdout=pqr train_100_at=step50 heldout_100_at=never reason=not-forced-by-train note=the-remaining-train-rows-do-not-force-the-held-out-underdetermined-nothing-pulls-them-to-the-right-output-gd-memorizes-and-never-generalizes'

echo
echo "== NO-5..6  the evidence: grokking PROVED, not just plotted =="
check "$evid" "$ROOT/fixtures/nand-margin-crossing.json" 'MARGIN-CROSSES-AT-GROK holdout=s fit_at=step50 grok_at=step100 held_margin_at_fit=-0.91 held_margin_at_grok=+0.09 monotone_climb=yes wnorm_rising_on_plateau=yes note=on-the-plateau-where-train-is-already-100-percent-the-held-out-margin-is-negative-climbs-monotonically-and-crosses-zero-exactly-at-the-grok-step-while-weight-norm-rises-the-flip-is-continued-descent-not-luck-the-plateau-is-not-a-stationary-point'
check "$evid" "$ROOT/fixtures/nand-endpoint-is-prior.json" 'SGD-ENDPOINT-IS-THE-PRIOR holdout=s endpoint_portrait=O<-Opqr|Z<-Zs prior_portrait=O<-Opqr|Z<-Zs match=yes note=trained-holding-out-s-the-sgd-endpoint-phase-portrait-equals-the-min-l1-prior-portrait-no-3-selects-analytically-the-dynamic-grok-and-the-static-implicit-bias-prior-are-the-same-object'

echo
echo "minimal-transformer: all modes green"
