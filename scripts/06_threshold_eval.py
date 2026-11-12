"""Stage 06 (gate P6): threshold, held-out evaluation, bootstrap CIs, plots.

Protocol: calibrate the abstention threshold on VAL p_wrong so that 70% of val
questions are answered; freeze it; apply to TEST (in-distribution TriviaQA pool)
and OOD (NQ-Open).  Headline = accuracy on answered questions at 70% coverage vs
base accuracy (answer-everything), with bootstrap 95% CI on the gain.  AUROC is
also reported for the probe and both logit baselines on test and OOD.

If OOD degrades, we report it -- every number lands in results/metrics.json and
the script prints an explicit "OOD degradation" line when acc@70 drops there.

Writes results/metrics.json, results/roc_test.png, results/reliability_val.png.
Gate P6: realised coverage on test and OOD printed next to the 70% target.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, ".")
import config  # noqa: E402
from common import (accuracy_at_threshold, auroc, bootstrap_ci,  # noqa: E402
                    calibrate_threshold_for_coverage)


def load_split(split):
    X = np.load(config.acts_path(split)).astype(np.float32)
    y = np.load(os.path.join(config.ACTS_DIR, f"{split}_y.npy")).astype(np.int8)
    lp = np.load(config.acts_logp_path(split))
    return X, y, lp


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    config.RunFlags.add_args(ap)
    args = ap.parse_args(argv)
    fl = config.RunFlags.from_args(args)

    import joblib

    bundle = joblib.load(config.PROBE_PATH)
    probe, layer = bundle["probe"], bundle["layer"]

    def p_wrong(X):
        return probe.predict_proba(X[:, layer, :])[:, 1]

    Xva, yva, lpva = load_split("val")
    thr = calibrate_threshold_for_coverage(yva, p_wrong(Xva), config.TARGET_COVERAGE)

    metrics = {
        "label_convention": config.LABEL_CONVENTION,
        "model_id": fl.model_id,
        "target_coverage": config.TARGET_COVERAGE,
        "threshold_p_wrong": float(thr),
        "probe": {"layer": int(layer), "C": bundle["C"]},
        "splits": {},
    }

    for split in ("val", "test", "ood"):
        X, y, lp = load_split(split)
        if split == "val":
            pass  # reused below
        pw = p_wrong(X)
        base_acc = float((y == 0).mean())
        at = accuracy_at_threshold(y, pw, thr)
        answered = pw <= thr
        per_row_correct = (y[answered] == 0).astype(float)
        mean, lo, hi = bootstrap_ci(per_row_correct, config.BOOTSTRAP_N,
                                    config.BOOTSTRAP_SEED + abs(hash(split)) % 1000)
        # bootstrap CI on the DIFFERENCE acc_answered - base_acc via paired resample
        rng = np.random.default_rng(config.BOOTSTRAP_SEED)
        allcorrect = (y == 0).astype(float)
        boots = []
        idx_all = np.arange(len(y))
        for _ in range(config.BOOTSTRAP_N):
            ii = rng.choice(idx_all, size=len(y), replace=True)
            aa = answered[ii]
            if aa.sum() < 5:
                continue
            boots.append(allcorrect[ii][aa].mean() - allcorrect[ii].mean())
        gain_lo, gain_hi = (float(np.percentile(boots, 2.5)),
                            float(np.percentile(boots, 97.5))) if boots else (float("nan"),) * 2
        metrics["splits"][split] = {
            "n": int(len(y)),
            "base_accuracy": base_acc,
            "realised_coverage": at["realised_coverage"],
            "n_answered": at["n_answered"],
            "accuracy_answered_70cov": at["accuracy_answered"],
            "accuracy_answered_ci95": [mean, lo, hi],
            "gain_vs_base": at["accuracy_answered"] - base_acc,
            "gain_ci95": [gain_lo, gain_hi],
            "auroc_probe": auroc(y, pw),
            "auroc_mean_logprob": auroc(y, -lp[:, 0]),   # higher confidence -> lower score
            "auroc_first_maxprob": auroc(y, -lp[:, 1]),
        }
        print(f"[06] {split}: coverage target={config.TARGET_COVERAGE:.2f} "
              f"realised={at['realised_coverage']:.3f} | "
              f"acc@cov={at['accuracy_answered']:.3f} vs base={base_acc:.3f} "
              f"(gain {at['accuracy_answered']-base_acc:+.3f} "
              f"CI[{gain_lo:+.3f},{gain_hi:+.3f}]) | "
              f"AUROC probe={metrics['splits'][split]['auroc_probe']:.3f} "
              f"meanLP={metrics['splits'][split]['auroc_mean_logprob']:.3f} "
              f"firstMP={metrics['splits'][split]['auroc_first_maxprob']:.3f}")

    t, o = metrics["splits"]["test"], metrics["splits"]["ood"]
    headline_ok = (t["accuracy_answered_70cov"] > t["base_accuracy"]
                   and o["accuracy_answered_70cov"] > o["base_accuracy"])
    ood_degraded = o["accuracy_answered_70cov"] - o["base_accuracy"] < \
                   t["accuracy_answered_70cov"] - t["base_accuracy"]
    metrics["headline_beats_base_both"] = bool(headline_ok)
    metrics["ood_degradation_reported"] = bool(ood_degraded)
    if ood_degraded:
        print("[06] OOD DEGRADATION (reported, not hidden): OOD gain "
              f"{o['accuracy_answered_70cov']-o['base_accuracy']:+.3f} < test gain "
              f"{t['accuracy_answered_70cov']-t['base_accuracy']:+.3f}")

    # ---------------- plots
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import RocCurveDisplay

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    Xte, yte, lpte = load_split("test")
    RocCurveDisplay.from_predictions(yte, p_wrong(Xte), name="probe (last prompt token)", ax=ax)
    RocCurveDisplay.from_predictions(yte, -lpte[:, 0], name="mean log-prob", ax=ax)
    RocCurveDisplay.from_predictions(yte, -lpte[:, 1], name="first max-prob", ax=ax)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(config.RESULTS_DIR, "roc_test.png"), dpi=150)

    fig, ax = plt.subplots(figsize=(5, 4))
    nb = 10
    bins = np.quantile(p_wrong(Xva), np.linspace(0, 1, nb + 1))
    xs, ys = [], []
    for i in range(nb):
        m = (p_wrong(Xva) >= bins[i]) & (p_wrong(Xva) <= bins[i + 1])
        if m.sum() >= 5:
            xs.append(float(p_wrong(Xva)[m].mean()))
            ys.append(float(yva[m].mean()))
    ax.plot(xs, ys, "o-", label="val")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    ax.set_xlabel("predicted P(wrong)")
    ax.set_ylabel("observed wrong rate")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(config.RESULTS_DIR, "reliability_val.png"), dpi=150)

    with open(config.METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[06] wrote {config.METRICS_PATH} + roc_test.png + reliability_val.png")
    print("[gate P6] realised coverages printed next to target above: OK")


if __name__ == "__main__":
    main()
