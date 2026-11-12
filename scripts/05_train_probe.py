"""Stage 05 (gate P5): per-layer linear probes + logit baselines.

Trains one LogisticRegression per hidden-state row (layer 0 = embedding, rows
1..L = post-block activations) on the last-prompt-token activations of the train
split; selects C from {0.01, 0.1, 1.0, 10.0} by val AUROC; keeps the best layer.
(NOTE: the spec said "3 values of C"; we use a 4-point grid and say so.)

Baselines scored on the same val labels:
  mean_logprob   (a) mean log-prob of the chosen token, real tokens only
  first_maxprob  (b) max softmax prob of the first generated token
Both are CONFIDENCE scores (higher = more likely correct), so their AUROC against
y (1=wrong) is computed as 1 - roc_auc_score(y, confidence).

Writes models/probe.joblib, models/probe_meta.json (per-layer AUROC table).
Gate P5: print whether best-layer val AUROC beats BOTH baselines -- if it does
not, the script says so plainly in its output and meta file.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, ".")
import config  # noqa: E402


def load_split(split):
    X = np.load(config.acts_path(split))
    y = np.load(os.path.join(config.ACTS_DIR, f"{split}_y.npy")).astype(np.int8)
    lp = np.load(config.acts_logp_path(split))
    return X.astype(np.float32), y, lp


def fit_probe(Xtr, ytr, Xva, yva, Cs=config.C_GRID):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    best = None
    for C in Cs:
        t0 = time.time()
        clf = LogisticRegression(C=C, max_iter=2000, solver="lbfgs")
        clf.fit(Xtr, ytr)
        p = clf.predict_proba(Xva)[:, 1]
        auc = float(roc_auc_score(yva, p))
        if best is None or auc > best["val_auroc"]:
            best = {"C": C, "clf": clf, "val_auroc": auc, "fit_seconds": time.time() - t0}
    return best


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    config.RunFlags.add_args(ap)
    args = ap.parse_args(argv)
    fl = config.RunFlags.from_args(args)

    Xtr, ytr, _ = load_split("train")
    Xva, yva, lpva = load_split("val")
    n_states, d_model = Xtr.shape[1], Xtr.shape[2]
    env = config.describe_environment(fl.model_id)
    assert n_states == env["n_hidden_states_rows"] and d_model == env["d_model"]

    per_layer = []
    best_overall = None
    for l in range(n_states):
        b = fit_probe(Xtr[:, l, :], ytr, Xva[:, l, :], yva)
        per_layer.append({"layer": l, "C": b["C"], "val_auroc": b["val_auroc"],
                          "fit_seconds": round(b["fit_seconds"], 2)})
        print(f"[05] layer {l:2d}: val AUROC={b['val_auroc']:.4f} (C={b['C']})")
        if best_overall is None or b["val_auroc"] > best_overall["val_auroc"]:
            best_overall = {"layer": l, **b}

    from sklearn.metrics import roc_auc_score

    # baselines: confidence scores -> flip sign vs y (1=wrong)
    bl_mean = 1.0 - float(roc_auc_score(yva, lpva[:, 0]))
    bl_first = 1.0 - float(roc_auc_score(yva, lpva[:, 1]))
    best_auc = best_overall["val_auroc"]
    beats_both = best_auc > bl_mean and best_auc > bl_first

    import joblib

    os.makedirs(config.MODELS_DIR, exist_ok=True)
    joblib.dump({"probe": best_overall["clf"], "layer": best_overall["layer"],
                 "C": best_overall["C"]}, config.PROBE_PATH)
    meta = {
        "model_id": fl.model_id,
        "label_convention": config.LABEL_CONVENTION,
        "best_layer": best_overall["layer"],
        "best_C": best_overall["C"],
        "best_val_auroc": best_auc,
        "baseline_val_auroc": {"mean_logprob": bl_mean, "first_maxprob": bl_first},
        "probe_beats_both_baselines": bool(beats_both),
        "per_layer_val_auroc": per_layer,
        "c_grid_note": "spec said 3 values of C; this run uses the 4-point grid "
                       f"{config.C_GRID}",
        "n_train": int(len(ytr)), "n_val": int(len(yva)),
        "d_model": d_model, "n_hidden_states_rows": n_states,
    }
    with open(config.PROBE_META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    verdict = ("PROBE BEATS BOTH BASELINES" if beats_both else
               "PLAIN STATEMENT: the best-layer probe does NOT beat both logit "
               "baselines on val AUROC")
    print(f"[05] best layer {best_overall['layer']} AUROC={best_auc:.4f} | "
          f"baselines mean_logprob={bl_mean:.4f} first_maxprob={bl_first:.4f}")
    print(f"[gate P5] {verdict}")


if __name__ == "__main__":
    main()
