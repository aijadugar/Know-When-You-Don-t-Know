"""04_train_probe.py — train linear probe and compare to baselines."""
from __future__ import annotations
import argparse
import json
import sys
import warnings
from pathlib import Path
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    ACTS_DIR,
    C_VALUES,
    DATA_DIR,
    MODEL_ID,
    SEED,
    SMOKE_MODEL_ID,
    SYSTEM_PROMPT,
    MODELS_DIR,
    RESULTS_DIR,
)
from common import load_jsonl, set_seed

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--model_id", type=str, default=None)
    return parser.parse_args()

def load_split(split_name: str):
    """Load activations and labels for one split."""
    acts = np.load(str(ACTS_DIR / f"{split_name}.npy"))
    rows = load_jsonl(str(DATA_DIR / f"{split_name}_labeled.jsonl"))
    y = np.array([r["y"] for r in rows], dtype=np.int32)
    mean_logprob = np.array([r["mean_logprob"] for r in rows], dtype=np.float32)
    first_token_maxprob = np.array(
        [r["first_token_maxprob"] for r in rows], dtype=np.float32
    )
    questions = [r["question"] for r in rows]
    return acts, y, mean_logprob, first_token_maxprob, questions

def main():
    args = parse_args()
    set_seed(SEED)

    print("=" * 70)
    print("P5: PROBE AND BASELINES")
    print("=" * 70)

    model_id = SMOKE_MODEL_ID if args.smoke else (args.model_id or MODEL_ID)
    print(f"Model: {model_id}")

    # Load train and val
    print("\n## Loading data...")
    train_acts, train_y, train_mlp, train_ftmx, train_qs = load_split("train")
    val_acts, val_y, val_mlp, val_ftmx, val_qs = load_split("val")

    print(f"  train: {train_acts.shape}, y={train_y.mean():.3f} wrong")
    print(f"  val:   {val_acts.shape}, y={val_y.mean():.3f} wrong")

    n_layers_plus_1 = train_acts.shape[1]
    d_model = train_acts.shape[2]

    # ── Baseline (a): risk = -mean_logprob ─────────────────────────────────
    risk_a = -val_mlp
    auroc_baseline_a = roc_auc_score(val_y, risk_a)
    print(f"\nBaseline (a) -mean_logprob AUROC: {auroc_baseline_a:.4f}")

    # ── Baseline (b): risk = 1 - first_token_maxprob ───────────────────────
    risk_b = 1.0 - val_ftmx
    auroc_baseline_b = roc_auc_score(val_y, risk_b)
    print(f"  Baseline (b) 1-first_token_maxprob AUROC: {auroc_baseline_b:.4f}")

    # ── Baseline (c): TF-IDF of question text ───────────────────────────────
    print("\n## TF-IDF baseline...")
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    train_tfidf = tfidf.fit_transform(train_qs)
    val_tfidf = tfidf.transform(val_qs)

    tfidf_probe = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        tfidf_probe.fit(train_tfidf, train_y)

    risk_tfidf = tfidf_probe.predict_proba(val_tfidf)[:, 1]
    auroc_tfidf = roc_auc_score(val_y, risk_tfidf)
    print(f"  TF-IDF probe AUROC: {auroc_tfidf:.4f}")

    # ── Hidden-state probe: search over layers and C ───────────────────────
    print(f"\n## Training probes over {n_layers_plus_1} layers × {len(C_VALUES)} C values...")

    # Track convergence warnings
    convergence_warnings = []
    layer_results = []

    for layer_idx in tqdm(range(n_layers_plus_1), desc="  Layers"):
        # Convert one layer at a time to float32 (save RAM)
        train_h = train_acts[:, layer_idx, :].astype(np.float32)
        val_h = val_acts[:, layer_idx, :].astype(np.float32)

        best_c = None
        best_auroc = -1.0

        for C in C_VALUES:
            probe = Pipeline([
                ("scaler", StandardScaler()),
                ("logreg", LogisticRegression(C=C, max_iter=2000, random_state=SEED)),
            ])

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always", ConvergenceWarning)
                probe.fit(train_h, train_y)

                if w:
                    for warning in w:
                        if issubclass(warning.category, ConvergenceWarning):
                            convergence_warnings.append({
                                "layer": layer_idx,
                                "C": C,
                                "message": str(warning.message),
                            })

            risk = probe.predict_proba(val_h)[:, 1]
            auroc = roc_auc_score(val_y, risk)

            if auroc > best_auroc:
                best_auroc = auroc
                best_c = C

        layer_results.append({
            "layer": layer_idx,
            "best_C": best_c,
            "val_auroc": best_auroc,
        })

    # Find best layer
    best_layer = max(layer_results, key=lambda x: x["val_auroc"])
    best_layer_idx = best_layer["layer"]
    best_layer_C = best_layer["best_C"]
    best_layer_auroc = best_layer["val_auroc"]

    print(f"\nBest layer: {best_layer_idx}, C={best_layer_C}, AUROC={best_layer_auroc:.4f}")

    # Convergence warnings summary
    if convergence_warnings:
        print(f"\nSklearn convergence warnings ({len(convergence_warnings)}):")
        for w in convergence_warnings[:5]:
            print(f"    layer={w['layer']}, C={w['C']}: {w['message'][:60]}")
        if len(convergence_warnings) > 5:
            print(f"    ... and {len(convergence_warnings) - 5} more")
    else:
        print("\nNo convergence warnings")

    # ── Print AUROC table ───────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("VAL AUROC TABLE")
    print(f"{'=' * 70}")
    print(f"{'Method':<30} {'Val AUROC':>12}")
    print("-" * 44)
    print(f"{'Probe (best layer)':<30} {best_layer_auroc:>12.4f}")
    print(f"{'Baseline (a) -mean_logprob':<30} {auroc_baseline_a:>12.4f}")
    print(f"{'Baseline (b) 1-first_token_maxprob':<30} {auroc_baseline_b:>12.4f}")
    print(f"{'Baseline (c) TF-IDF':<30} {auroc_tfidf:>12.4f}")

    # ── Gate: probe must beat both baselines ───────────────────────────────
    beats_a = best_layer_auroc > auroc_baseline_a
    beats_b = best_layer_auroc > auroc_baseline_b
    beats_both = beats_a and beats_b

    print(f"\n{'=' * 70}")
    print("P5 GATE CHECK")
    print(f"{'=' * 70}")
    print(f"  Probe AUROC {best_layer_auroc:.4f} vs baseline (a) {auroc_baseline_a:.4f}: "
          f"{'✓ beats' if beats_a else '✗ does not beat'}")
    print(f"  Probe AUROC {best_layer_auroc:.4f} vs baseline (b) {auroc_baseline_b:.4f}: "
          f"{'✓ beats' if beats_b else '✗ does not beat'}")

    if not beats_both:
        print(f"\n*** Probe does NOT beat both baselines. Stopping. ***")
        print(f"  probe={best_layer_auroc:.4f}, baseline_a={auroc_baseline_a:.4f}, "
              f"baseline_b={auroc_baseline_b:.4f}")
        sys.exit(1)

    # ── Train final probe on best layer + C ────────────────────────────────
    print(f"\n## Training final probe on layer {best_layer_idx}, C={best_layer_C}...")

    train_h = train_acts[:, best_layer_idx, :].astype(np.float32)
    val_h = val_acts[:, best_layer_idx, :].astype(np.float32)

    final_probe = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(C=best_layer_C, max_iter=2000, random_state=SEED)),
    ])
    final_probe.fit(train_h, train_y)

    # ── Save outputs ───────────────────────────────────────────────────────
    import joblib
    joblib.dump(final_probe, str(MODELS_DIR / "probe.joblib"))
    print(f"  Saved models/probe.joblib")

    # probe_meta.json (threshold filled by P6)
    probe_meta = {
        "model_id": model_id,
        "hidden_state_index": best_layer_idx,
        "n_layers": n_layers_plus_1,
        "d_model": d_model,
        "C": best_layer_C,
        "dtype_used": "float32",  # We always train in float32
        "system_prompt": SYSTEM_PROMPT,
        "val_auroc": float(best_layer_auroc),
        "baseline_a_val_auroc": float(auroc_baseline_a),
        "baseline_b_val_auroc": float(auroc_baseline_b),
        "tfidf_val_auroc": float(auroc_tfidf),
        "seed": SEED,
        "threshold": None,
        "target_coverage": None,
        "realised_val_coverage": None,
        "val_risk_quantiles": None,
    }

    with open(MODELS_DIR / "probe_meta.json", "w") as f:
        json.dump(probe_meta, f, indent=2)
    print(f"  Saved models/probe_meta.json")

    # layer_val_auroc.json
    with open(RESULTS_DIR / "layer_val_auroc.json", "w") as f:
        json.dump(layer_results, f, indent=2)
    print(f"  Saved results/layer_val_auroc.json")

    print(f"\n{'=' * 70}")
    print(f"P5 GATE: PASSED — probe AUROC {best_layer_auroc:.4f} beats both baselines "
          f"({auroc_baseline_a:.4f}, {auroc_baseline_b:.4f}) on val")
    print(f"{'=' * 70}")

if __name__ == "__main__":
    main()
