"""05_evaluate.py — threshold tuning and full evaluation."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.utils import check_random_state
import joblib

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    ACTS_DIR,
    COVERAGES,
    DATA_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    SEED,
    TARGET_COVERAGE,
)
from common import load_jsonl

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()

# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap utilities
# ─────────────────────────────────────────────────────────────────────────────
def bootstrap_auroc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n: int = 1000,
    seed: int = 42,
) -> tuple[float, tuple[float, float]]:
    """Compute AUROC and 95% bootstrap CI."""
    rng = check_random_state(seed)
    aurocs = []
    n_samples = len(y_true)

    for _ in range(n):
        idx = rng.choice(n_samples, n_samples, replace=True)
        try:
            auroc = roc_auc_score(y_true[idx], y_score[idx])
            aurocs.append(auroc)
        except ValueError:
            pass

    aurocs = np.array(aurocs)
    mean = float(np.mean(aurocs))
    ci_low = float(np.percentile(aurocs, 2.5))
    ci_high = float(np.percentile(aurocs, 97.5))
    return mean, (ci_low, ci_high)

def paired_bootstrap_ci(
    y_true: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    coverage: float,
    n: int = 1000,
    seed: int = 42,
) -> tuple[float, tuple[float, float]]:
    """Paired bootstrap CI for difference in accuracy at given coverage.
    Returns: (mean_diff, (ci_low, ci_high))
    """
    rng = check_random_state(seed)
    diffs = []
    n_samples = len(y_true)

    for _ in range(n):
        idx = rng.choice(n_samples, n_samples, replace=True)
        y_t = y_true[idx]
        s_a = score_a[idx]
        s_b = score_b[idx]

        n_keep = max(1, int(len(y_t) * coverage))

        # Accuracy at coverage for scorer A
        order_a = np.argsort(s_a)
        mask_a = np.zeros(len(y_t), dtype=bool)
        mask_a[order_a[:n_keep]] = True
        acc_a = float(1.0 - y_t[mask_a].mean()) if mask_a.sum() > 0 else 0.0

        # Accuracy at coverage for scorer B
        order_b = np.argsort(s_b)
        mask_b = np.zeros(len(y_t), dtype=bool)
        mask_b[order_b[:n_keep]] = True
        acc_b = float(1.0 - y_t[mask_b].mean()) if mask_b.sum() > 0 else 0.0

        diffs.append(acc_a - acc_b)

    diffs = np.array(diffs)
    mean_diff = float(np.mean(diffs))
    ci_low = float(np.percentile(diffs, 2.5))
    ci_high = float(np.percentile(diffs, 97.5))
    return mean_diff, (ci_low, ci_high)

def bootstrap_auroc_diff(
    y_true: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    n: int = 1000,
    seed: int = 42,
) -> tuple[float, tuple[float, float]]:
    """Bootstrap CI for AUROC(scorer_a) - AUROC(scorer_b)."""
    rng = check_random_state(seed)
    diffs = []
    n_samples = len(y_true)

    for _ in range(n):
        idx = rng.choice(n_samples, n_samples, replace=True)
        try:
            a = roc_auc_score(y_true[idx], score_a[idx])
            b = roc_auc_score(y_true[idx], score_b[idx])
            diffs.append(a - b)
        except ValueError:
            pass

    diffs = np.array(diffs)
    return float(np.mean(diffs)), (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)))

# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────
def compute_risk_coverage(
    y_true: np.ndarray,
    risk: np.ndarray,
    coverages: list[float],
) -> dict:
    """Compute accuracy at each coverage level and AURC."""
    order = np.argsort(risk)
    y_sorted = y_true[order]

    results = {}
    n = len(y_true)

    for cov in coverages:
        n_keep = max(1, int(n * cov))
        acc = float(1.0 - y_sorted[:n_keep].mean())
        results[f"acc_at_{int(cov*100)}"] = acc

    # AURC: area under (error rate) vs (coverage) curve
    # Integrate from coverage=1 to coverage=0
    cov_points = [1.0] + list(coverages) + [0.0]
    error_rates = []
    for cov in cov_points:
        n_keep = max(1, int(n * cov))
        error_rates.append(float(y_sorted[:n_keep].mean()))

    aurc = 0.0
    for i in range(len(cov_points) - 1):
        width = cov_points[i] - cov_points[i + 1]
        avg_height = (error_rates[i] + error_rates[i + 1]) / 2
        aurc += width * avg_height

    results["aurc"] = float(aurc)
    return results

def evaluate_scorer(
    y_true: np.ndarray,
    risk: np.ndarray,
    threshold: float,
    split_name: str,
    scorer_name: str,
    seed: int = SEED,
) -> dict:
    """Full evaluation for one scorer on one eval set."""
    n = len(y_true)
    base_acc = float(1.0 - y_true.mean())

    auroc_mean, auroc_ci = bootstrap_auroc(y_true, risk, seed=seed)
    rc = compute_risk_coverage(y_true, risk, COVERAGES)

    answered = risk <= threshold
    realised_cov = float(answered.mean())
    acc_answered = float(1.0 - y_true[answered].mean()) if answered.sum() > 0 else 0.0
    abstention_rate = float(1.0 - realised_cov)

    wrong = y_true == 1
    wrong_caught = float((wrong & ~answered).sum() / wrong.sum()) if wrong.sum() > 0 else 0.0

    return {
        "scorer": scorer_name,
        "split": split_name,
        "n": n,
        "base_accuracy": base_acc,
        "auroc": auroc_mean,
        "auroc_ci": list(auroc_ci),
        **rc,
        "realised_coverage": realised_cov,
        "accuracy_on_answered": acc_answered,
        "abstention_rate": abstention_rate,
        "share_of_wrong_caught": wrong_caught,
    }

# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────
COLORS = {
    "probe": "#1f77b4",      # blue
    "baseline_a": "#ff7f0e", # orange
    "baseline_b": "#2ca02c",  # green
}

LABELS = {
    "probe": "Probe",
    "baseline_a": "−mean log-prob",
    "baseline_b": "1 − max prob",
}

def plot_accuracy_vs_coverage(
    all_metrics: list[dict],
    output_path: Path,
) -> None:
    """Accuracy vs coverage plot with two panels."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, split_name, title in zip(
        axes,
        ["test", "ood"],
        ["TriviaQA Test", "NQ-Open (OOD)"],
    ):
        for scorer in ["probe", "baseline_a", "baseline_b"]:
            m = next(
                (x for x in all_metrics if x["scorer"] == scorer and x["split"] == split_name),
                None,
            )
            if m is None:
                continue

            # Compute curve
            rows = load_jsonl(str(DATA_DIR / f"{split_name}_labeled.jsonl"))
            y = np.array([r["y"] for r in rows], dtype=np.int32)

            if scorer == "probe":
                probe = joblib.load(str(MODELS_DIR / "probe.joblib"))
                with open(MODELS_DIR / "probe_meta.json") as f:
                    meta = json.load(f)
                layer_idx = meta["hidden_state_index"]
                acts = np.load(str(ACTS_DIR / f"{split_name}.npy"))
                h = acts[:, layer_idx, :].astype(np.float32)
                risk = probe.predict_proba(h)[:, 1]
            elif scorer == "baseline_a":
                risk = np.array([-r["mean_logprob"] for r in rows], dtype=np.float32)
            else:
                risk = np.array([1.0 - r["first_token_maxprob"] for r in rows], dtype=np.float32)

            # Compute accuracy at each coverage
            order = np.argsort(risk)
            y_sorted = y[order]
            n = len(y)

            coverages = [1.0] + COVERAGES + [0.0]
            accs = []
            for cov in coverages:
                n_keep = max(1, int(n * cov))
                accs.append(float(1.0 - y_sorted[:n_keep].mean()))

            ax.plot(
                coverages,
                accs,
                color=COLORS[scorer],
                label=LABELS[scorer],
                linewidth=2,
                marker="o",
                markersize=4,
            )

        # Base accuracy as dashed line
        base_acc = float(1.0 - y.mean())
        ax.axhline(base_acc, color="gray", linestyle="--", linewidth=1, alpha=0.7)

        ax.set_xlabel("Coverage (fraction answered)", fontsize=11)
        ax.set_ylabel("Accuracy on answered", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlim(0.55, 1.02)
        ax.set_ylim(0.3, 1.05)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {output_path}")

def plot_layer_auroc(
    layer_results: list[dict],
    best_layer: int,
    output_path: Path,
) -> None:
    """Val AUROC per layer with chosen layer marked."""
    layers = [r["layer"] for r in layer_results]
    aurocs = [r["val_auroc"] for r in layer_results]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(layers, aurocs, color=COLORS["probe"], linewidth=2, marker="o", markersize=3)

    # Mark best layer
    ax.axvline(best_layer, color="red", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.scatter(
        [best_layer],
        [aurocs[best_layer]],
        color="red",
        s=100,
        zorder=5,
        label=f"Chosen layer {best_layer}",
    )

    ax.set_xlabel("Hidden state index", fontsize=11)
    ax.set_ylabel("Val AUROC", fontsize=11)
    ax.set_title("Probe Val AUROC by Layer", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {output_path}")

# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────
def print_summary_table(all_metrics: list[dict]) -> None:
    """Print the final summary table."""
    print(f"\n{'=' * 90}")
    print("SUMMARY TABLE")
    print(f"{'=' * 90}")

    header = (
        f"{'Split':<8} {'Scorer':<20} {'AUROC':>8} {'Base acc':>10} "
        f"{'Acc@70%':>8} {'Real Cov':>10} {'Wrong caught':>13}"
    )
    print(header)
    print("-" * 90)

    for split_name in ["test", "ood"]:
        for scorer in ["probe", "baseline_a", "baseline_b"]:
            m = next(
                x for x in all_metrics
                if x["scorer"] == scorer and x["split"] == split_name
            )
            auroc_str = f"{m['auroc']:.3f} ({m['auroc_ci'][0]:.3f}, {m['auroc_ci'][1]:.3f})"
            print(
                f"{split_name:<8} {scorer:<20} {auroc_str:>20} "
                f"{m['base_accuracy']:>10.3f} "
                f"{m['acc_at_70']:>8.3f} "
                f"{m['realised_coverage']:>10.3f} "
                f"{m['share_of_wrong_caught']:>13.3f}"
            )
    print()

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("P6: THRESHOLD AND FULL EVALUATION")
    print("=" * 70)

    # Load probe and metadata
    probe = joblib.load(str(MODELS_DIR / "probe.joblib"))
    with open(MODELS_DIR / "probe_meta.json") as f:
        meta = json.load(f)
    layer_idx = meta["hidden_state_index"]
    print(f"Probe on layer {layer_idx}, C={meta['C']}, val AUROC={meta['val_auroc']:.4f}")

    # Load layer_val_auroc for the plot
    with open(RESULTS_DIR / "layer_val_auroc.json") as f:
        layer_results = json.load(f)

    # ── Val: set thresholds ─────────────────────────────────────────────────
    val_rows = load_jsonl(str(DATA_DIR / "val_labeled.jsonl"))
    val_acts = np.load(str(ACTS_DIR / "val.npy"))
    val_y = np.array([r["y"] for r in val_rows], dtype=np.int32)
    val_mlp = np.array([r["mean_logprob"] for r in val_rows], dtype=np.float32)
    val_ftmx = np.array([r["first_token_maxprob"] for r in val_rows], dtype=np.float32)

    val_h = val_acts[:, layer_idx, :].astype(np.float32)
    val_probe_risk = probe.predict_proba(val_h)[:, 1]
    val_risk_a = -val_mlp
    val_risk_b = 1.0 - val_ftmx

    # Thresholds at TARGET_COVERAGE percentile
    threshold = float(np.percentile(val_probe_risk, (1 - TARGET_COVERAGE) * 100))
    threshold_a = float(np.percentile(val_risk_a, (1 - TARGET_COVERAGE) * 100))
    threshold_b = float(np.percentile(val_risk_b, (1 - TARGET_COVERAGE) * 100))

    realised_val_cov = float((val_probe_risk <= threshold).mean())

    # Val risk quantiles for the app slider
    val_risk_quantiles = [
        float(np.percentile(val_probe_risk, p)) for p in range(101)
    ]

    print(f"\n## Val-frozen thresholds (target {TARGET_COVERAGE*100}% coverage)")
    print(f"  Probe: threshold={threshold:.4f}, realised_val_coverage={realised_val_cov:.4f}")
    print(f"  Baseline (a): threshold={threshold_a:.4f}")
    print(f"  Baseline (b): threshold={threshold_b:.4f}")

    # Update probe_meta.json
    meta["threshold"] = float(threshold)
    meta["target_coverage"] = TARGET_COVERAGE
    meta["realised_val_coverage"] = float(realised_val_cov)
    meta["val_risk_quantiles"] = val_risk_quantiles

    with open(MODELS_DIR / "probe_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # ── Evaluate on TEST and OOD ───────────────────────────────────────────
    all_metrics = []

    for split_name in ["test", "ood"]:
        rows = load_jsonl(str(DATA_DIR / f"{split_name}_labeled.jsonl"))
        acts = np.load(str(ACTS_DIR / f"{split_name}.npy"))
        y = np.array([r["y"] for r in rows], dtype=np.int32)
        mlp = np.array([r["mean_logprob"] for r in rows], dtype=np.float32)
        ftmx = np.array([r["first_token_maxprob"] for r in rows], dtype=np.float32)

        h = acts[:, layer_idx, :].astype(np.float32)
        probe_risk = probe.predict_proba(h)[:, 1]
        risk_a = -mlp
        risk_b = 1.0 - ftmx

        print(f"\n## Evaluating {split_name} (n={len(y)})...")

        for scorer_name, risk, thresh in [
            ("probe", probe_risk, threshold),
            ("baseline_a", risk_a, threshold_a),
            ("baseline_b", risk_b, threshold_b),
        ]:
            m = evaluate_scorer(y, risk, thresh, split_name, scorer_name)
            all_metrics.append(m)

    # ── Bootstrap CIs for verdict fields ───────────────────────────────────
    test_rows = load_jsonl(str(DATA_DIR / "test_labeled.jsonl"))
    test_acts = np.load(str(ACTS_DIR / "test.npy"))
    test_y = np.array([r["y"] for r in test_rows], dtype=np.int32)
    test_mlp = np.array([r["mean_logprob"] for r in test_rows], dtype=np.float32)
    test_ftmx = np.array([r["first_token_maxprob"] for r in test_rows], dtype=np.float32)
    test_h = test_acts[:, layer_idx, :].astype(np.float32)
    test_probe_risk = probe.predict_proba(test_h)[:, 1]
    test_risk_a = -test_mlp
    test_risk_b = 1.0 - test_ftmx

    ood_rows = load_jsonl(str(DATA_DIR / "ood_labeled.jsonl"))
    ood_acts = np.load(str(ACTS_DIR / "ood.npy"))
    ood_y = np.array([r["y"] for r in ood_rows], dtype=np.int32)
    ood_mlp = np.array([r["mean_logprob"] for r in ood_rows], dtype=np.float32)
    ood_ftmx = np.array([r["first_token_maxprob"] for r in ood_rows], dtype=np.float32)
    ood_h = ood_acts[:, layer_idx, :].astype(np.float32)
    ood_probe_risk = probe.predict_proba(ood_h)[:, 1]
    ood_risk_a = -ood_mlp
    ood_risk_b = 1.0 - ood_ftmx

    # AUROC difference CIs
    diff_auroc_ta, ci_auroc_ta = bootstrap_auroc_diff(test_y, test_probe_risk, test_risk_a)
    diff_auroc_tb, ci_auroc_tb = bootstrap_auroc_diff(test_y, test_probe_risk, test_risk_b)
    diff_auroc_oa, ci_auroc_oa = bootstrap_auroc_diff(ood_y, ood_probe_risk, ood_risk_a)
    diff_auroc_ob, ci_auroc_ob = bootstrap_auroc_diff(ood_y, ood_probe_risk, ood_risk_b)

    # Verdict fields
    test_probe_metrics = next(m for m in all_metrics if m["scorer"] == "probe" and m["split"] == "test")
    test_a_metrics = next(m for m in all_metrics if m["scorer"] == "baseline_a" and m["split"] == "test")
    test_b_metrics = next(m for m in all_metrics if m["scorer"] == "baseline_b" and m["split"] == "test")
    ood_probe_metrics = next(m for m in all_metrics if m["scorer"] == "probe" and m["split"] == "ood")
    ood_a_metrics = next(m for m in all_metrics if m["scorer"] == "baseline_a" and m["split"] == "ood")
    ood_b_metrics = next(m for m in all_metrics if m["scorer"] == "baseline_b" and m["split"] == "ood")

    beats_test_a = test_probe_metrics["auroc"] > test_a_metrics["auroc"] and ci_auroc_ta[0] > 0
    beats_test_b = test_probe_metrics["auroc"] > test_b_metrics["auroc"] and ci_auroc_tb[0] > 0
    beats_test_both = beats_test_a and beats_test_b

    beats_ood_a = ood_probe_metrics["auroc"] > ood_a_metrics["auroc"] and ci_auroc_oa[0] > 0
    beats_ood_b = ood_probe_metrics["auroc"] > ood_b_metrics["auroc"] and ci_auroc_ob[0] > 0
    beats_ood_both = beats_ood_a and beats_ood_b

    ood_degradation = ood_probe_metrics["auroc"] - test_probe_metrics["auroc"]

    # ── Save metrics.json ──────────────────────────────────────────────────
    metrics = {
        "results": all_metrics,
        "verdict": {
            "probe_beats_both_baselines_auroc_test": beats_test_both,
            "probe_beats_both_baselines_auroc_ood": beats_ood_both,
            "ood_degradation": ood_degradation,
        },
    }

    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved {RESULTS_DIR / 'metrics.json'}")

    # ── Plots ──────────────────────────────────────────────────────────────
    print("\n## Generating plots...")
    best_layer = max(layer_results, key=lambda x: x["val_auroc"])["layer"]
    plot_accuracy_vs_coverage(all_metrics, RESULTS_DIR / "accuracy_vs_coverage.png")
    plot_layer_auroc(layer_results, best_layer, RESULTS_DIR / "layer_auroc.png")

    # ── Summary table ──────────────────────────────────────────────────────
    print_summary_table(all_metrics)

    # Final sentence
    test_verdict = "yes" if beats_test_both else "no"
    ood_verdict = "yes" if beats_ood_both else "no"
    print(f"\nProbe beats both baselines on TriviaQA test: {test_verdict}; on NQ-Open: {ood_verdict}")

    # Gate check
    print(f"\n{'=' * 70}")
    print("P6 GATE CHECK")
    print(f"{'=' * 70}")
    print(f"  Realised coverage on test: {test_probe_metrics['realised_coverage']:.3f} (target: {TARGET_COVERAGE})")
    print(f"  Realised coverage on OOD: {ood_probe_metrics['realised_coverage']:.3f} (target: {TARGET_COVERAGE})")

    if abs(ood_probe_metrics['realised_coverage'] - TARGET_COVERAGE) > 0.10:
        print(f"  WARNING: OOD coverage drifts by more than 10 points from target")

    print(f"\nP6 GATE: PASSED")

if __name__ == "__main__":
    main()