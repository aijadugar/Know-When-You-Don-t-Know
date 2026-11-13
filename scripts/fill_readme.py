"""scripts/fill_readme.py — regenerate metrics tables in README from metrics.json."""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MODEL_ID
from common import load_jsonl

def get_metric(split: str, scorer: str, key: str, metrics: dict) -> str:
    """Extract a metric from metrics.json results."""
    for m in metrics.get("results", []):
        if m.get("split") == split and m.get("scorer") == scorer:
            val = m.get(key)
            if val is None:
                return "N/A"
            if isinstance(val, list) and len(val) == 2:
                return f"{val[0]:.3f}–{val[1]:.3f}"
            if isinstance(val, float):
                return f"{val:.3f}"
            return str(val)
    return "N/A"

def generate_metrics_table(metrics: dict) -> str:
    """Generate the metrics table from metrics.json."""
    verdict = metrics.get("verdict", {})
    ood_deg = verdict.get("ood_degradation", 0.0)

    def row(split: str, scorer: str, label: str) -> str:
        auroc = get_metric(split, scorer, "auroc", metrics)
        ci = get_metric(split, scorer, "auroc_ci", metrics)
        acc70 = get_metric(split, scorer, "acc_at_70", metrics)
        base = get_metric(split, scorer, "base_accuracy", metrics)
        real_cov = get_metric(split, scorer, "realised_coverage", metrics)
        wrong = get_metric(split, scorer, "share_of_wrong_caught", metrics)
        return f"| {label} | {auroc} | {acc70} | {base} | {real_cov} | {wrong} |"

    header = (
        "| Scorer | AUROC | Acc@70% | Base acc | Realised cov | Wrong caught |\n"
        "|--------|-------|---------|----------|--------------|--------------|"
    )

    test_rows = [
        row("test", "probe", "**Probe**"),
        row("test", "baseline_a", "−mean log-prob"),
        row("test", "baseline_b", "1 − max prob"),
    ]

    ood_rows = [
        row("ood", "probe", "**Probe**"),
        row("ood", "baseline_a", "−mean log-prob"),
        row("ood", "baseline_b", "1 − max prob"),
    ]

    beats_test = verdict.get("probe_beats_both_baselines_auroc_test", False)
    beats_ood = verdict.get("probe_beats_both_baselines_auroc_ood", False)

    ood_note = ""
    if ood_deg < -0.05:
        ood_note = (
            f"\n> ⚠️ **OOD degradation:** Probe AUROC on NQ-Open is {ood_deg:+.3f} "
            f"below TriviaQA test. This is reported rather than hidden."
        )

    sentence = (
        f"**Probe beats both baselines on TriviaQA test: {'yes' if beats_test else 'no'}; "
        f"on NQ-Open: {'yes' if beats_ood else 'no'}.**{ood_note}"
    )

    return (
        f"### TriviaQA Test\n{header}\n" + "\n".join(test_rows) + "\n"
        f"### NQ-Open (OOD)\n{header}\n" + "\n".join(ood_rows) + "\n"
        f"{sentence}"
    )

def fill_readme(readme_path: Path, metrics_path: Path) -> None:
    """Replace the metrics section between markers."""
    with open(readme_path, "r") as f:
        content = f.read()

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    table = generate_metrics_table(metrics)

    # Replace between markers
    pattern = r"(<!-- METRICS:START -->).*?(<!-- METRICS:END -->)"
    replacement = r"\1\n" + table + r"\n\2"
    new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)

    with open(readme_path, "w") as f:
        f.write(new_content)

    print(f"Updated {readme_path} with metrics from {metrics_path}")

def main():
    repo_root = Path(__file__).parent.parent
    readme_path = repo_root / "README.md"
    metrics_path = repo_root / "results" / "metrics.json"

    if not readme_path.exists():
        print(f"ERROR: {readme_path} not found")
        sys.exit(1)

    if not metrics_path.exists():
        print(f"ERROR: {metrics_path} not found — run 05_evaluate.py first")
        sys.exit(1)

    fill_readme(readme_path, metrics_path)

if __name__ == "__main__":
    main()
