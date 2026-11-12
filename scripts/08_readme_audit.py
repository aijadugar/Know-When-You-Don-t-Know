"""Stage 08 (gate P9): README audit.

Rule: every number quoted in README.md must trace to a key in
results/metrics.json (or models/probe_meta.json).  This script regenerates the
numbers section of the README from the JSONs between marker comments, then
audits: any decimal number OUTSIDE the generated block that does not appear in
either JSON fails the gate.  Hand-typed or stale numbers cannot survive.
"""
from __future__ import annotations

import argparse
import json
import re
import sys

sys.path.insert(0, ".")
import config  # noqa: E402

BEGIN = "<!-- BEGIN GENERATED NUMBERS -->"
END = "<!-- END GENERATED NUMBERS -->"


def render_numbers(m: dict) -> str:
    s = m["splits"]
    rows = []
    for name, k in (("TriviaQA test pool", "test"), ("NQ-Open OOD", "ood"),
                    ("TriviaQA val (threshold calibration)", "val")):
        d = s[k]
        g = d["gain_ci95"]
        rows.append(
            f"| {name} | {d['n']} | {d['base_accuracy']:.3f} | "
            f"{d['realised_coverage']:.3f} | {d['accuracy_answered_70cov']:.3f} | "
            f"{d['gain_vs_base']:+.3f} [{g[0]:+.3f}, {g[1]:+.3f}] | "
            f"{d['auroc_probe']:.3f} | {d['auroc_mean_logprob']:.3f} | "
            f"{d['auroc_first_maxprob']:.3f} |")
    return (
        f"{BEGIN}\n"
        f"Target coverage **{m['target_coverage']:.2f}**, threshold on P(wrong) "
        f"**{m['threshold_p_wrong']:.4f}**, probe layer **{m['probe']['layer']}** "
        f"(C={m['probe']['C']}), model `{m['model_id']}`.\n\n"
        "| split | n | base acc | realised coverage | acc@cov | gain (95% CI) | probe AUROC | mean-logprob AUROC | first-maxprob AUROC |\n"
        "|---|---|---|---|---|---|---|---|---|\n"
        + "\n".join(rows)
        + f"\n\nHeadline criterion met on both held-out splits: "
          f"**{m['headline_beats_base_both']}**; OOD degradation flagged: "
          f"**{m['ood_degradation_reported']}**.\n"
          f"{END}")


def known_numbers(metrics: dict, meta: dict) -> set[str]:
    """All decimal strings derivable from the two JSONs (any formatting)."""
    nums: set[str] = set()

    def walk(o):
        if isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk(v)
        elif isinstance(o, bool):
            pass
        elif isinstance(o, (int, float)):
            nums.add(str(o))
            for nd in range(0, 6):
                nums.add(f"{o:.{nd}f}")
                nums.add(f"{o:+.{nd}f}")
        elif isinstance(o, str):
            for mt in re.findall(r"\d+\.\d+", o):
                nums.add(mt)
    walk(metrics)
    walk(meta)
    return nums


def audit(readme_text: str, metrics: dict, meta: dict) -> None:
    outside = re.sub(re.escape(BEGIN) + r".*?" + re.escape(END), "", readme_text, flags=re.S)
    allowed = known_numbers(metrics, meta)
    # integers and well-known structural numbers are exempt; decimals must trace
    offenders = []
    for num in re.findall(r"(?<![\w.])\d+\.\d+(?![\w.])", outside):
        if num in allowed:
            continue
        # allow version-like strings (torch/transformers pins etc.)
        offenders.append(num)
    if offenders:
        raise AssertionError(
            "gate P9 FAILED: README numbers with no metrics.json/probe_meta.json "
            f"source outside the generated block: {sorted(set(offenders))}")
    print("[gate P9] every decimal outside the generated block traces to a JSON key: OK")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    config.RunFlags.add_args(ap)
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args(argv)

    with open(config.METRICS_PATH) as f:
        metrics = json.load(f)
    with open(config.PROBE_META_PATH) as f:
        meta = json.load(f)
    block = render_numbers(metrics)

    with open("README.md") as f:
        text = f.read()
    if BEGIN in text and END in text:
        text = re.sub(re.escape(BEGIN) + r".*?" + re.escape(END), lambda _: block, text, flags=re.S)
    else:
        text = text.rstrip() + "\n\n## Results\n\n" + block + "\n"
    if not args.check_only:
        with open("README.md", "w") as f:
            f.write(text)
        print(f"[08] regenerated numbers block in README.md ({len(block)} chars)")
    audit(text, metrics, meta)


if __name__ == "__main__":
    main()
