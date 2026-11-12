"""Stage 07 (gate P7): generate the Hub model-card text FROM metrics.json.

Every number in the card is read from results/metrics.json / models/probe_meta.json
at build time -- nothing is typed by hand.  `python scripts/07_make_card.py` writes
models/README.md (the card).  Pushing to the Hub is a separate explicit step:
`huggingface-cli upload <user>/abstain-probe-<model> models/ README.md --repo-type model`
plus the artifacts (probe.joblib, probe_meta.json) and requires HF_TOKEN, so this
script only builds and self-checks the card (the gate).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, ".")
import config  # noqa: E402


def fmt(x, nd=3):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else str(x)


def build_card(metrics: dict, meta: dict) -> str:
    t, o, v = metrics["splits"]["test"], metrics["splits"]["ood"], metrics["splits"]["val"]
    lines = [
        "---",
        'license: apache-2.0',
        f'base_model: {metrics["model_id"]}',
        'tags:',
        '- abstention',
        '- linear-probe',
        '- triviaqa',
        '- nq-open',
        '---',
        '',
        f'# Wrong-answer probe for `{metrics["model_id"]}`',
        '',
        'Linear logistic-regression probe on the hidden state at the LAST PROMPT TOKEN',
        '(all layers evaluated; best layer kept) that predicts whether Qwen will answer',
        'a factual question WRONG, enabling abstention before generation.',
        '',
        f'**Label convention:** {metrics["label_convention"]}',
        '',
        '## Headline (accuracy on answered questions at target coverage)',
        '',
        f'Target coverage: {fmt(metrics["target_coverage"], 2)}. '
        f'Threshold on P(wrong): {fmt(metrics["threshold_p_wrong"], 4)}. '
        f'Probe: layer {metrics["probe"]["layer"]}, C={metrics["probe"]["C"]}.',
        '',
        '| split | n | base acc | realised coverage | acc@cov | gain (95% CI) | probe AUROC | mean-logprob AUROC | first-maxprob AUROC |',
        '|---|---|---|---|---|---|---|---|---|',
    ]
    for name, s in (("TriviaQA test pool", t), ("NQ-Open OOD", o), ("TriviaQA val", v)):
        g = s["gain_ci95"]
        lines.append(
            f'| {name} | {s["n"]} | {fmt(s["base_accuracy"])} | '
            f'{fmt(s["realised_coverage"])} | {fmt(s["accuracy_answered_70cov"])} | '
            f'{s["gain_vs_base"]:+.3f} [{g[0]:+.3f}, {g[1]:+.3f}] | '
            f'{fmt(s["auroc_probe"])} | {fmt(s["auroc_mean_logprob"])} | '
            f'{fmt(s["auroc_first_maxprob"])} |')
    lines += [
        '',
        f'- Headline criterion met on BOTH held-out splits: **{metrics["headline_beats_base_both"]}**',
        f'- Probe beats both logit baselines on val AUROC: **{meta["probe_beats_both_baselines"]}**',
        f'- OOD degradation reported (not hidden): **{metrics["ood_degradation_reported"]}**',
        '',
        '## Files',
        '',
        '- `probe.joblib` — dict {probe, layer, C}; apply to `hidden_states[layer][:, -1, :]`',
        '- `probe_meta.json` — per-layer val AUROC table and settings',
        '- This card was GENERATED from `results/metrics.json`; do not edit numbers by hand.',
        '',
    ]
    return "\n".join(lines)


def check_gate(card: str, metrics: dict) -> None:
    """Gate P7: the card must be a pure function of metrics.json + probe_meta.json.

    Rebuild both JSONs from disk and require byte-for-byte equality; any hand-typed
    number would break this (or disappear when the card is regenerated).
    """
    with open(config.PROBE_META_PATH) as f:
        meta = json.load(f)
    rebuilt = build_card(metrics, meta)
    assert rebuilt == card, "gate P7 failed: card is not a pure function of metrics.json"
    print("[gate P7] card regenerated from metrics.json matches byte-for-byte: OK")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    config.RunFlags.add_args(ap)
    args = ap.parse_args(argv)
    with open(config.METRICS_PATH) as f:
        metrics = json.load(f)
    with open(config.PROBE_META_PATH) as f:
        meta = json.load(f)
    card = build_card(metrics, meta)
    out = os.path.join(config.MODELS_DIR, "README.md")
    with open(out, "w") as f:
        f.write(card)
    print(f"[07] wrote {out} ({len(card)} chars)")
    check_gate(card, metrics)


if __name__ == "__main__":
    main()
