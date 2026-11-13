"""scripts/audit_claims.py — verify every number in README appears in config/requirements/metrics."""
from __future__ import annotations
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    MODEL_ID, SMOKE_MODEL_ID, SYSTEM_PROMPT, MAX_NEW_TOKENS,
    N_TRAIN, N_VAL, N_TEST, N_OOD, C_VALUES, TARGET_COVERAGE,
    COVERAGES, SEED,
)

def extract_numbers(text: str) -> list[tuple[str, str]]:
    """Extract all numbers from text as (raw_text, context) pairs."""
    numbers = []
    # Match numbers (integers and floats) that are not inside code blocks
    for match in re.finditer(r'\b\d+(?:\.\d+)?\b', text):
        start = max(0, match.start() - 40)
        end = min(len(text), match.end() + 40)
        context = text[start:end].replace('\n', ' ')
        numbers.append((match.group(), context))
    return numbers

def load_allowed_numbers() -> set[str]:
    """Build set of allowed numbers from config.py and requirements.txt."""
    allowed = set()

    # From config.py
    allowed.add(str(N_TRAIN))
    allowed.add(str(N_VAL))
    allowed.add(str(N_TEST))
    allowed.add(str(N_OOD))
    allowed.add(str(TARGET_COVERAGE))
    allowed.add(str(MAX_NEW_TOKENS))
    allowed.add(str(SEED))

    for c in C_VALUES:
        allowed.add(str(c))

    for cov in COVERAGES:
        allowed.add(str(int(cov * 100)))  # e.g. 90 from 0.9

    allowed.add(str(len(C_VALUES)))
    allowed.add(str(len(COVERAGES)))

    # From requirements.txt
    req_path = Path(__file__).parent.parent / "requirements.txt"
    if req_path.exists():
        with open(req_path) as f:
            for line in f:
                # Extract version numbers like gradio==4.44.0
                for m in re.finditer(r'==(\d+(?:\.\d+)*)', line):
                    allowed.add(m.group(1))

    return allowed

def load_metrics_numbers(metrics_path: Path) -> set[str]:
    """Extract all numbers from metrics.json."""
    import json
    allowed = set()

    if not metrics_path.exists():
        return allowed

    with open(metrics_path) as f:
        metrics = json.load(f)

    def flatten(obj):
        if isinstance(obj, (int, float)):
            # Round floats to 3 decimal places for comparison
            if isinstance(obj, float):
                return {f"{obj:.3f}", str(obj)}
            return {str(obj)}
        elif isinstance(obj, dict):
            result = set()
            for v in obj.values():
                result.update(flatten(v))
            return result
        elif isinstance(obj, list):
            result = set()
            for item in obj:
                result.update(flatten(item))
            return result
        return set()

    return flatten(metrics)

def audit_readme(readme_path: Path) -> bool:
    """Check every number in README.md against allowed sources."""
    with open(readme_path) as f:
        content = f.read()

    # Remove code blocks
    content_no_code = re.sub(r'```[\s\S]*?```', '', content)
    content_no_code = re.sub(r'`[^`]+`', '', content_no_code)

    # Remove metrics markers section
    content_no_code = re.sub(
        r'<!-- METRICS:START -->.*?<!-- METRICS:END -->',
        '',
        content_no_code,
        flags=re.DOTALL,
    )

    # Remove YAML front matter
    content_no_code = re.sub(r'^---\n[\s\S]*?---\n', '', content_no_code)

    numbers = extract_numbers(content_no_code)

    allowed_config = load_allowed_numbers()
    metrics_path = readme_path.parent / "results" / "metrics.json"
    allowed_metrics = load_metrics_numbers(metrics_path)

    failures = []
    for num, context in numbers:
        if num not in allowed_config and num not in allowed_metrics:
            failures.append((num, context))

    if failures:
        print(f"AUDIT FAILED: {len(failures)} numbers not in allowed sources:")
        for num, context in failures:
            print(f"  '{num}' in: ...{context}...")
        return False

    print(f"AUDIT PASSED: all {len(numbers)} numbers traced to config/requirements/metrics.json")
    return True

def main():
    repo_root = Path(__file__).parent.parent
    readme_path = repo_root / "README.md"

    if not readme_path.exists():
        print(f"ERROR: {readme_path} not found")
        sys.exit(1)

    success = audit_readme(readme_path)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
