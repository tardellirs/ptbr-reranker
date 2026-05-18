"""Publish a trained checkpoint to the Hugging Face Hub.

Refuses to push unless the release criteria in docs/quality-tests.md are met
(metrics file exists and thresholds pass).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

RELEASE_THRESHOLDS = {
    "mmarco_mrr10_gain_over_serafim_ir": 0.03,
    "ece": 0.10,
    "robustness_drop_no_accents": 0.10,
    "robustness_drop_typos": 0.15,
    "robustness_drop_case": 0.05,
    "qualitative_pass_rate": 0.80,
}


def check_release_criteria(metrics_path: Path) -> tuple[bool, list[str]]:
    """Validate that all release thresholds are satisfied."""
    metrics = json.loads(metrics_path.read_text())
    failures: list[str] = []

    gain = metrics.get("mmarco_mrr10_gain_over_serafim_ir")
    if gain is None or gain < RELEASE_THRESHOLDS["mmarco_mrr10_gain_over_serafim_ir"]:
        failures.append(f"mmarco_mrr10_gain_over_serafim_ir={gain} < 0.03")

    ece = metrics.get("ece")
    if ece is None or ece >= RELEASE_THRESHOLDS["ece"]:
        failures.append(f"ece={ece} >= 0.10")

    for k in ("robustness_drop_no_accents", "robustness_drop_typos", "robustness_drop_case"):
        v = metrics.get(k)
        if v is None or v >= RELEASE_THRESHOLDS[k]:
            failures.append(f"{k}={v} >= {RELEASE_THRESHOLDS[k]}")

    pass_rate = metrics.get("qualitative_pass_rate")
    if pass_rate is None or pass_rate < RELEASE_THRESHOLDS["qualitative_pass_rate"]:
        failures.append(f"qualitative_pass_rate={pass_rate} < 0.80")

    return (not failures), failures


def push(
    checkpoint_dir: Path,
    repo_id: str,
    token: str,
    *,
    private: bool = False,
    commit_message: str = "Initial release",
) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(
        folder_path=str(checkpoint_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message,
    )
    print(f"Pushed {checkpoint_dir} → {repo_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Push trained model to Hugging Face Hub")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True, help="JSON file with eval metrics")
    parser.add_argument(
        "--repo-id",
        default=os.environ.get("HF_REPO", "stekel/cross-encoder-albertina-ptbr-mmarco"),
    )
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--message", default="Initial release")
    parser.add_argument("--force", action="store_true", help="Skip release-criteria check")
    args = parser.parse_args()

    if not args.force:
        passed, failures = check_release_criteria(args.metrics)
        if not passed:
            print("Release criteria not met:", file=sys.stderr)
            for f in failures:
                print(f"  - {f}", file=sys.stderr)
            sys.exit(1)
        print("Release criteria satisfied.")

    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN environment variable is required")

    push(args.checkpoint, args.repo_id, token, private=args.private, commit_message=args.message)


if __name__ == "__main__":
    main()
