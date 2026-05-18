"""Validate `docs/model-card.md` and optionally sync it to the Hugging Face Hub."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

MODEL_CARD_PATH = Path("docs/model-card.md")
REQUIRED_FRONTMATTER_KEYS = {
    "language",
    "license",
    "library_name",
    "pipeline_tag",
    "tags",
    "datasets",
    "base_model",
    "model-index",
}


def parse_frontmatter(path: Path) -> dict[str, object]:
    import yaml

    content = path.read_text(encoding="utf-8")
    if not content.startswith("---\n"):
        raise SystemExit(f"{path} is missing YAML frontmatter")
    end = content.find("\n---\n", 4)
    if end == -1:
        raise SystemExit(f"{path} has malformed frontmatter (no closing ---)")
    return yaml.safe_load(content[4:end])  # type: ignore[no-any-return]


def validate(path: Path) -> None:
    frontmatter = parse_frontmatter(path)
    missing = REQUIRED_FRONTMATTER_KEYS - frontmatter.keys()
    if missing:
        raise SystemExit(f"Missing required frontmatter keys: {sorted(missing)}")
    if frontmatter.get("language") != ["pt"] and frontmatter.get("language") != "pt":
        raise SystemExit("language must be 'pt' (Portuguese)")
    print(f"OK: {path} frontmatter is valid")


def publish(path: Path, repo_id: str, token: str | None) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=str(path),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Sync model card from docs/model-card.md",
    )
    print(f"Uploaded {path} → {repo_id}/README.md")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate / publish model card")
    parser.add_argument("--path", type=Path, default=MODEL_CARD_PATH)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()

    if not (args.validate or args.publish):
        parser.error("Specify --validate or --publish")

    if not args.path.exists():
        sys.exit(f"{args.path} not found")

    if args.validate:
        validate(args.path)

    if args.publish:
        repo_id = os.environ.get("HF_REPO", "stekel/cross-encoder-albertina-ptbr-mmarco")
        token = os.environ.get("HF_TOKEN")
        if not token:
            sys.exit("HF_TOKEN environment variable is required to publish")
        publish(args.path, repo_id, token)


if __name__ == "__main__":
    main()
