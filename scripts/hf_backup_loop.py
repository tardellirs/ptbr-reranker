"""Watch a training output dir and push checkpoint-N folders to HF Hub as they appear.

Used during long training runs on disposable pods. The training job writes
checkpoints under ``$HF_BACKUP_DIR/checkpoint-NNN/`` on the local SSD; this loop
notices new ones and uploads each to a private HF model repo so that if the pod
dies, training can resume on a fresh pod via ``snapshot_download``.

Environment variables:

- ``HF_TOKEN``        (required) write-scoped HF token
- ``HF_BACKUP_DIR``   (required) the training ``output_dir`` to watch
- ``HF_BACKUP_REPO``  (required) e.g. ``tardellirs/ptbr-reranker-stage1b-inprogress``
- ``HF_BACKUP_POLL``  (optional) seconds between scans (default 60)
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path

from huggingface_hub import HfApi, upload_folder, create_repo


def main() -> int:
    token = os.environ["HF_TOKEN"]
    backup_dir = Path(os.environ["HF_BACKUP_DIR"])
    repo_id = os.environ["HF_BACKUP_REPO"]
    poll = int(os.environ.get("HF_BACKUP_POLL", "60"))

    api = HfApi(token=token)
    try:
        create_repo(repo_id, repo_type="model", private=True, token=token, exist_ok=True)
    except TypeError:
        try:
            create_repo(repo_id, repo_type="model", private=True, token=token)
        except Exception:
            pass

    pushed: set[str] = set()
    print(f"[hf_backup] watching {backup_dir} → {repo_id} every {poll}s", flush=True)
    while True:
        try:
            if backup_dir.exists():
                ckpts = sorted(p for p in backup_dir.iterdir() if p.is_dir() and p.name.startswith("checkpoint-"))
                for ckpt in ckpts:
                    if ckpt.name in pushed:
                        continue
                    # only push when the directory looks complete (has a config + weights)
                    if not (ckpt / "config.json").exists():
                        continue
                    weights_exist = any(ckpt.glob("*.safetensors")) or any(ckpt.glob("*.bin"))
                    if not weights_exist:
                        continue
                    print(f"[hf_backup] uploading {ckpt.name}...", flush=True)
                    upload_folder(
                        folder_path=str(ckpt),
                        path_in_repo=ckpt.name,
                        repo_id=repo_id,
                        repo_type="model",
                        token=token,
                        commit_message=f"backup {ckpt.name}",
                    )
                    pushed.add(ckpt.name)
                    print(f"[hf_backup] pushed {ckpt.name}", flush=True)
                # also keep best/ in sync if present
                best = backup_dir / "best"
                if best.exists() and any(best.glob("*.safetensors")):
                    print(f"[hf_backup] syncing best/...", flush=True)
                    upload_folder(
                        folder_path=str(best),
                        path_in_repo="best",
                        repo_id=repo_id,
                        repo_type="model",
                        token=token,
                        commit_message="sync best/",
                    )
        except Exception as exc:
            print(f"[hf_backup] error: {exc}", flush=True)
            traceback.print_exc()
        time.sleep(poll)


if __name__ == "__main__":
    sys.exit(main() or 0)
