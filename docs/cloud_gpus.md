# Cloud GPU Providers — Quick Reference

Compact cheat sheet for renting GPU compute. Prices as observed 2026-05.
All workflows assume `.env` at repo root with provider keys + `HF_TOKEN`.

## Provider comparison (Albertina-100m cross-encoder workload)

| Provider | GPU | $/h | VRAM | Stability | Best for |
|---|---|---|---|---|---|
| **GPUhub** 5090 | RTX 5090 | $0.36 | 32 GB | datacenter | **default** — cheapest fast Ada |
| **GPUhub** 4080 Super | RTX 4080S | $0.23 | 32 GB | datacenter | budget train/eval |
| **GPUhub** 4090 | RTX 4090 (48GB!) | $0.44 | 48 GB | datacenter | mining (needs >32GB VRAM) |
| GPUhub A800 / PRO 6000 | A800 / RTX PRO 6000 | $0.91-0.93 | 80-96 GB | datacenter | Albertina-900M |
| Runpod Secure 4090 | RTX 4090 | $0.69 | 24 GB | very stable | API-driven flows |
| Runpod Secure 5090 | RTX 5090 | $0.99 | 32 GB | very stable | premium, only if GPUhub out |
| Runpod Community 4090 | RTX 4090 | $0.34 | 24 GB | can be evicted | short jobs only |
| Salad medium 3090 | RTX 3090 | $0.197 | 24 GB | **migrates often** | one-off mining if cheap matters |

**Rule of thumb**: GPUhub by default, Runpod Secure as fallback. Salad community only for jobs that fit in <2h or have full HF-backup recovery.

## GPUhub (gpuhub.com — international branch of AutoDL)

**Catch**: API only supports *Elastic Deployment* (inference-style). For SSH-accessible *Cloud Container Instance* (training), **create via web console manually**. Document the SSH host/port and provide to whoever runs the job.

- Console: `https://www.gpuhub.com/console/home/host`
- Docs: `https://docs.gpuhub.com/`
- API base: `https://api.gpuhub.com/api/v1/dev/...` (POST + `Authorization: <token>` header — no "Bearer")
- Auth: token from console → put as `GPUHUB_TOKEN` in `.env`

```bash
# Check balance (the only useful API endpoint for us)
curl -sS -H "Authorization: $GPUHUB_TOKEN" -X POST \
  "https://api.gpuhub.com/api/v1/dev/wallet/balance" \
  -H "Content-Type: application/json"
# assets is in milli-USD: divide by 1000
```

**Workflow for a training job:**
1. Console → Create Instance → pick GPU + image (PyTorch 2.8 / CUDA 12.8 conda env pre-installed at `/root/miniconda3`)
2. Copy SSH details from instance page → typical form: `ssh -p <PORT> root@connect.singapore-a.gpuhub.com` + password
3. First login: `ssh-copy-id` your public key, then key-only after that
   ```bash
   sshpass -p "$PASS" ssh -p $P root@$HOST \
     "mkdir -p ~/.ssh && echo '$(cat ~/.ssh/id_salad.pub)' >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"
   ```
4. All work goes in `/root/autodl-tmp/` (50 GB dedicated disk; `/` is only 30 GB overlay — fills fast)
5. Python is at `/root/miniconda3/bin/python` (not in PATH by default). Either prepend PATH or use full path.
6. **No tmux preinstalled** — `apt install -y tmux` first.
7. **Image is in Chinese locale** — banner says "docs.runpod.io" but that's because the image is `runpod/pytorch:*` from Docker Hub; the host is GPUhub.

**Gotchas:**
- Disk: 50 GB data + 30 GB system. With FAISS mining cache (27 GB) + checkpoints (5 GB) you're close to full — `rm -rf` aggressively when phases finish.
- API has 12 endpoints; everything else is web-only. Listing instances via API does work (`POST /api/v1/instance` with `{"page_index":1,"page_size":10}`).
- Console is partly Chinese; navigate by icons or use browser translate.

## Runpod (runpod.io)

- Two tiers: **Community** (cheap, evictable) and **Secure** (datacenter, on-demand).
- Full API for everything, including creating/deleting pods.
- API base: `https://api.runpod.io/graphql`
- Auth: API key as `Bearer` token. Store as `RUNPOD_API_KEY` in `.env`. Multiple accounts: `RUNPOD_API_KEY_ACC2`, etc.

```bash
# Balance + active pods
curl -sS -X POST -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query":"{ myself { clientBalance currentSpendPerHr pods { id name machine { gpuDisplayName } } } }"}' \
  https://api.runpod.io/graphql | jq

# Deploy on-demand (Secure or Community via cloudType)
curl -sS -X POST -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d "$(python3 -c "
import json, os
print(json.dumps({'query':'mutation Deploy(\$input: PodFindAndDeployOnDemandInput!) { podFindAndDeployOnDemand(input: \$input) { id machine { gpuDisplayName } } }',
'variables':{'input':{
    'cloudType':'SECURE',  # or 'COMMUNITY'
    'gpuCount':1, 'volumeInGb':0, 'containerDiskInGb':40,
    'minVcpuCount':8, 'minMemoryInGb':16,
    'gpuTypeId':'NVIDIA GeForce RTX 5090',
    'name':'job-name', 'imageName':'runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04',
    'ports':'22/tcp', 'volumeMountPath':'/workspace',
    'env':[{'key':'PUBLIC_KEY','value':open(os.path.expanduser('~/.ssh/id_salad.pub')).read().strip()}],
}}}))")" \
  https://api.runpod.io/graphql

# Terminate
curl -sS -X POST -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query":"mutation { podTerminate(input: {podId: \"<POD_ID>\"}) }"}' \
  https://api.runpod.io/graphql
```

Helper script in repo: `scripts/runpod_terminate.sh <pod_id>` (refuses without arg + lists remaining pods).

**Gotchas:**
- Community pods migrate without warning. Salad-level instability — avoid for jobs >2h.
- Community pods sometimes ship with broken CUDA passthrough (driver/container mismatch). Probe with `python3 -c 'import torch; torch.cuda.mem_get_info()'` before committing to a long job.
- `cloudType: SECURE` requires Secure-tier GPUs (some classes Community-only).
- Pod port for SSH is dynamic — get it from the `runtime.ports` field after status flips to `RUNNING`.

## Salad (salad.com)

Distributed GPU network — literally gaming PCs sharing compute via app. Cheapest by far, but **migrations are routine** during long jobs.

- API: `https://api.salad.com/api/public/organizations/{ORG}/projects/{PROJECT}/containers/...`
- Auth: `Salad-Api-Key: $SALAD_API_KEY` header
- Organization/project names are part of every URL. Get from web console.

```bash
# GPU classes (catalog + pricing per priority tier)
curl -sS -H "Salad-Api-Key: $SALAD_API_KEY" \
  "https://api.salad.com/api/public/organizations/$ORG/gpu-classes" | jq '.items[] | {name, prices}'

# Deploy
curl -sS -X POST -H "Salad-Api-Key: $SALAD_API_KEY" -H "Content-Type: application/json" \
  "https://api.salad.com/api/public/organizations/$ORG/projects/$PROJ/containers" \
  -d @payload.json   # see below
```

**Container group payload skeleton:**
```json
{
  "name": "dns-safe-name",
  "display_name": "Human Readable (no parens, regex ^[ ,-.0-9A-Za-z]+$)",
  "autostart_policy": true,
  "replicas": 1,
  "restart_policy": "never",
  "container": {
    "image": "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04",
    "priority": "medium",
    "resources": {
      "cpu": 8,
      "memory": 49152,
      "gpu_classes": ["<gpu-uuid-1>", "<gpu-uuid-2>"],
      "storage_amount": 53687091200
    },
    "environment_variables": {"HF_TOKEN": "<token>", "PUBLIC_KEY": "<ssh pub key>"},
    "command": ["bash", "-c", "sleep infinity"]
  }
}
```

**Gotchas:**
- SSH key: must be added in Salad **Portal → SSH Keys** (one-time, manual). The `PUBLIC_KEY` env var only works on some images.
- `display_name` regex is restrictive; no parentheses, slashes, colons.
- `priority: medium` is the cost/availability sweet spot. `high` rarely worth it for batch jobs.
- SSH host/port surfaces in `instances[].ssh_ip` + `instances[].ssh_port` after `ready=True`.
- Salad SSH wrapper banner = `Connecting to container <uuid>` prefix on stdout — strip with `| grep -v '^Connecting'` when scripting.
- Multi-line commands over SSH need `bash -c "'...'"` quoting — otherwise the wrapper splits on whitespace.

## Common workflow (any provider)

Once the pod has SSH + Python + repo cloned:

```bash
# 1. install deps
pip install -q -e ".[dev]" huggingface_hub faiss-gpu-cu12

# 2. data + checkpoint snapshot from HF
python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='tardellirs/ptbr-reranker-mining-cache',
                  repo_type='dataset', local_dir='data/processed',
                  allow_patterns=['hard_negatives.cache.passage_emb.bin*'])
"

# 3. launch training in tmux (survives SSH drops)
tmux new-session -d -s train 'python -m src.train --config configs/train_xxx.yaml --resume-from-checkpoint runs/xxx/checkpoint-NNN 2>&1 | tee logs/train.log; echo TRAIN_EXIT=$? >> logs/train.log; sleep 600'

# 4. always run hf_backup_loop alongside training in a separate tmux
tmux new-session -d -s hfbackup 'HF_TOKEN=$HF_TOKEN HF_BACKUP_DIR=runs/xxx HF_BACKUP_REPO=tardellirs/xxx-inprogress python -u /workspace/hf_backup_loop_v1x.py'
```

## HF Hub as the universal persistence layer

Independent of provider. Push everything important here so pods are disposable:

- **Model checkpoints**: `tardellirs/<model-name>` (private repo). Push from pod via `upload_folder`. Resume on new pod via `snapshot_download`.
- **Mining caches**: `tardellirs/<project>-mining-cache` (dataset) — encoded passages, FAISS index, query embeddings.
- **Eval results**: `tardellirs/<project>-eval-results` (dataset) — per-query parquet + raw rerank scores for bootstrap CIs.
- **In-progress training**: `tardellirs/<project>-inprogress` (model repo) — checkpoint-N folders pushed by `hf_backup_loop_v1x.py` every save_steps.

HF Hub transfer speeds observed: 30-220 MB/s inbound to pods, 5-50 MB/s outbound from pods. Way faster and more reliable than Mac↔pod scp.

## Picking a provider, decision tree

```
Need >32 GB VRAM (Albertina-900M, IVFFlat-without-PQ on small disk)?
  └─ yes → GPUhub 4090 48GB ($0.44) or PRO 6000 96GB ($0.91)
  └─ no
     ├─ Need API-driven create/destroy at scale?
     │  └─ yes → Runpod Secure 4090 ($0.69) or 5090 ($0.99)
     ├─ Job <2h and OK with eviction?
     │  └─ yes → Salad medium 3090 ($0.197) or Runpod Community 4090 ($0.34)
     └─ default
        └─ GPUhub 5090 32GB ($0.36) if available, else 4080 Super 32GB ($0.23)
```
