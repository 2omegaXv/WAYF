# Classifier Usage (Source Language Detection)

This document explains how to train and run the **source-language classifier** reliably when network connections are unstable.

The classifier backbone is a Hugging Face encoder (default: `hfl/chinese-roberta-wwm-ext`). You can either rely on the default Hugging Face cache, or (recommended) **snapshot the model into a project-local folder** and later load **offline-only**.

---

## What Gets Saved (Artifacts)

When you run [scripts/06_train_classifier.py](scripts/06_train_classifier.py), it writes the following to `--output_dir`:

- `label_map.json`
  - Mapping of source-language string (e.g. `"english"`) → integer class id.
- `best_model.pt`
  - PyTorch `state_dict()` of the best checkpoint (selected by validation macro-F1).
- `test_metrics.json`
  - Final metrics on the test split.
- Tokenizer files (via `tokenizer.save_pretrained(output_dir)`), typically:
  - `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`
  - plus vocab/merges files depending on the tokenizer

Additionally, if you pass `--hf_local_dir`, the Hugging Face backbone model will be stored **there** (snapshot of the repo), e.g.:

- `models/hf_backbones/chinese-roberta-wwm-ext/` (contains config + weights + tokenizer files from the Hub snapshot)

---

## Recommended Workflow: Download Once → Offline Later

### 0) Choose a local backbone folder

Pick a stable project-local directory to store the backbone snapshot, for example:

- `models/hf_backbones/chinese-roberta-wwm-ext`

### 1) First run (downloads/snapshots, then trains)

This will download the backbone (once) into `--hf_local_dir` and then train:

```bash
python scripts/06_train_classifier.py --config configs/classifier_config.yaml
```

### 2) Subsequent runs (offline-only)

This forces Hugging Face to only load files from disk:

```bash
python scripts/06_train_classifier.py --config configs/classifier_config.yaml --offline
```

If the snapshot folder is incomplete/missing, `--offline` will fail (by design), which is what you want to avoid silently falling back to network.

---

## Optional: Use a Custom HF Cache Directory

If you don’t want to snapshot to `--hf_local_dir`, you can still stabilize downloads by pointing Hugging Face to a custom cache directory:

```bash
python scripts/06_train_classifier.py \
  ... \
  --hf_cache_dir /some/big/disk/hf_cache
```

This still relies on Hugging Face cache semantics (not a full repo snapshot). The `--hf_local_dir` approach is more “portable”.

---

## Two-Stage Transfer Learning Schedule (Freeze → Unfreeze)

A common and safe schedule for small-ish datasets is:

1. **Stage A (head warmup):** freeze the backbone for a short period so the classifier head learns quickly.
2. **Stage B (full fine-tune):** unfreeze the backbone and fine-tune end-to-end with a small learning rate.

### Option 1 (single command): freeze for N epochs, then unfreeze automatically

```bash
python scripts/06_train_classifier.py --config configs/classifier_config.yaml --offline --freeze_backbone_epochs 2
```

### Option 2 (two commands): resume from Stage A checkpoint

Stage A (freeze the backbone for the whole run):

```bash
python scripts/06_train_classifier.py \
  --train_file data/translations/pool_a_train.jsonl \
  --valid_file data/translations/pool_a_valid.jsonl \
  --test_file  data/translations/pool_a_test.jsonl \
  --output_dir models/classifier_stageA \
  --hf_local_dir models/hf_backbones/chinese-roberta-wwm-ext \
  --freeze_backbone_epochs 999
```

Stage B (resume and fine-tune end-to-end):

```bash
python scripts/06_train_classifier.py \
  --train_file data/translations/pool_a_train.jsonl \
  --valid_file data/translations/pool_a_valid.jsonl \
  --test_file  data/translations/pool_a_test.jsonl \
  --output_dir models/classifier_stageB \
  --hf_local_dir models/hf_backbones/chinese-roberta-wwm-ext \
  --resume_from models/classifier_stageA/best_model.pt
```

Notes:
- Keep `--model_name` consistent across stages.
- `--resume_from` expects a `state_dict` saved by this training script.

---

## Troubleshooting

- If you see network-related errors, ensure you are using `--hf_local_dir ... --offline` and that the folder exists.
- If you want to hard-enforce offline across all Hugging Face calls, you can also set `HF_HUB_OFFLINE=1` in your environment, but the script-level `--offline` should be sufficient.
