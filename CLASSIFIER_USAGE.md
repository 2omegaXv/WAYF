# Classifier Usage (Source Language Detection)

This document explains how to train and run the **source-language classifier**.

Most of the configurations are in [configs/classifier_config.yaml](configs/classifier_config.yaml). We only manipulate whether the model is loaded from disk (`--offline`), whether we use a frozen backbone during the whole training (`--frozen_backbone`), and how many epochs to freeze the backbone (`--freeze_backbone_epochs`). The main training script is [scripts/05_train_classifier.py](scripts/05_train_classifier.py).

---

## Run for first time (download backbone from huggingface)

Pick a place (e.g. `models/hf_backbones/chinese-roberta-wwm-ext`) to store the backbone model, fill it in [configs/classifier_config.yaml](configs/classifier_config.yaml) under `hf.local_dir`. Then run

```bash
python scripts/05_train_classifier.py --config configs/classifier_config.yaml
```

## Subsequent runs (offline-only)

First change the `model.backbone` in [configs/classifier_config.yaml](configs/classifier_config.yaml) to the `local_dir` path where the backbone model is stored. Then run:

```bash
python scripts/05_train_classifier.py --config configs/classifier_config.yaml --offline
```

If the snapshot folder is incomplete/missing, `--offline` will fail (by design).

---

## Optional: Use a Custom HF Cache Directory

If you don’t want to snapshot to `hf.local_dir`, you can still stabilize downloads by pointing Hugging Face to a custom cache directory. Specify it in [configs/classifier_config.yaml](configs/classifier_config.yaml) under `hf.cache_dir`.

This still relies on Hugging Face cache semantics (not a full repo snapshot). The `hf.local_dir` approach is more “portable”.

---              

## Two-Stage Transfer Learning Schedule (Freeze → Unfreeze)

### Option 1 (single command): freeze for N epochs, then unfreeze automatically

```bash
python scripts/05_train_classifier.py --config configs/classifier_config.yaml --offline --freeze_backbone_epochs 1
```

### Option 2 (two commands): resume from Stage A checkpoint

Stage A (freeze the backbone for the whole run):

First set the epochs in the YAML to the desired number of frozen epochs (e.g. 2), then set the output directory of Stage A (e.g. `models/classifier_stageA`). After that, run:

```bash
python scripts/05_train_classifier.py --config configs/classifier_config.yaml --offline --frozen_backbone
```

Stage B (resume and fine-tune end-to-end):

```bash
python scripts/05_train_classifier.py --config configs/classifier_config.yaml --offline --resume_from models/classifier_stageA/best_model.pt
```

## LoRA Adapters

Use `--lora` tag in terminal to enable LoRA adapters in the backbone model. For example:

```bash
python scripts/05_train_classifier.py --config configs/classifier_config.yaml --offline --freeze_backbone_epochs 1 --lora
```

The LoRA hyperparameters (rank, alpha, dropout) can be set in [configs/classifier_config.yaml](configs/classifier_config.yaml) under `lora`.

---

## Baseline with LLMs

Run the LLM baseline with:

```bash
python scripts/classifier_baseline.py --config configs/classifier_baseline_config.yaml
```


## Troubleshooting

- If you see network-related errors, ensure you are using `--hf_local_dir ... --offline` and that the folder exists.
- If you want to hard-enforce offline across all Hugging Face calls, you can also set `HF_HUB_OFFLINE=1` in your environment, but the script-level `--offline` should be sufficient.

---

## What Gets Saved (Artifacts)

When you run [scripts/05_train_classifier.py](scripts/05_train_classifier.py), it writes the following to `--output_dir`:

- `label_map.json`
  - Mapping of source-language string (e.g. `"english"`) → integer class id.
- `best_model.pt`
  - PyTorch `state_dict()` of the best checkpoint (selected by validation macro-F1).
- `test_metrics.json`
  - Final metrics on the test split.
- Tokenizer files (via `tokenizer.save_pretrained(output_dir)`), typically:
  - `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`
  - plus vocab/merges files depending on the tokenizer

Additionally, if you specify `local_dir` in the YAML, the Hugging Face backbone model will be stored **there** (snapshot of the repo), e.g.:

- `models/hf_backbones/chinese-roberta-wwm-ext/` (contains config + weights + tokenizer files from the Hub snapshot)