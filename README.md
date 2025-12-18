# WAYF: Weakening the Artifact of Your Foreign-language translations

Research project for reducing translationese in multilingual-to-Chinese neural machine translation using reinforcement learning with source language classifiers as reward signals.

## Overview

This project implements a novel approach to reduce translationese (translation artifacts) in neural machine translation by:
1. Training a classifier to detect source language from Chinese translations
2. Using the classifier as a reward model in reinforcement learning
3. Fine-tuning translation models to produce more natural Chinese translations

## Project Structure

```
WAYF/
├── configs/              # Configuration files for experiments
│   ├── data_config.yaml
│   ├── classifier_config.yaml
│   ├── sft_config.yaml
│   ├── rl_grpo_config.yaml
│   ├── rl_dpo_config.yaml
│   └── eval_config.yaml
├── data/                 # Data directory (not tracked)
│   ├── pools/           # Data pools (A: classifier, B: RL, C: validation)
│   ├── translations/    # LLM-translated data
│   ├── sft/            # SFT training data
│   └── rl/             # RL training data
├── models/              # Trained models (not tracked)
│   ├── classifier/      # Source language classifier
│   ├── baselines/       # Baseline classifiers
│   ├── sft/            # SFT model checkpoint
│   ├── rl_grpo/        # GRPO-trained model
│   └── rl_dpo/         # DPO-trained model
├── outputs/             # Evaluation results (not tracked)
├── scripts/             # Training and evaluation scripts
│   ├── 01_sample_xlsum.py
│   ├── 02_translate_with_llm.py
│   ├── 03_create_sft_dataset.py
│   ├── 04_create_rl_dataset.py
│   ├── 05_classifier_model.py
│   ├── 06_train_classifier.py
│   ├── 07_train_baselines.py
│   ├── 08_train_sft.py
│   ├── 09_train_rl_grpo.py
│   ├── 10_train_rl_dpo.py
│   ├── 11_evaluate_quality.py
│   └── 12_evaluate_translationese.py
├── utils/               # Utility modules
│   ├── __init__.py
│   ├── data_utils.py
│   ├── metrics.py
│   └── model_utils.py
├── project_procedure_steps.md
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/WAYF.git
cd WAYF

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Pipeline

### 1. Data Preparation

```bash
# Sample XL-Sum data and split articles into 5-7 sentence paragraphs
# Dataset: csebuetnlp/xlsum (uses 'text' field)
# Creates equal number of paragraphs per language
python scripts/01_sample_xlsum.py \
    --languages english french japanese korean russian spanish \
    --paragraphs_per_lang 1000 \
    --min_sentences 5 \
    --max_sentences 7 \
    --output_dir data/pools

# Translate paragraphs with LLM
python scripts/02_translate_with_llm.py \
    --input_file data/pools/pool_a.jsonl \
    --output_file data/translations/pool_a_translated.jsonl \
    --model_name gpt-4

# Create SFT dataset
python scripts/03_create_sft_dataset.py \
    --input_file data/translations/pool_a_translated.jsonl \
    --output_file data/sft/train.jsonl

# Create RL dataset
python scripts/04_create_rl_dataset.py \
    --mode grpo \
    --input_file data/pools/pool_b.jsonl \
    --output_file data/rl/pool_b.jsonl
```

### 2. Classifier Training

```bash
# Train source language classifier
python scripts/06_train_classifier.py \
    --train_file data/translations/pool_a_train.jsonl \
    --valid_file data/translations/pool_a_valid.jsonl \
    --test_file data/translations/pool_a_test.jsonl \
    --output_dir models/classifier

# Train baseline classifiers
python scripts/07_train_baselines.py \
    --train_file data/translations/pool_a_train.jsonl \
    --test_file data/translations/pool_a_test.jsonl \
    --output_dir models/baselines
```

### 3. SFT Training

```bash
python scripts/08_train_sft.py \
    --train_file data/sft/train.jsonl \
    --valid_file data/sft/valid.jsonl \
    --output_dir models/sft \
    --model_name Qwen/Qwen2-7B-Instruct
```

### 4. RL Training

```bash
# GRPO
python scripts/09_train_rl_grpo.py \
    --train_file data/rl/pool_b.jsonl \
    --sft_model_path models/sft \
    --classifier_path models/classifier/best_model.pt \
    --label_map_path models/classifier/label_map.json \
    --output_dir models/rl_grpo

# DPO (alternative)
python scripts/10_train_rl_dpo.py \
    --train_file data/rl/dpo_train.jsonl \
    --valid_file data/rl/dpo_valid.jsonl \
    --sft_model_path models/sft \
    --output_dir models/rl_dpo
```

### 5. Evaluation

```bash
# Translation quality
python scripts/11_evaluate_quality.py \
    --eval_file data/pools/pool_c.jsonl \
    --sft_model_path models/sft \
    --rl_model_path models/rl_grpo \
    --output_file outputs/quality_results.json

# Translationese reduction
python scripts/12_evaluate_translationese.py \
    --eval_file data/pools/pool_c.jsonl \
    --probe_classifier_path models/probe_classifier/best_model.pt \
    --label_map_path models/probe_classifier/label_map.json \
    --sft_model_path models/sft \
    --rl_model_path models/rl_grpo \
    --output_file outputs/translationese_results.json
```

## Key Features

- **Disjoint Data Pools**: Separate data for classifier training, RL, and validation
- **Source Language Classifier**: RoBERTa-based classifier for detecting translationese
- **RL with Reward Shaping**: Combines classifier reward, quality metrics, and KL penalty
- **Multiple RL Algorithms**: Support for GRPO (on-policy) and DPO (offline)
- **Comprehensive Evaluation**: Translation quality (chrF, BLEU, COMET) and translationese metrics

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- TRL (Transformer Reinforcement Learning)
- PEFT (Parameter-Efficient Fine-Tuning)
- sacrebleu
- scikit-learn
- Optional: COMET, fasttext

## Citation

```bibtex
@article{wayf2025,
  title={Weakening the Artifact of Your Foreign-language translations},
  author={Your Name},
  year={2025}
}
```

## License

MIT License

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Contact

For questions or issues, please open an issue on GitHub.
