# Where Are You From: Source Language Detection in Translation

Research project for reducing translationese in multilingual-to-Chinese neural machine translation using reinforcement learning with source language classifiers as reward signals.

## Overview

This project implements a novel approach to reduce translationese (translation artifacts) in neural machine translation by:
1. Training a classifier to detect source language from Chinese translations
2. Using the classifier as a reward model in reinforcement learning
3. Fine-tuning translation models to produce more natural Chinese translations

## Installation

```bash
# Clone the repository
git clone https://github.com/2omegaXv/WAYF
cd WAYF

# Install dependencies
pip install -r requirements.txt
```

## Pipeline

### 1. Data Preparation

See [TRANSLATION_USAGE.md](TRANSLATION_USAGE.md) for detailed instructions.

### 2. Classifier Training

See [CLASSIFIER_USAGE.md](CLASSIFIER_USAGE.md) for detailed instructions.

### 3. Visualize Classifier

```bash
# Visualize UMAP of BERT Embeddings
python visualize/visualize_umap.py --use_baseline

# Visualize Word Heatmap
python visualize/visualize_token_heatmap.py --input_file visualize/test_samples.json
```

### 4. RL Training

```bash
python scripts/06_train_rl_grpo.py --config configs/rl_grpo_config.yaml
```

### 5. RL Evaluation

```bash
# Evaluate predicted source language distribution before and after RL fine-tuning
python scripts/07_evaluate_language_distribution.py \
    --base_model /path/to/base/model \
    --rl_model /path/to/rl/model \
    --input_data /path/to/evaluation/data.jsonl \
    --classifier_path /path/to/classifier/model \
```

## Key Features

- **Disjoint Data Pools**: Separate data for classifier training, RL, and validation
- **Source Language Classifier**: RoBERTa-based classifier for detecting translationese
- **RL with Reward Shaping**: Combines classifier reward, quality metrics, and KL penalty