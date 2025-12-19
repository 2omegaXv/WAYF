"""
Training script for source language classifier.
Implements step 2.2 from project procedure.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List
import yaml
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import sys

sys.path.append(str(Path(__file__).parent))
from classifier_model import (
    FrozenBackboneClassifier,
    SourceLanguageClassifier,
    resolve_pretrained_source,
)


class TranslationDataset(Dataset):
    """Dataset for classifier training."""

    def __init__(
        self,
        data_files: Path | List[Path],
        tokenizer,
        label_map: Dict[str, int],
        max_length: int = 512,
    ):
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.max_length = max_length

        if isinstance(data_files, Path):
            data_files = [data_files]
        else:
            data_files = list(data_files)

        # Load data
        self.examples = []
        for data_file in data_files:
            with open(data_file, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line)
                    zh_text = item.get('zh_mt', '')
                    src_lang = item.get('src_lang', '')

                    if zh_text and src_lang in label_map:
                        self.examples.append({
                            'text': zh_text,
                            'label': label_map[src_lang]
                        })
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            example['text'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(example['label'], dtype=torch.long)
        }


def create_label_map(data_files: Path | List[Path]) -> Dict[str, int]:
    """Create mapping from language to label ID."""
    if isinstance(data_files, Path):
        data_files = [data_files]
    else:
        data_files = list(data_files)

    languages = set()
    for data_file in data_files:
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                languages.add(item.get('src_lang', ''))

    return {lang: idx for idx, lang in enumerate(sorted(languages))}


def compute_metrics(logits: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Compute evaluation metrics."""
    preds = np.argmax(logits, axis=1)
    
    # Accuracy
    accuracy = accuracy_score(labels, preds)
    
    # Macro F1
    macro_f1 = f1_score(labels, preds, average='macro')
    
    # AUC (one-vs-rest)
    try:
        from sklearn.preprocessing import label_binarize
        n_classes = logits.shape[1]
        labels_bin = label_binarize(labels, classes=range(n_classes))
        probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
        auc = roc_auc_score(labels_bin, probs, average='macro', multi_class='ovr')
    except:
        auc = 0.0
    
    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'auc': auc
    }


def train_epoch(model, dataloader, optimizer, scheduler, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs['loss']
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    """Evaluate model."""
    model.eval()
    all_logits = []
    all_labels = []
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            
            total_loss += outputs['loss'].item()
            all_logits.append(outputs['logits'].cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    metrics = compute_metrics(all_logits, all_labels)
    metrics['loss'] = total_loss / len(dataloader)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train source language classifier")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional YAML config file (e.g. configs/classifier_config.yaml). If provided, YAML drives training settings.",
    )
    # NOTE: When --config is provided, these CLI flags are ignored (YAML is source of truth).
    # They remain available to preserve backwards compatibility.
    parser.add_argument("--train_file", type=str, default=None, help="Training data file (ignored if --config is set)")
    parser.add_argument(
        "--train_file_supplement",
        type=str,
        default=None,
        help="Optional supplemental training data file (ignored if --config is set)",
    )
    parser.add_argument("--valid_file", type=str, default=None, help="Validation data file (ignored if --config is set)")
    parser.add_argument("--test_file", type=str, default=None, help="Test data file (ignored if --config is set)")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (ignored if --config is set)")
    parser.add_argument("--model_name", type=str, default=None, help="Backbone model name/path (ignored if --config is set)")
    parser.add_argument("--hf_cache_dir", type=str, default=None, help="HF cache dir (ignored if --config is set)")
    parser.add_argument("--hf_local_dir", type=str, default=None, help="HF local snapshot dir (ignored if --config is set)")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="If set, only load model/tokenizer files from disk (no network).",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Optional path to a .pt state_dict to resume from (must match model architecture).",
    )
    parser.add_argument("--frozen_backbone", action="store_true", help="Use frozen backbone baseline")
    parser.add_argument(
        "--freeze_backbone_epochs",
        type=int,
        default=0,
        help=(
            "If >0 and not using --frozen_backbone, freeze the backbone for the first N epochs, "
            "then unfreeze for remaining epochs."
        ),
    )
    parser.add_argument("--max_length", type=int, default=None, help="(ignored if --config is set)")
    parser.add_argument("--batch_size", type=int, default=None, help="(ignored if --config is set)")
    parser.add_argument("--epochs", type=int, default=None, help="(ignored if --config is set)")
    parser.add_argument("--lr", type=float, default=None, help="(ignored if --config is set)")
    parser.add_argument("--warmup_ratio", type=float, default=None, help="(ignored if --config is set)")
    parser.add_argument("--seed", type=int, default=None, help="(ignored if --config is set)")
    
    args = parser.parse_args()

    # If --config is provided, YAML is the source of truth for training settings.
    cfg = {}
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        # model
        model_cfg = cfg.get("model", {}) or {}
        args.model_name = model_cfg.get("backbone")

        # training
        training_cfg = cfg.get("training", {}) or {}
        args.max_length = int(training_cfg.get("max_length", 512))
        args.batch_size = int(training_cfg.get("batch_size", 32))
        args.epochs = int(training_cfg.get("epochs", 5))
        args.lr = float(training_cfg.get("learning_rate", 2e-5))
        args.warmup_ratio = float(training_cfg.get("warmup_ratio", 0.1))
        args.seed = int(training_cfg.get("seed", 42))

        # paths
        paths_cfg = cfg.get("paths", {}) or {}
        args.train_file = paths_cfg.get("train_file")
        args.train_file_supplement = paths_cfg.get("train_file_supplement")
        args.valid_file = paths_cfg.get("valid_file")
        args.test_file = paths_cfg.get("test_file")
        args.output_dir = paths_cfg.get("output_dir")

        # Optional HF cache/snapshot paths from YAML if present
        hf_cfg = cfg.get("hf", {}) or {}
        if args.hf_cache_dir is None:
            args.hf_cache_dir = hf_cfg.get("cache_dir")
        if args.hf_local_dir is None:
            args.hf_local_dir = hf_cfg.get("local_dir")

    else:
        # No config: fall back to CLI defaults
        args.model_name = args.model_name or "hfl/chinese-roberta-wwm-ext"
        args.max_length = 512 if args.max_length is None else args.max_length
        args.batch_size = 32 if args.batch_size is None else args.batch_size
        args.epochs = 5 if args.epochs is None else args.epochs
        args.lr = 2e-5 if args.lr is None else args.lr
        args.warmup_ratio = 0.1 if args.warmup_ratio is None else args.warmup_ratio
        args.seed = 42 if args.seed is None else args.seed

    # Validate required values after config/CLI resolution
    missing = [
        name
        for name in ("train_file", "valid_file", "test_file", "output_dir", "model_name")
        if not getattr(args, name)
    ]
    if missing:
        raise ValueError(
            "Missing required settings: " + ", ".join(missing) +
            ". Provide them in --config YAML (recommended) or via CLI flags."
        )

    if args.train_file_supplement:
        supplement_path = Path(args.train_file_supplement)
        if not supplement_path.exists():
            raise FileNotFoundError(f"train_file_supplement not found: {supplement_path}")

    def set_backbone_trainable(model_obj, trainable: bool):
        if not hasattr(model_obj, "backbone"):
            return
        for param in model_obj.backbone.parameters():
            param.requires_grad = trainable
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Initialize Weights & Biases (optional)
    wandb_run = None
    wandb_cfg = (cfg.get("wandb", {}) or {}) if args.config else {}

    if wandb_cfg.get("enabled", False):
        try:
            import wandb  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "wandb is enabled in config but not importable. "
                "Install it with: pip install wandb"
            ) from e

        api_key = wandb_cfg.get("api_key")
        if api_key:
            os.environ.setdefault("WANDB_API_KEY", str(api_key))

        wandb_mode = wandb_cfg.get("mode", "online")
        wandb_run = wandb.init(
            project=wandb_cfg.get("project") or "WAYF",
            entity=wandb_cfg.get("entity"),
            name=wandb_cfg.get("name"),
            tags=wandb_cfg.get("tags") or None,
            notes=wandb_cfg.get("notes"),
            mode=wandb_mode,
            config={
                "model_name": args.model_name,
                "max_length": args.max_length,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "lr": args.lr,
                "warmup_ratio": args.warmup_ratio,
                "seed": args.seed,
                "frozen_backbone": bool(args.frozen_backbone),
                "freeze_backbone_epochs": int(args.freeze_backbone_epochs),
                "offline": bool(args.offline),
                "hf_cache_dir": args.hf_cache_dir,
                "hf_local_dir": args.hf_local_dir,
                "train_file": args.train_file,
                "valid_file": args.valid_file,
                "test_file": args.test_file,
                "output_dir": args.output_dir,
            },
        )
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create label map (train = primary + optional supplement)
    train_files: List[Path] = [Path(args.train_file)]
    if args.train_file_supplement:
        train_files.append(Path(args.train_file_supplement))

    label_map = create_label_map(train_files)
    num_languages = len(label_map)
    print(f"Number of languages: {num_languages}")
    print(f"Languages: {list(label_map.keys())}")
    
    # Save label map
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "label_map.json", 'w') as f:
        json.dump(label_map, f, indent=2)
    
    # Load tokenizer
    resolved_backbone = resolve_pretrained_source(
        args.model_name,
        cache_dir=args.hf_cache_dir,
        local_dir=args.hf_local_dir,
        local_files_only=args.offline,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        resolved_backbone,
        cache_dir=args.hf_cache_dir,
        local_files_only=args.offline,
    )
    
    # Create datasets
    train_dataset = TranslationDataset(train_files, tokenizer, label_map, args.max_length)
    valid_dataset = TranslationDataset(Path(args.valid_file), tokenizer, label_map, args.max_length)
    test_dataset = TranslationDataset(Path(args.test_file), tokenizer, label_map, args.max_length)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # Create model
    if args.frozen_backbone:
        model = FrozenBackboneClassifier(
            num_languages,
            args.model_name,
            cache_dir=args.hf_cache_dir,
            local_dir=args.hf_local_dir,
            local_files_only=args.offline,
        )
    else:
        model = SourceLanguageClassifier(
            num_languages,
            args.model_name,
            cache_dir=args.hf_cache_dir,
            local_dir=args.hf_local_dir,
            local_files_only=args.offline,
        )

        if args.freeze_backbone_epochs > 0:
            set_backbone_trainable(model, trainable=False)
    
    model = model.to(device)

    # Resume from checkpoint if provided
    if args.resume_from:
        ckpt_path = Path(args.resume_from)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"resume_from checkpoint not found: {ckpt_path}")
        state = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state)
        print(f"Resumed weights from: {ckpt_path}")
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    # Training loop
    best_valid_f1 = 0
    
    for epoch in range(args.epochs):
        print(f"\n=== Epoch {epoch + 1}/{args.epochs} ===")

        if (not args.frozen_backbone) and args.freeze_backbone_epochs > 0:
            if epoch == args.freeze_backbone_epochs:
                set_backbone_trainable(model, trainable=True)
                print("Unfroze backbone; continuing end-to-end fine-tuning")
        
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        print(f"Train loss: {train_loss:.4f}")
        
        valid_metrics = evaluate(model, valid_loader, device)
        print(f"Valid - Loss: {valid_metrics['loss']:.4f}, "
              f"Acc: {valid_metrics['accuracy']:.4f}, "
              f"F1: {valid_metrics['macro_f1']:.4f}, "
              f"AUC: {valid_metrics['auc']:.4f}")

        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch + 1,
                    "train/loss": train_loss,
                    "valid/loss": valid_metrics["loss"],
                    "valid/accuracy": valid_metrics["accuracy"],
                    "valid/macro_f1": valid_metrics["macro_f1"],
                    "valid/auc": valid_metrics["auc"],
                    "lr": optimizer.param_groups[0].get("lr", args.lr),
                },
                step=epoch + 1,
            )
        
        # Save best model
        if valid_metrics['macro_f1'] > best_valid_f1:
            best_valid_f1 = valid_metrics['macro_f1']
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            print(f"Saved best model (F1: {best_valid_f1:.4f})")
            if wandb_run is not None:
                wandb_run.summary["best_valid_macro_f1"] = float(best_valid_f1)
    
    # Load best model and evaluate on test
    model.load_state_dict(torch.load(output_dir / "best_model.pt"))
    test_metrics = evaluate(model, test_loader, device)
    
    print(f"\n=== Test Results ===")
    print(f"Test - Loss: {test_metrics['loss']:.4f}, "
          f"Acc: {test_metrics['accuracy']:.4f}, "
          f"F1: {test_metrics['macro_f1']:.4f}, "
          f"AUC: {test_metrics['auc']:.4f}")
    
    # Save test metrics
    with open(output_dir / "test_metrics.json", 'w') as f:
        json.dump(test_metrics, f, indent=2)

    if wandb_run is not None:
        wandb_run.log(
            {
                "test/loss": test_metrics["loss"],
                "test/accuracy": test_metrics["accuracy"],
                "test/macro_f1": test_metrics["macro_f1"],
                "test/auc": test_metrics["auc"],
            },
            step=args.epochs + 1,
        )
        wandb_run.finish()
    
    # Save tokenizer
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
