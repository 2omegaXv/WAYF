"""
Training script for source language classifier.
Implements step 2.2 from project procedure.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import sys

sys.path.append(str(Path(__file__).parent))
from classifier_model import SourceLanguageClassifier, FrozenBackboneClassifier


class TranslationDataset(Dataset):
    """Dataset for classifier training."""
    
    def __init__(self, data_file: Path, tokenizer, label_map: Dict[str, int], max_length: int = 512):
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.max_length = max_length
        
        # Load data
        self.examples = []
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


def create_label_map(data_file: Path) -> Dict[str, int]:
    """Create mapping from language to label ID."""
    languages = set()
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
    parser.add_argument("--train_file", type=str, required=True, help="Training data file")
    parser.add_argument("--valid_file", type=str, required=True, help="Validation data file")
    parser.add_argument("--test_file", type=str, required=True, help="Test data file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--model_name", type=str, default="hfl/chinese-roberta-wwm-ext")
    parser.add_argument("--frozen_backbone", action="store_true", help="Use frozen backbone baseline")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create label map
    label_map = create_label_map(Path(args.train_file))
    num_languages = len(label_map)
    print(f"Number of languages: {num_languages}")
    print(f"Languages: {list(label_map.keys())}")
    
    # Save label map
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "label_map.json", 'w') as f:
        json.dump(label_map, f, indent=2)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Create datasets
    train_dataset = TranslationDataset(Path(args.train_file), tokenizer, label_map, args.max_length)
    valid_dataset = TranslationDataset(Path(args.valid_file), tokenizer, label_map, args.max_length)
    test_dataset = TranslationDataset(Path(args.test_file), tokenizer, label_map, args.max_length)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # Create model
    if args.frozen_backbone:
        model = FrozenBackboneClassifier(num_languages, args.model_name)
    else:
        model = SourceLanguageClassifier(num_languages, args.model_name)
    
    model = model.to(device)
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    # Training loop
    best_valid_f1 = 0
    
    for epoch in range(args.epochs):
        print(f"\n=== Epoch {epoch + 1}/{args.epochs} ===")
        
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        print(f"Train loss: {train_loss:.4f}")
        
        valid_metrics = evaluate(model, valid_loader, device)
        print(f"Valid - Loss: {valid_metrics['loss']:.4f}, "
              f"Acc: {valid_metrics['accuracy']:.4f}, "
              f"F1: {valid_metrics['macro_f1']:.4f}, "
              f"AUC: {valid_metrics['auc']:.4f}")
        
        # Save best model
        if valid_metrics['macro_f1'] > best_valid_f1:
            best_valid_f1 = valid_metrics['macro_f1']
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            print(f"Saved best model (F1: {best_valid_f1:.4f})")
    
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
    
    # Save tokenizer
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
