import sys
import json
import argparse
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional, List
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score
import jieba.posseg as pseg

# --- Model definitions (copied from classifier_model.py) ---
def resolve_pretrained_source(
    model_name_or_path: str,
    *,
    cache_dir: Optional[str] = None,
    local_dir: Optional[str] = None,
    local_files_only: bool = False,
    revision: Optional[str] = None,
    token: Optional[str] = None,
) -> str:
    candidate_path = Path(model_name_or_path)
    if candidate_path.exists():
        return str(candidate_path)
    if local_dir is None:
        return model_name_or_path
    from huggingface_hub import snapshot_download
    local_dir_path = Path(local_dir)
    local_dir_path.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=model_name_or_path,
        local_dir=str(local_dir_path),
        local_dir_use_symlinks=False,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        revision=revision,
        token=token,
    )
    return str(local_dir_path)

class ClassifierHead(nn.Module):
    def __init__(self, num_languages: int, hidden_size: int, dropout: float = 0.1, hidden_dim: int = 512):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_languages),
        )
    def forward(self, cls_output: torch.Tensor) -> torch.Tensor:
        return self.classifier(cls_output)

class SourceLanguageClassifier(nn.Module):
    def __init__(
        self,
        num_languages: int,
        model_name: str = "hfl/chinese-roberta-wwm-ext",
        dropout: float = 0.1,
        hidden_dim: int = 512,
        *,
        cache_dir: Optional[str] = None,
        local_dir: Optional[str] = None,
        local_files_only: bool = False,
        revision: Optional[str] = None,
        token: Optional[str] = None,
    ):
        super().__init__()
        self.num_languages = num_languages
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.local_dir = local_dir
        self.local_files_only = local_files_only
        self.revision = revision
        self.token = token
        resolved_source = resolve_pretrained_source(
            model_name,
            cache_dir=cache_dir,
            local_dir=local_dir,
            local_files_only=local_files_only,
            revision=revision,
            token=token,
        )
        self.backbone = AutoModel.from_pretrained(
            resolved_source,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            revision=revision,
            token=token,
        )
        self.hidden_size = self.backbone.config.hidden_size
        self.classifier = ClassifierHead(
            num_languages=num_languages,
            hidden_size=self.hidden_size,
            dropout=dropout,
            hidden_dim=hidden_dim,
        )
        self.loss_fn = nn.CrossEntropyLoss()
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
        return {"loss": loss, "logits": logits}

# --- Entity masking ---
ENTITY_MAP = {
    'nr': '[unused1]',  # Person Name
    'ns': '[unused2]',  # Place Name
    'nt': '[unused3]',  # Organization
    'nz': '[unused4]',  # Other Proper Noun
}
def mask_entities(text: str) -> List[str]:
    words = pseg.cut(text)
    segments = []
    for word, flag in words:
        if flag.startswith('nr'):
            segments.append(ENTITY_MAP['nr'])
        elif flag.startswith('ns'):
            segments.append(ENTITY_MAP['ns'])
        elif flag.startswith('nt'):
            segments.append(ENTITY_MAP['nt'])
        elif flag.startswith('nz'):
            segments.append(ENTITY_MAP['nz'])
        else:
            segments.append(word)
    return segments

# --- Datasets ---
class TranslationDataset(Dataset):
    def __init__(self, data_files, tokenizer, label_map, max_length=512, masked=False):
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.max_length = max_length
        self.masked = masked
        if isinstance(data_files, Path):
            data_files = [data_files]
        else:
            data_files = list(data_files)
        self.examples = []
        for data_file in data_files:
            with open(data_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    src_lang = item.get('src_lang', '')
                    zh_text = item.get('zh_mt', '')
                    if not zh_text and src_lang == 'chinese':
                        zh_text = item.get('src_text', '')
                    if zh_text and src_lang in label_map:
                        if masked:
                            segments = mask_entities(zh_text)
                            self.examples.append({'segments': segments, 'label': label_map[src_lang]})
                        else:
                            self.examples.append({'text': zh_text, 'label': label_map[src_lang]})
    def __len__(self):
        return len(self.examples)
    def __getitem__(self, idx):
        example = self.examples[idx]
        if self.masked:
            segments = example['segments']
            input_ids = [self.tokenizer.cls_token_id]
            for seg in segments:
                if seg in ENTITY_MAP.values():
                    seg_id = self.tokenizer.convert_tokens_to_ids(seg)
                    if seg_id is None:
                        seg_id = self.tokenizer.unk_token_id
                    input_ids.append(seg_id)
                elif seg == "[MASK]":
                    input_ids.append(self.tokenizer.mask_token_id)
                else:
                    seg_ids = self.tokenizer.encode(seg, add_special_tokens=False)
                    input_ids.extend(seg_ids)
            input_ids.append(self.tokenizer.sep_token_id)
            if len(input_ids) > self.max_length:
                input_ids = input_ids[: self.max_length - 1] + [self.tokenizer.sep_token_id]
            attention_mask = [1] * len(input_ids)
            padding_length = self.max_length - len(input_ids)
            if padding_length > 0:
                input_ids += [self.tokenizer.pad_token_id] * padding_length
                attention_mask += [0] * padding_length
            return {
                'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
                'labels': torch.tensor(example['label'], dtype=torch.long)
            }
        else:
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

def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return accuracy_score(all_labels, all_preds), all_preds, all_labels


# Global config: models and modes
MODELS = [
    # {"name": "DeepSeek", "dir": "models/classifier/DeepSeek_full", "mode": "full"},
    # {"name": "DeepSeek_lora", "dir": "models/classifier/DeepSeek_lora", "mode": "lora"},
    # {"name": "Qwen", "dir": "models/classifier/Qwen_full", "mode": "full"},
    {"name": "Qwen_lora", "dir": "models/classifier/Qwen_lora", "mode": "lora"},
]
DATASETS = ["DeepSeek-V3.2", "Qwen3-Next-80B-A3B-Instruct"]
LANG_ORDER = ["chinese", "english", "french", "german", "japanese", "korean", "russian", "spanish"]

def run_eval(model_dir, backbone_path, batch_size, max_length, masked, dataset_names):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = Path(model_dir) / "best_model.pt"
    label_map_path = Path(model_dir) / "label_map.json"
    tokenizer_path = Path(model_dir)
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        return None
    with open(label_map_path, 'r') as f:
        label_map = json.load(f)
    from classifier_model import SourceLanguageClassifier, LoraClassifier
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    # Choose model type based on mode
    if "lora" in model_dir:
        # Use default LoRA config (match training)
        model = LoraClassifier(
            num_languages=len(label_map),
            model_name=backbone_path,
            hidden_dim=512,
            dropout=0.1,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            cache_dir=None,
            local_dir=None,
            local_files_only=False,
        )
    else:
        model = SourceLanguageClassifier(
            num_languages=len(label_map),
            model_name=backbone_path,
            hidden_dim=512,
            dropout=0.1,
            cache_dir=None,
            local_dir=None,
            local_files_only=False,
        )
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model = model.to(device)
    base_dir = Path(__file__).parent.parent / "new_data/translated"
    results = {}
    for subdir_name in dataset_names:
        subdir = base_dir / subdir_name
        if not subdir.exists():
            print(f"Directory not found: {subdir}")
            continue
        c_files = sorted(list(subdir.glob("c*.jsonl")))
        lang_acc = {}
        all_preds = []
        all_labels = []
        for file_path in c_files:
            dataset = TranslationDataset(file_path, tokenizer, label_map, max_length, masked=masked)
            if len(dataset) == 0:
                continue
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            acc, preds, labels = evaluate(model, dataloader, device)
            # Infer language from filename
            lang = file_path.name.split('_')[1]
            lang_acc[lang] = acc
            all_preds.extend(preds)
            all_labels.extend(labels)
        overall_acc = accuracy_score(all_labels, all_preds) if all_labels else 0.0
        results[subdir_name] = {"lang_acc": lang_acc, "overall": overall_acc}
    return results

def print_result_table(model_name, dataset_results_masked, dataset_results_unmasked):
    print(f"\n# {model_name} classifier accuracy")
    print("|language|deepseek|deepseek (masked)|qwen3|qwen3 (masked)|")
    print("|:---:|:---:|:---:|:---:|:---:|")
    for lang in LANG_ORDER:
        d_acc = dataset_results_unmasked.get("DeepSeek-V3.2", {}).get("lang_acc", {}).get(lang, 0.0)
        d_acc_mask = dataset_results_masked.get("DeepSeek-V3.2", {}).get("lang_acc", {}).get(lang, 0.0)
        q_acc = dataset_results_unmasked.get("Qwen3-Next-80B-A3B-Instruct", {}).get("lang_acc", {}).get(lang, 0.0)
        q_acc_mask = dataset_results_masked.get("Qwen3-Next-80B-A3B-Instruct", {}).get("lang_acc", {}).get(lang, 0.0)
        print(f"|{lang}|{d_acc:.4f}|{d_acc_mask:.4f}|{q_acc:.4f}|{q_acc_mask:.4f}|")
    d_overall = dataset_results_unmasked.get("DeepSeek-V3.2", {}).get("overall", 0.0)
    d_overall_mask = dataset_results_masked.get("DeepSeek-V3.2", {}).get("overall", 0.0)
    q_overall = dataset_results_unmasked.get("Qwen3-Next-80B-A3B-Instruct", {}).get("overall", 0.0)
    q_overall_mask = dataset_results_masked.get("Qwen3-Next-80B-A3B-Instruct", {}).get("overall", 0.0)
    print(f"|overall|{d_overall:.4f}|{d_overall_mask:.4f}|{q_overall:.4f}|{q_overall_mask:.4f}|")

def main():
    parser = argparse.ArgumentParser(description="Evaluate all classifiers on translation data (masked/unmasked)")
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--max_length', type=int, default=512, help='Max sequence length')
    args = parser.parse_args()
    backbone_path = str(Path(__file__).parent.parent / "models/hf_backbones/chinese-roberta-wwm-ext")
    for model_cfg in MODELS:
        print(f"\nEvaluating {model_cfg['name']} ({model_cfg['mode']}) ...")
        # Unmasked
        results_unmasked = run_eval(model_cfg['dir'], backbone_path, args.batch_size, args.max_length, masked=False, dataset_names=DATASETS)
        # Masked
        results_masked = run_eval(model_cfg['dir'], backbone_path, args.batch_size, args.max_length, masked=True, dataset_names=DATASETS)
        print_result_table(model_cfg['name'], results_masked, results_unmasked)

if __name__ == "__main__":
    main()
