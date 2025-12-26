import sys
import os
import json
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import yaml
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer
import jieba.posseg as pseg
import re

# Add scripts to path to import classifier_model
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / "scripts"))

try:
    from classifier_model import SourceLanguageClassifier, LoraClassifier
except ImportError as e:
    print(f"Error: Could not import SourceLanguageClassifier/LoraClassifier from scripts/classifier_model.py. Exception: {e}")
    sys.exit(1)

# --- Copied from test_classifier_masked_entities.py ---
ENTITY_MAP = {
    'nr': '[unused1]',  # Person Name (人名) -> [unused1]
    'ns': '[unused2]',  # Place Name (地名) -> [unused2]
    'nt': '[unused3]',  # Organization (机构) -> [unused3]
    'nz': '[unused4]',  # Other Proper Noun (其他专名) -> [unused4]
}

def mask_entities(text: str):
    """
    Mask person names (nr), place names (ns), organization names (nt), 
    and other proper nouns (nz) with specific tokens.
    Returns a list of segments.
    """
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

def get_embeddings(model, tokenizer, texts, device, batch_size=32):
    model.eval()
    all_embeddings = []
    
    # Process in batches
    for i in tqdm(range(0, len(texts), batch_size), desc="Extracting embeddings"):
        batch_texts = texts[i:i+batch_size]
        
        # Tokenize with masking logic
        batch_input_ids = []
        batch_attention_mask = []
        
        for text in batch_texts:
            segments = mask_entities(text)
            
            input_ids = []
            input_ids.append(tokenizer.cls_token_id)
            
            for seg in segments:
                if seg in ENTITY_MAP.values():
                    seg_id = tokenizer.convert_tokens_to_ids(seg)
                    input_ids.append(seg_id)
                elif seg == "[MASK]":
                    input_ids.append(tokenizer.mask_token_id)
                else:
                    seg_ids = tokenizer.encode(seg, add_special_tokens=False)
                    input_ids.extend(seg_ids)
            
            input_ids.append(tokenizer.sep_token_id)
            
            # Truncate
            if len(input_ids) > 512:
                input_ids = input_ids[:511] + [tokenizer.sep_token_id]
                
            # Pad
            padding_length = 512 - len(input_ids)
            attention_mask = [1] * len(input_ids) + [0] * padding_length
            input_ids = input_ids + [tokenizer.pad_token_id] * padding_length
            
            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            
        input_ids_tensor = torch.tensor(batch_input_ids, dtype=torch.long).to(device)
        attention_mask_tensor = torch.tensor(batch_attention_mask, dtype=torch.long).to(device)
        
        with torch.no_grad():
            # We want the [CLS] embedding from the backbone
            outputs = model.backbone(input_ids=input_ids_tensor, attention_mask=attention_mask_tensor)
            # last_hidden_state: [batch_size, seq_len, hidden_size]
            # CLS token is at index 0
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(cls_embeddings)
            
    return np.vstack(all_embeddings)

def main():
    parser = argparse.ArgumentParser(description="Visualize UMAP of [CLS] embeddings")
    parser.add_argument("--use_baseline", action="store_true", help="Use baseline Chinese BERT without fine-tuning")
    args = parser.parse_args()

    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load classifier config (optional)
    config_path = project_root / "configs/classifier_config.yaml"
    config = {}
    if config_path.exists():
        print(f"Loading config from {config_path}")
        with open(config_path, 'r', encoding='utf-8') as cf:
            config = yaml.safe_load(cf) or {}
    else:
        print(f"Config not found at {config_path}, using defaults")

    translation_model = config.get('experiment', {}).get('translation_model', 'DeepSeek-V3.2')

    # Determine model directory: prefer configured output_dir, else use models/classifier/{translation_model}
    output_dir_cfg = config.get('paths', {}).get('output_dir')
    if output_dir_cfg:
        model_dir = project_root / output_dir_cfg
    else:
        model_dir = project_root / f"models/classifier/{translation_model}"

    model_path = model_dir / "best_model.pt"
    label_map_path = model_dir / "label_map.json"
    tokenizer_path = model_dir

    # Fallback to LoRA model dir if full model not found
    if not model_path.exists():
        alt = project_root / "models/classifier/DeepSeek_lora/best_model.pt"
        if alt.exists():
            model_path = alt
            label_map_path = alt.parent / "label_map.json"
            tokenizer_path = alt.parent
        else:
            print(f"Model not found at {model_path}")
            return

    # Load label map
    print(f"Loading label map from {label_map_path}")
    with open(label_map_path, 'r') as f:
        label_map = json.load(f)
    
    # Load Tokenizer
    print(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # Load Model
    print("Loading model...")
    if args.use_baseline:
        print("Using baseline Chinese BERT (no fine-tuning)")
        model = SourceLanguageClassifier(
            num_languages=len(label_map),
            model_name=str(project_root / "models/hf_backbones/chinese-roberta-wwm-ext"),
            hidden_dim=512,
            dropout=0.1
        )
        # Do NOT load state dict
    elif "lora" in str(model_path):
        print("Detected LoRA model, using LoraClassifier")
        model = LoraClassifier(
            num_languages=len(label_map),
            model_name=str(project_root / "models/hf_backbones/chinese-roberta-wwm-ext"),
            hidden_dim=512,
            dropout=0.1
        )
        print(f"Loading weights from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("Using standard SourceLanguageClassifier")
        model = SourceLanguageClassifier(
            num_languages=len(label_map),
            model_name=str(project_root / "models/hf_backbones/chinese-roberta-wwm-ext"),
            hidden_dim=512,
            dropout=0.1
        )
        print(f"Loading weights from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    
    model.to(device)
    
    # Load Data files from config paths.test_file if provided, else fallback to new_data/translated/{translation_model}
    files = []
    cfg_test_files = config.get('paths', {}).get('test_file', []) if config else []
    if cfg_test_files:
        for p in cfg_test_files:
            ppath = Path(p)
            if not ppath.is_absolute():
                ppath = project_root / ppath
            if ppath.exists():
                if ppath.is_file():
                    files.append(ppath)
                elif ppath.is_dir():
                    files.extend(sorted(ppath.glob("*.jsonl")))
            else:
                # Try glob relative to project root
                files.extend(sorted(list(project_root.glob(str(p)))))
    else:
        data_dir = project_root / f"new_data/translated/{translation_model}"
        if data_dir.exists():
            files = sorted(list(data_dir.glob("c_*_zh.jsonl")))

    # Deduplicate and sort
    files = sorted(list(dict.fromkeys(files)))
    
    texts = []
    labels = []
    
    if not files:
        print("No test files found. Please check config paths.test_file or new_data/translated folder.")
        return

    print(f"Loading data from {len(files)} files...")
    for file_path in files:
        print(f"  Reading {file_path.name}...")
        # Build label from filename: strip leading 'c_' and trailing '_zh*'
        stem = file_path.stem
        label_for_file = re.sub(r'^c_', '', stem)
        label_for_file = re.sub(r'_zh.*$', '', label_for_file)

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    src_lang = item.get('src_lang', '')

                    # Logic to get text
                    text = item.get('zh_mt', '')
                    if not text and src_lang == 'chinese':
                        text = item.get('src_text', '')

                    if text:
                        texts.append(text)
                        labels.append(label_for_file)
                except json.JSONDecodeError:
                    continue
    
    print(f"Total samples: {len(texts)}")
    
    # # Limit samples for visualization if too many (optional, but good for speed)
    # # Let's keep all for now unless it's huge.
    # if len(texts) > 10000:
    #     print("Subsampling to 10000 samples for visualization...")
    #     indices = np.random.choice(len(texts), 10000, replace=False)
    #     texts = [texts[i] for i in indices]
    #     labels = [labels[i] for i in indices]

    # Extract Embeddings
    embeddings = get_embeddings(model, tokenizer, texts, device)
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Run UMAP
    print("Running UMAP...")
    reducer = umap.UMAP()
    embedding_2d = reducer.fit_transform(embeddings)
    
    # Plot
    print("Plotting...")
    plt.figure(figsize=(12, 10))
    
    unique_labels = sorted(list(set(labels)))
    n_labels = len(unique_labels)
    # Choose palettes to maximize distinctiveness depending on number of labels
    if n_labels <= 10:
        palette = sns.color_palette("tab10", n_labels)
    else:
        palette = sns.color_palette("tab20", n_labels)
    label_to_color = dict(zip(unique_labels, palette))

    for label in unique_labels:
        indices = [i for i, l in enumerate(labels) if l == label]
        plt.scatter(
            embedding_2d[indices, 0],
            embedding_2d[indices, 1],
            c=[label_to_color[label]],
            label=label,
            alpha=0.8,
            s=18,
            edgecolors='none'
        )
        
    plt.title("UMAP projection of [CLS] embeddings by Source Language", fontsize=16)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    if args.use_baseline:
        output_file = project_root / "visualize/umap_plot_baseline.png"
    else:
        output_file = project_root / f"visualize/umap_plot_large.png"
    plt.savefig(output_file, dpi=300)
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    main()
