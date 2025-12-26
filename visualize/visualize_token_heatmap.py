import sys
import os
import json
import argparse
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer
from captum.attr import LayerIntegratedGradients
from captum.attr import visualization as viz
import jieba.posseg as pseg

# Add scripts to path to import classifier_model
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / "scripts"))

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
        # nr: Person name
        # ns: Place name
        # nt: Organization
        # nz: Other proper noun
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
# ------------------------------------------------------

try:
    from classifier_model import SourceLanguageClassifier
except ImportError as e:
    print(f"Error: Could not import SourceLanguageClassifier from scripts/classifier_model.py. Exception: {e}")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Visualize token heatmap for classifier")
    parser.add_argument("--input_file", type=str, help="Path to input JSON/JSONL file with 'text' and 'label' fields")
    args = parser.parse_args()

    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Paths
    model_path = project_root / "models/classifier/No_conf_full2/best_model.pt"
    label_map_path = project_root / "models/classifier/No_conf_full2/label_map.json"
    tokenizer_path = project_root / "models/classifier/No_conf_full2"
    
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        return

    # Load label map
    print(f"Loading label map from {label_map_path}")
    with open(label_map_path, 'r') as f:
        label_map = json.load(f)
    id2label = {v: k for k, v in label_map.items()}

    # Load Tokenizer
    print(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # Load Model
    print("Loading model...")
    # Initialize model structure
    model = SourceLanguageClassifier(
        num_languages=len(label_map),
        model_name=str(project_root / "models/hf_backbones/chinese-roberta-wwm-ext"),
        hidden_dim=512,
        dropout=0.1
    )
    
    # Load weights
    print(f"Loading weights from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Define forward function for Captum
    # Captum expects the forward function to return the output logits
    def forward_func(input_ids, attention_mask=None):
        # SourceLanguageClassifier.forward returns logits directly
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        return logits

    # Initialize LayerIntegratedGradients
    # We attribute to the embeddings layer of the backbone
    # For BERT/RoBERTa, this is usually model.backbone.embeddings
    lig = LayerIntegratedGradients(forward_func, model.backbone.embeddings)

    # Get samples
    samples = []
    
    if args.input_file:
        input_path = Path(args.input_file)
        if not input_path.exists():
            print(f"Error: Input file {input_path} does not exist.")
            return
        
        print(f"Loading samples from {input_path}")
        with open(input_path, 'r', encoding='utf-8') as f:
            # Check if it looks like JSONL (first char is '{') or JSON list (first char is '[')
            first_char = f.read(1)
            f.seek(0)
            
            if first_char == '[':
                # Assume JSON list
                try:
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            text = item.get('text') or item.get('zh_mt') or item.get('src_text')
                            label = item.get('label') or item.get('src_lang')
                            if text and label:
                                samples.append((text, label))
                    elif isinstance(data, dict):
                         # Single item?
                        text = data.get('text') or data.get('zh_mt') or data.get('src_text')
                        label = data.get('label') or data.get('src_lang')
                        if text and label:
                            samples.append((text, label))
                except json.JSONDecodeError:
                    print("Error: Failed to decode JSON file.")
                    return
            else:
                # Assume JSONL
                for line in f:
                    try:
                        item = json.loads(line)
                        text = item.get('text') or item.get('zh_mt') or item.get('src_text')
                        label = item.get('label') or item.get('src_lang')
                        if text and label:
                            samples.append((text, label))
                    except json.JSONDecodeError:
                        continue

    # Try to load real samples from data if no input file provided
    if not samples and not args.input_file:
        data_dir = project_root / "new_data/translated/DeepSeek-V3.2"
        if data_dir.exists():
            jsonl_files = sorted(list(data_dir.glob("*.jsonl")))
            if jsonl_files:
                data_file = jsonl_files[0]
                print(f"Loading samples from {data_file}")
                with open(data_file, 'r', encoding='utf-8') as f:
                    count = 0
                    for line in f:
                        if count >= 5: break
                        try:
                            item = json.loads(line)
                            src_lang = item.get('src_lang', '')
                            
                            # Logic to get text: try 'zh_mt', if missing and src_lang is chinese, use 'src_text'
                            text = item.get('zh_mt', '')
                            if not text and src_lang == 'chinese':
                                text = item.get('src_text', '')
                                
                            if text and src_lang:
                                samples.append((text, src_lang))
                                count += 1
                        except json.JSONDecodeError:
                            continue
    
    if not samples:
        print("No samples found in data directory, using dummy samples.")
        samples = [
            ("这是一个测试句子。", "chinese"),
            ("This is a test sentence translated to Chinese.", "english"), 
        ]

    print(f"Visualizing {len(samples)} samples...")
    
    vis_data_records = []

    for i, (text, true_label_str) in enumerate(samples):
        print(f"Processing sample {i+1}/{len(samples)}")
        
        if true_label_str not in label_map:
            print(f"  Skipping unknown label: {true_label_str}")
            continue
            
        true_label_idx = label_map[true_label_str]
        
        # Apply masking
        segments = mask_entities(text)
        
        # Manual tokenization to preserve [MASK] ID and special tokens
        input_ids_list = []
        token_to_segment_idx = [] # Map token index to segment index
        
        # 1. Add CLS
        input_ids_list.append(tokenizer.cls_token_id)
        token_to_segment_idx.append(-1) # -1 for CLS
        
        # 2. Process segments
        for i, seg in enumerate(segments):
            if seg in ENTITY_MAP.values():
                seg_id = tokenizer.convert_tokens_to_ids(seg)
                input_ids_list.append(seg_id)
                token_to_segment_idx.append(i)
            elif seg == "[MASK]":
                input_ids_list.append(tokenizer.mask_token_id)
                token_to_segment_idx.append(i)
            else:
                seg_ids = tokenizer.encode(seg, add_special_tokens=False)
                input_ids_list.extend(seg_ids)
                token_to_segment_idx.extend([i] * len(seg_ids))
        
        # 3. Add SEP
        input_ids_list.append(tokenizer.sep_token_id)
        token_to_segment_idx.append(-2) # -2 for SEP
        
        # 4. Truncation (max_length=512)
        if len(input_ids_list) > 512:
            input_ids_list = input_ids_list[:511] + [tokenizer.sep_token_id]
            token_to_segment_idx = token_to_segment_idx[:511] + [-2]
            
        # Convert to tensor (batch size 1)
        input_ids = torch.tensor([input_ids_list], dtype=torch.long).to(device)
        attention_mask = torch.ones_like(input_ids).to(device)
        
        # Predict to get the predicted label
        with torch.no_grad():
            # SourceLanguageClassifier.forward returns logits directly
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            pred_label_idx = torch.argmax(logits, dim=1).item()
            probs = torch.softmax(logits, dim=1)
            pred_prob = probs[0, pred_label_idx].item()
            print(f" Chinese probability: {probs[0, label_map.get('chinese', 0)].item():.4f}, True label probability: {probs[0, true_label_idx].item():.4f}, Predicted: {id2label[pred_label_idx]} ({pred_prob:.4f}), True: {id2label[true_label_idx]}")

        # Compute attributions
        # We attribute to the predicted class
        # input_ids is passed as the first argument to forward_func
        # attention_mask is passed as additional_forward_args
        baseline_idx = torch.full_like(input_ids, tokenizer.pad_token_id).to(device)
        try:
            attributions, delta = lig.attribute(
                inputs=input_ids,
                baselines=baseline_idx,
                additional_forward_args=(attention_mask,),
                target=pred_label_idx,
                return_convergence_delta=True,
                internal_batch_size=1 # Reduce memory usage
            )
        except Exception as e:
            print(f"  Error computing attributions: {e}")
            continue
        
        # attributions shape: [batch_size, seq_len, embedding_dim]
        # Sum across embedding dimension to get token importance
        attributions_sum = attributions.abs().sum(dim=2).squeeze(0)
        attributions_sum = attributions_sum / attributions_sum.max()
        
        # Aggregate to words
        word_attrs = {}
        for idx, attr in zip(token_to_segment_idx, attributions_sum):
            if idx not in word_attrs:
                word_attrs[idx] = 0.0
            word_attrs[idx] += attr.item()
            
        # Construct final lists for visualization
        final_tokens = []
        final_attrs = []
        
        # CLS
        final_tokens.append("[CLS]")
        final_attrs.append(0.0) # Zero out as requested
        
        # Segments
        # Get unique segment indices in order, excluding -1 and -2
        seen_segments = []
        last_idx = -999
        for idx in token_to_segment_idx:
            if idx >= 0 and idx != last_idx:
                seen_segments.append(idx)
                last_idx = idx
        
        for idx in seen_segments:
            final_tokens.append(segments[idx])
            final_attrs.append(word_attrs[idx])
            
        # SEP
        final_tokens.append("[SEP]")
        final_attrs.append(0.0) # Zero out
        
        # Normalize
        final_attrs_np = np.array(final_attrs)
        final_attrs_np = final_attrs_np / np.max(final_attrs_np)
        
        # Create visualization record
        # VisualizationDataRecord(word_attributions, pred_prob, pred_class, true_class, attr_class, attr_score, raw_input_ids, convergence_score)
        record = viz.VisualizationDataRecord(
            final_attrs_np,
            pred_prob,
            id2label[pred_label_idx], # Predicted Class
            id2label[true_label_idx], # True Class
            id2label[pred_label_idx], # Attribution Class (we attributed to predicted)
            final_attrs_np.sum(),    # Total attribution
            final_tokens,
            delta
        )
        vis_data_records.append(record)

    # Generate HTML
    if vis_data_records:
        print("Generating HTML visualization...")
        html = viz.visualize_text(vis_data_records)
        
        output_path = project_root / "visualize/token_heatmap.html"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html.data)
        
        print(f"Visualization saved to {output_path}")
    else:
        print("No records to visualize.")

if __name__ == "__main__":
    main()
