"""
Evaluation script for translationese reduction (probe classifier).
Implements step 4.2 from project procedure.
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm
import numpy as np
import sys

sys.path.append(str(Path(__file__).parent))
from classifier_model import SourceLanguageClassifier


def load_eval_data(data_file: Path):
    """Load evaluation dataset."""
    data = []
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            data.append(item)
    return data


def generate_translations(model, tokenizer, prompts: List[str], 
                         max_length: int = 512, batch_size: int = 8):
    """Generate translations for prompts."""
    translations = []
    
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
        batch_prompts = prompts[i:i+batch_size]
        
        inputs = tokenizer(
            batch_prompts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_length,
                num_beams=1,
                do_sample=False
            )
        
        batch_translations = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        for prompt, full_output in zip(batch_prompts, batch_translations):
            translation = full_output[len(prompt):].strip()
            translations.append(translation)
    
    return translations


def evaluate_with_probe(translations: List[str], 
                       labels: List[str],
                       probe_classifier: SourceLanguageClassifier,
                       probe_tokenizer,
                       label_map: Dict[str, int],
                       device) -> Dict[str, float]:
    """Evaluate translations using probe classifier."""
    
    probe_classifier.eval()
    
    all_logits = []
    all_labels = []
    
    batch_size = 32
    
    for i in tqdm(range(0, len(translations), batch_size), desc="Probing"):
        batch_texts = translations[i:i+batch_size]
        batch_labels = labels[i:i+batch_size]
        
        # Tokenize
        inputs = probe_tokenizer(
            batch_texts,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=512
        ).to(device)
        
        # Get predictions
        with torch.no_grad():
            outputs = probe_classifier(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )
            logits = outputs['logits']
        
        all_logits.append(logits.cpu().numpy())
        
        label_ids = [label_map[lbl] for lbl in batch_labels]
        all_labels.extend(label_ids)
    
    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.array(all_labels)
    
    # Compute metrics
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    from sklearn.preprocessing import label_binarize
    
    predictions = np.argmax(all_logits, axis=1)
    
    accuracy = accuracy_score(all_labels, predictions)
    macro_f1 = f1_score(all_labels, predictions, average='macro')
    
    # AUC
    try:
        n_classes = all_logits.shape[1]
        labels_bin = label_binarize(all_labels, classes=range(n_classes))
        probs = torch.softmax(torch.from_numpy(all_logits), dim=1).numpy()
        auc = roc_auc_score(labels_bin, probs, average='macro', multi_class='ovr')
    except:
        auc = 0.0
    
    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'auc': auc
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate translationese reduction")
    parser.add_argument("--eval_file", type=str, required=True)
    parser.add_argument("--probe_classifier_path", type=str, required=True)
    parser.add_argument("--label_map_path", type=str, required=True)
    parser.add_argument("--sft_model_path", type=str, required=True)
    parser.add_argument("--rl_model_path", type=str, required=True)
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2-7B-Instruct")
    parser.add_argument("--probe_model", type=str, default="hfl/chinese-roberta-wwm-ext")
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--instruction_template", type=str,
                       default="Translate the following {src_lang} text to Chinese:\n\n{src_text}")
    parser.add_argument("--max_length", type=int, default=512)
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load probe classifier
    print("Loading probe classifier...")
    with open(args.label_map_path, 'r') as f:
        label_map = json.load(f)
    
    num_languages = len(label_map)
    probe_classifier = SourceLanguageClassifier(num_languages, args.probe_model)
    probe_classifier.load_state_dict(torch.load(args.probe_classifier_path))
    probe_classifier = probe_classifier.to(device)
    probe_tokenizer = probe_classifier.get_tokenizer()
    
    # Load translation tokenizer
    trans_tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if trans_tokenizer.pad_token is None:
        trans_tokenizer.pad_token = trans_tokenizer.eos_token
    
    # Load evaluation data
    print("Loading evaluation data...")
    eval_data = load_eval_data(Path(args.eval_file))
    print(f"Evaluation samples: {len(eval_data)}")
    
    # Prepare prompts and labels
    prompts = []
    labels = []
    
    for item in eval_data:
        src_lang = item['src_lang']
        src_text = item['src_text']
        prompt = args.instruction_template.format(src_lang=src_lang, src_text=src_text)
        prompts.append(prompt)
        labels.append(src_lang)
    
    results = {}
    
    # Evaluate SFT model
    print("\n=== Evaluating SFT Model ===")
    sft_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    sft_model = PeftModel.from_pretrained(sft_model, args.sft_model_path)
    sft_model.eval()
    
    sft_translations = generate_translations(sft_model, trans_tokenizer, prompts, args.max_length)
    sft_probe_metrics = evaluate_with_probe(sft_translations, labels, probe_classifier, 
                                            probe_tokenizer, label_map, device)
    
    results['sft'] = sft_probe_metrics
    print(f"SFT Probe - Acc: {sft_probe_metrics['accuracy']:.4f}, "
          f"F1: {sft_probe_metrics['macro_f1']:.4f}, "
          f"AUC: {sft_probe_metrics['auc']:.4f}")
    
    # Clean up
    del sft_model
    torch.cuda.empty_cache()
    
    # Evaluate RL model
    print("\n=== Evaluating RL Model ===")
    rl_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    rl_model = PeftModel.from_pretrained(rl_model, args.rl_model_path)
    rl_model.eval()
    
    rl_translations = generate_translations(rl_model, trans_tokenizer, prompts, args.max_length)
    rl_probe_metrics = evaluate_with_probe(rl_translations, labels, probe_classifier,
                                          probe_tokenizer, label_map, device)
    
    results['rl'] = rl_probe_metrics
    print(f"RL Probe - Acc: {rl_probe_metrics['accuracy']:.4f}, "
          f"F1: {rl_probe_metrics['macro_f1']:.4f}, "
          f"AUC: {rl_probe_metrics['auc']:.4f}")
    
    # Compute reduction (lower is better for translationese)
    results['reduction'] = {
        'accuracy': sft_probe_metrics['accuracy'] - rl_probe_metrics['accuracy'],
        'macro_f1': sft_probe_metrics['macro_f1'] - rl_probe_metrics['macro_f1'],
        'auc': sft_probe_metrics['auc'] - rl_probe_metrics['auc']
    }
    
    # Save results
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    print("\n=== Summary ===")
    print(f"Translationese Reduction - Acc: {results['reduction']['accuracy']:.4f}, "
          f"F1: {results['reduction']['macro_f1']:.4f}, "
          f"AUC: {results['reduction']['auc']:.4f}")
    print("(Positive values indicate successful translationese reduction)")


if __name__ == "__main__":
    main()
