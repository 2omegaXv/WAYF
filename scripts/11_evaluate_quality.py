"""
Evaluation script for translation quality metrics.
Implements step 4.1 from project procedure.
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm
import sacrebleu
from datasets import load_metric


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
                do_sample=False,
                temperature=1.0
            )
        
        batch_translations = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # Extract only the generated part (after prompt)
        for prompt, full_output in zip(batch_prompts, batch_translations):
            translation = full_output[len(prompt):].strip()
            translations.append(translation)
    
    return translations


def compute_chrf(predictions: List[str], references: List[str]) -> float:
    """Compute chrF score."""
    chrf = sacrebleu.corpus_chrf(predictions, [references])
    return chrf.score


def compute_bleu(predictions: List[str], references: List[str]) -> float:
    """Compute BLEU score."""
    bleu = sacrebleu.corpus_bleu(predictions, [references])
    return bleu.score


def compute_comet(predictions: List[str], references: List[str], sources: List[str]) -> float:
    """Compute COMET score (requires comet-ml)."""
    try:
        from comet import download_model, load_from_checkpoint
        
        model_path = download_model("Unbabel/wmt22-comet-da")
        model = load_from_checkpoint(model_path)
        
        data = [
            {"src": src, "mt": pred, "ref": ref}
            for src, pred, ref in zip(sources, predictions, references)
        ]
        
        results = model.predict(data, batch_size=8, gpus=1 if torch.cuda.is_available() else 0)
        return results['system_score']
    except ImportError:
        print("COMET not installed. Skipping COMET evaluation.")
        return 0.0


def evaluate_model(model_path: str, 
                  base_model: str,
                  eval_data: List[Dict],
                  tokenizer,
                  instruction_template: str,
                  max_length: int = 512) -> Dict[str, float]:
    """Evaluate a model on translation quality."""
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    if Path(model_path).exists():
        model = PeftModel.from_pretrained(model, model_path)
    
    model.eval()
    
    # Prepare prompts and references
    prompts = []
    references = []
    sources = []
    
    for item in eval_data:
        src_lang = item['src_lang']
        src_text = item['src_text']
        zh_ref = item.get('zh_mt', '')
        
        prompt = instruction_template.format(src_lang=src_lang, src_text=src_text)
        prompts.append(prompt)
        references.append(zh_ref)
        sources.append(src_text)
    
    # Generate translations
    predictions = generate_translations(model, tokenizer, prompts, max_length)
    
    # Compute metrics
    chrf_score = compute_chrf(predictions, references)
    bleu_score = compute_bleu(predictions, references)
    comet_score = compute_comet(predictions, references, sources)
    
    return {
        'chrf': chrf_score,
        'bleu': bleu_score,
        'comet': comet_score
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate translation quality")
    parser.add_argument("--eval_file", type=str, required=True)
    parser.add_argument("--sft_model_path", type=str, required=True)
    parser.add_argument("--rl_model_path", type=str, required=True)
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2-7B-Instruct")
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--instruction_template", type=str,
                       default="Translate the following {src_lang} text to Chinese:\n\n{src_text}")
    parser.add_argument("--max_length", type=int, default=512)
    
    args = parser.parse_args()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load evaluation data
    print("Loading evaluation data...")
    eval_data = load_eval_data(Path(args.eval_file))
    print(f"Evaluation samples: {len(eval_data)}")
    
    results = {}
    
    # Evaluate SFT model
    print("\n=== Evaluating SFT Model ===")
    sft_metrics = evaluate_model(
        model_path=args.sft_model_path,
        base_model=args.base_model,
        eval_data=eval_data,
        tokenizer=tokenizer,
        instruction_template=args.instruction_template,
        max_length=args.max_length
    )
    results['sft'] = sft_metrics
    print(f"SFT - chrF: {sft_metrics['chrf']:.2f}, BLEU: {sft_metrics['bleu']:.2f}, COMET: {sft_metrics['comet']:.4f}")
    
    # Evaluate RL model
    print("\n=== Evaluating RL Model ===")
    rl_metrics = evaluate_model(
        model_path=args.rl_model_path,
        base_model=args.base_model,
        eval_data=eval_data,
        tokenizer=tokenizer,
        instruction_template=args.instruction_template,
        max_length=args.max_length
    )
    results['rl'] = rl_metrics
    print(f"RL - chrF: {rl_metrics['chrf']:.2f}, BLEU: {rl_metrics['bleu']:.2f}, COMET: {rl_metrics['comet']:.4f}")
    
    # Compute improvements
    results['improvement'] = {
        'chrf': rl_metrics['chrf'] - sft_metrics['chrf'],
        'bleu': rl_metrics['bleu'] - sft_metrics['bleu'],
        'comet': rl_metrics['comet'] - sft_metrics['comet']
    }
    
    # Save results
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    print("\n=== Summary ===")
    print(f"Improvement - chrF: {results['improvement']['chrf']:.2f}, "
          f"BLEU: {results['improvement']['bleu']:.2f}, "
          f"COMET: {results['improvement']['comet']:.4f}")


if __name__ == "__main__":
    main()
