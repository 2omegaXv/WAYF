"""
Script for RL training using DPO (Direct Preference Optimization).
Implements step 3.7 from project procedure.
"""

import argparse
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import PeftModel
from trl import DPOTrainer, DPOConfig
from datasets import Dataset


def load_dpo_data(data_file: Path):
    """Load DPO preference dataset."""
    data = []
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            data.append({
                'prompt': item['prompt'],
                'chosen': item['chosen'],
                'rejected': item['rejected']
            })
    return Dataset.from_list(data)


def main():
    parser = argparse.ArgumentParser(description="RL training with DPO")
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--valid_file", type=str, required=True)
    parser.add_argument("--sft_model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2-7B-Instruct")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta parameter")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load SFT model as policy
    print("Loading SFT model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(model, args.sft_model_path)
    
    # Load reference model (frozen SFT)
    print("Loading reference model...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    ref_model = PeftModel.from_pretrained(ref_model, args.sft_model_path)
    ref_model.eval()
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = load_dpo_data(Path(args.train_file))
    valid_dataset = load_dpo_data(Path(args.valid_file))
    
    # Configure DPO
    training_args = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        load_best_model_at_end=True,
        fp16=True,
        gradient_checkpointing=True,
        report_to="none",
        seed=args.seed,
        # DPO-specific
        beta=args.beta,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length
    )
    
    # Initialize DPO trainer
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer
    )
    
    # Train
    print("Starting DPO training...")
    trainer.train()
    
    # Save model
    print("Saving model...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    print("DPO training complete!")


if __name__ == "__main__":
    main()
