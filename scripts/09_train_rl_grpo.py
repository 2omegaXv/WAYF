"""
Script for RL training using GRPO (Group Relative Policy Optimization).
Implements step 3 from project procedure.
"""

import argparse
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset
import sys

sys.path.append(str(Path(__file__).parent))
from classifier_model import SourceLanguageClassifier


class RewardModel:
    """Reward model combining classifier and quality metrics."""
    
    def __init__(self, 
                 classifier_path: Path,
                 label_map_path: Path,
                 w_cls: float = 1.0,
                 w_qual: float = 0.5,
                 w_kl: float = 0.1):
        """
        Initialize reward model.
        
        Args:
            classifier_path: Path to trained classifier
            label_map_path: Path to label mapping
            w_cls: Weight for classifier reward
            w_qual: Weight for quality reward
            w_kl: Weight for KL penalty
        """
        self.w_cls = w_cls
        self.w_qual = w_qual
        self.w_kl = w_kl
        
        # Load classifier
        with open(label_map_path, 'r') as f:
            label_map = json.load(f)
        
        num_languages = len(label_map)
        self.classifier = SourceLanguageClassifier(num_languages)
        self.classifier.load_state_dict(torch.load(classifier_path))
        self.classifier.eval()
        
        self.tokenizer = self.classifier.get_tokenizer()
        self.label_map = label_map
        self.inv_label_map = {v: k for k, v in label_map.items()}
    
    def compute_classifier_reward(self, zh_output: str, true_src_lang: str) -> float:
        """Compute classifier-based reward."""
        # Tokenize
        inputs = self.tokenizer(
            zh_output,
            return_tensors='pt',
            max_length=512,
            truncation=True,
            padding='max_length'
        )
        
        # Get classifier logits
        with torch.no_grad():
            outputs = self.classifier(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )
            logits = outputs['logits']
        
        # Compute probability for true source language
        probs = torch.softmax(logits, dim=1)[0]
        true_label_idx = self.label_map[true_src_lang]
        p_true = probs[true_label_idx].item()
        
        # Reward is negative log probability (we want to minimize this)
        epsilon = 1e-8
        r_cls = -torch.log(torch.tensor(max(epsilon, p_true))).item()
        
        return r_cls
    
    def compute_quality_reward(self, zh_output: str, zh_ref: str) -> float:
        """Compute quality reward (placeholder for chrF/BLEU)."""
        # TODO: Implement actual chrF or BLEU computation
        # For now, use simple length ratio as proxy
        len_ratio = min(len(zh_output), len(zh_ref)) / max(len(zh_output), len(zh_ref))
        return len_ratio
    
    def compute_total_reward(self, zh_output: str, true_src_lang: str, zh_ref: str = None) -> float:
        """Compute total reward."""
        r_cls = self.compute_classifier_reward(zh_output, true_src_lang)
        
        r_qual = 0.0
        if zh_ref:
            r_qual = self.compute_quality_reward(zh_output, zh_ref)
        
        # Total reward (KL term handled by GRPO trainer)
        total_reward = self.w_cls * (-r_cls) + self.w_qual * r_qual
        
        return total_reward


def load_rl_data(data_file: Path):
    """Load RL training data."""
    data = []
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            data.append(item)
    return Dataset.from_list(data)


def main():
    parser = argparse.ArgumentParser(description="RL training with GRPO")
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--sft_model_path", type=str, required=True)
    parser.add_argument("--classifier_path", type=str, required=True)
    parser.add_argument("--label_map_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2-7B-Instruct")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--w_cls", type=float, default=1.0)
    parser.add_argument("--w_qual", type=float, default=0.5)
    parser.add_argument("--w_kl", type=float, default=0.1)
    parser.add_argument("--num_generations", type=int, default=4, 
                       help="Number of generations per prompt for GRPO")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load SFT model
    print("Loading SFT model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(model, args.sft_model_path)
    
    # Load reference model (frozen SFT model for KL)
    print("Loading reference model...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    ref_model = PeftModel.from_pretrained(ref_model, args.sft_model_path)
    ref_model.eval()
    
    # Load reward model
    print("Loading reward model...")
    reward_model = RewardModel(
        classifier_path=Path(args.classifier_path),
        label_map_path=Path(args.label_map_path),
        w_cls=args.w_cls,
        w_qual=args.w_qual,
        w_kl=args.w_kl
    )
    
    # Load RL dataset
    print("Loading RL dataset...")
    train_dataset = load_rl_data(Path(args.train_file))
    
    # Define reward function
    def reward_fn(prompts, generations, metadata):
        """Compute rewards for generations."""
        rewards = []
        for prompt, gen, meta in zip(prompts, generations, metadata):
            src_lang = meta.get('src_lang', 'unknown')
            zh_ref = meta.get('zh_ref', None)
            reward = reward_model.compute_total_reward(gen, src_lang, zh_ref)
            rewards.append(reward)
        return rewards
    
    # Configure GRPO
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        logging_steps=10,
        save_strategy="epoch",
        fp16=True,
        report_to="none",
        seed=args.seed,
        # GRPO-specific
        num_generation_per_prompt=args.num_generations,
        kl_coef=args.w_kl
    )
    
    # Initialize trainer
    trainer = GRPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        reward_fn=reward_fn
    )
    
    # Train
    print("Starting RL training...")
    trainer.train()
    
    # Save model
    print("Saving model...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    print("RL training complete!")


if __name__ == "__main__":
    main()
