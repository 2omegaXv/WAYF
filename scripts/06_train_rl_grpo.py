"""
Script for RL training using GRPO (Group Relative Policy Optimization).
Implements step 3 from project procedure.
"""

import os
import sys
import json
import yaml
import torch
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOTrainer, GRPOConfig
from peft import LoraConfig, get_peft_model, PeftModel
import wandb

# Add scripts directory to path to import classifier_model
sys.path.append(str(Path(__file__).parent))
from classifier_model import SourceLanguageClassifier, resolve_pretrained_source

def load_config(config_path: str):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def _maybe_set_pad_token(tokenizer) -> None:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


def _format_prompts_like_eval(tokenizer, prompts: List[str]) -> List[str]:
    """Match scripts/13_evaluate_language_distribution.py prompt formatting.

    For chat/instruct models, this is crucial: training on raw text prompts but
    evaluating with a chat template can make the learned adapter look like it
    "does nothing" at evaluation time.
    """
    if getattr(tokenizer, "chat_template", None):
        rendered: List[str] = []
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            rendered.append(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        return rendered
    return prompts

def main():
    parser = argparse.ArgumentParser(description="Train RL model with GRPO")
    parser.add_argument("--config", type=str, default="configs/rl_grpo_config.yaml", help="Path to config file")
    args = parser.parse_args()

    config = load_config(args.config)

    wandb.init(
        project=config['wandb'].get('project', 'WAYF-RL'),
        entity=config['wandb'].get('entity', None),
        name=config['wandb'].get('name', 'grpo-training'),
        mode=config['wandb'].get('mode', 'online'),
        config=config
    )

    print("Loading reward model...")
    classifier_path = config['reward']['classifier_path']
    label_map_path = config['reward']['label_map_path']

    # Resolve paths relative to repo root (prevents silent failures when cwd differs).
    repo_root = Path(__file__).resolve().parent.parent
    classifier_path_p = Path(classifier_path)
    if not classifier_path_p.is_absolute():
        classifier_path_p = (repo_root / classifier_path_p).resolve()
    label_map_path_p = Path(label_map_path)
    if not label_map_path_p.is_absolute():
        label_map_path_p = (repo_root / label_map_path_p).resolve()
    
    # Load label map
    with open(label_map_path_p, 'r') as f:
        label_map = json.load(f)

    chinese_label = None
    for key in ("chinese", "zh", "zh-cn", "zh_cn", "Chinese", "ZH"):
        if key in label_map:
            chinese_label = label_map[key]
            break
    
    if chinese_label is None:
        raise ValueError(f"Could not find 'chinese' or 'zh' in label map: {label_map}")
    
    print(f"Chinese label index: {chinese_label}")

    classifier_backbone = config['reward']['classifier_backbone']

    classifier_tokenizer = AutoTokenizer.from_pretrained(classifier_backbone)
    _maybe_set_pad_token(classifier_tokenizer)
    
    # Initialize model structure
    reward_model = SourceLanguageClassifier(
        model_name=classifier_backbone,
        num_languages=len(label_map),
        hidden_dim=512
    )
    
    if classifier_path_p.exists():
        state_dict = torch.load(str(classifier_path_p), map_location='cpu')
        reward_model.load_state_dict(state_dict)
    else:
        raise FileNotFoundError(
            f"Classifier checkpoint not found: {classifier_path_p} (cwd={Path.cwd()}). "
            "Refusing to continue with random reward weights."
        )
    
    reward_model.eval()
    reward_model.to("cuda" if torch.cuda.is_available() else "cpu")

    def reward_function(prompts, completions, **kwargs):
        
        rewards = []
        
        inputs = classifier_tokenizer(
            completions,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(reward_model.backbone.device)
        
        with torch.no_grad():
            logits = reward_model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )
            
            for i in range(len(completions)):
                current_logits = logits[i]
                score_chinese = current_logits[chinese_label]
                
                mask = torch.ones_like(current_logits, dtype=torch.bool)
                mask[chinese_label] = False
                score_others = torch.max(current_logits[mask])
                
                r_cls = score_chinese - score_others
                
                final_reward = config['reward']['w_cls'] * r_cls.item()
                rewards.append(final_reward)
                
        return rewards

    # 3. Load Policy Model
    print("Loading policy model...")
    model_name = config['model']['base_name']
    sft_checkpoint = config['model'].get('sft_checkpoint')

    policy_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    _maybe_set_pad_token(policy_tokenizer)
    
    # Load base model
    use_fp16 = bool(config.get('training', {}).get('fp16', False))
    if use_fp16:
        policy_dtype = torch.float16
    else:
        policy_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    policy_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=policy_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    
    if sft_checkpoint and os.path.exists(sft_checkpoint):
        print(f"Loading SFT adapter from {sft_checkpoint}")
        policy_model = PeftModel.from_pretrained(policy_model, sft_checkpoint)
        policy_model = policy_model.merge_and_unload()
    
    # Configure LoRA for RL
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 4. Load Dataset
    print("Loading dataset...")
    from datasets import load_dataset
    
    train_file = config['paths']['train_file']
    dataset = load_dataset("json", data_files=train_file, split="train")

    # IMPORTANT: match evaluation prompt formatting (chat template) during training.
    # We rewrite the dataset prompt strings into their chat-rendered form if applicable.
    if "prompt" not in dataset.column_names:
        raise ValueError(
            f"GRPO dataset is missing required 'prompt' field. Found columns: {dataset.column_names}"
        )

    def _map_prompt(batch):
        batch_prompts = batch["prompt"]
        batch["prompt"] = _format_prompts_like_eval(policy_tokenizer, batch_prompts)
        return batch

    dataset = dataset.map(_map_prompt, batched=True, desc="Formatting prompts (chat template)")
    
    # 5. Configure Trainer
    training_args = GRPOConfig(
        output_dir=config['paths']['output_dir'],
        learning_rate=float(config['training']['learning_rate']),
        per_device_train_batch_size=config['training']['batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        num_generations=config['grpo']['num_generations'],
        max_prompt_length=512,
        max_completion_length=512,
        num_train_epochs=config['training']['epochs'],
        logging_steps=10,
        save_steps=100,
        bf16=(policy_dtype == torch.bfloat16),
        beta=config['reward'].get('w_kl', 0.1), # KL penalty weight
        report_to="wandb",
        run_name=config['wandb'].get('name', 'grpo-training')
    )

    trainer = GRPOTrainer(
        model=policy_model,
        reward_funcs=reward_function,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
    )

    # 6. Train
    print("Starting training...")
    trainer.train()
    
    # Save
    trainer.save_model(config['paths']['output_dir'])
    print("Training complete.")

if __name__ == "__main__":
    main()
