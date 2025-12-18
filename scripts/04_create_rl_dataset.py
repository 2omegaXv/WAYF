"""
Script for creating RL datasets in different formats (GRPO, DPO).
Implements step 1.5 from project procedure.
"""

import argparse
import json
from pathlib import Path
from tqdm import tqdm


def create_grpo_dataset(input_file: Path, output_file: Path, instruction_template: str):
    """Create on-policy RL dataset for GRPO/PPO."""
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        lines = f_in.readlines()
        
        for line in tqdm(lines, desc="Creating GRPO dataset"):
            item = json.loads(line)
            
            src_lang = item.get('src_lang')
            src_text = item.get('src_text')
            item_id = item.get('id')
            zh_ref = item.get('zh_mt')
            
            if not all([src_lang, src_text]):
                continue
            
            prompt = instruction_template.format(src_lang=src_lang, src_text=src_text)
            
            grpo_record = {
                'prompt': prompt,
                'id': item_id,
                'src_lang': src_lang,
                'src_text': src_text,
                'zh_ref': zh_ref
            }
            
            f_out.write(json.dumps(grpo_record, ensure_ascii=False) + '\n')
    
    print(f"GRPO dataset saved to {output_file}")


def create_dpo_dataset(input_file: Path, 
                      chosen_file: Path,
                      rejected_file: Path,
                      output_file: Path,
                      instruction_template: str):
    """Create preference RL dataset for DPO."""
    
    # Load base prompts
    prompts = {}
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            item_id = item.get('id')
            src_lang = item.get('src_lang')
            src_text = item.get('src_text')
            prompts[item_id] = instruction_template.format(src_lang=src_lang, src_text=src_text)
    
    # Load chosen responses
    chosen = {}
    with open(chosen_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            chosen[item['id']] = item['response']
    
    # Load rejected responses
    rejected = {}
    with open(rejected_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            rejected[item['id']] = item['response']
    
    # Create DPO dataset
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for item_id in tqdm(prompts.keys(), desc="Creating DPO dataset"):
            if item_id in chosen and item_id in rejected:
                dpo_record = {
                    'prompt': prompts[item_id],
                    'chosen': chosen[item_id],
                    'rejected': rejected[item_id]
                }
                f_out.write(json.dumps(dpo_record, ensure_ascii=False) + '\n')
    
    print(f"DPO dataset saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Create RL datasets")
    parser.add_argument("--mode", type=str, choices=['grpo', 'dpo'], required=True,
                       help="Dataset format: grpo or dpo")
    parser.add_argument("--input_file", type=str, required=True,
                       help="Input JSONL file")
    parser.add_argument("--output_file", type=str, required=True,
                       help="Output JSONL file")
    parser.add_argument("--instruction_template", type=str,
                       default="Translate the following {src_lang} text to Chinese:\n\n{src_text}",
                       help="Instruction template")
    
    # DPO-specific arguments
    parser.add_argument("--chosen_file", type=str,
                       help="File with chosen responses (for DPO)")
    parser.add_argument("--rejected_file", type=str,
                       help="File with rejected responses (for DPO)")
    
    args = parser.parse_args()
    
    if args.mode == 'grpo':
        create_grpo_dataset(
            input_file=Path(args.input_file),
            output_file=Path(args.output_file),
            instruction_template=args.instruction_template
        )
    elif args.mode == 'dpo':
        if not args.chosen_file or not args.rejected_file:
            raise ValueError("DPO mode requires --chosen_file and --rejected_file")
        
        create_dpo_dataset(
            input_file=Path(args.input_file),
            chosen_file=Path(args.chosen_file),
            rejected_file=Path(args.rejected_file),
            output_file=Path(args.output_file),
            instruction_template=args.instruction_template
        )


if __name__ == "__main__":
    main()
