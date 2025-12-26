"""
Script for creating RL datasets in different formats (GRPO, DPO).
Implements step 1.5 from project procedure.
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm
import random

def create_grpo_dataset(input_path: str, output_file: Path, instruction_template: str):
    """Create on-policy RL dataset for GRPO/PPO."""
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    import glob
    # Check if input_path contains wildcards
    if '*' in input_path or '?' in input_path or '[' in input_path:
        input_files = [Path(p) for p in glob.glob(input_path)]
    else:
        input_files = [Path(input_path)]
        
    if not input_files:
        print(f"No files found matching {input_path}")
        return

    print(f"Processing {len(input_files)} files for GRPO...")
    count = 0
    
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for input_file in input_files:
            print(f"Reading {input_file}...")
            with open(input_file, 'r', encoding='utf-8') as f_in:
                for line in tqdm(f_in, desc=f"Processing {input_file.name}"):
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    
                    src_lang = item.get('src_lang')
                    src_text = item.get('src_text')
                    item_id = item.get('id')
                    
                    if not all([src_lang, src_text]):
                        continue
                    
                    prompt = instruction_template.format(src_lang=src_lang, src_text=src_text)
                    
                    grpo_record = {
                        'prompt': prompt,
                        'id': item_id,
                        'src_lang': src_lang,
                        'src_text': src_text,
                    }
                    
                    # Include reference if available (optional)
                    if 'zh_ref' in item:
                        grpo_record['zh_ref'] = item['zh_ref']
                    
                    f_out.write(json.dumps(grpo_record, ensure_ascii=False) + '\n')
                    count += 1
    
    print(f"GRPO dataset saved to {output_file} ({count} samples)")


def create_dpo_dataset(input_file: Path, 
                      chosen_file: Optional[Path],
                      rejected_file: Optional[Path],
                      candidates_file: Optional[Path],
                      output_file: Path,
                      instruction_template: str):
    """Create preference RL dataset for DPO."""
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    dpo_records = []

    # Mode 1: From separate chosen/rejected files
    if chosen_file and rejected_file:
        print("Creating DPO dataset from separate chosen/rejected files...")
        
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
        
        for item_id, prompt in prompts.items():
            if item_id in chosen and item_id in rejected:
                dpo_records.append({
                    'prompt': prompt,
                    'chosen': chosen[item_id],
                    'rejected': rejected[item_id]
                })

    # Mode 2: From candidates file (containing list of responses with scores)
    elif candidates_file:
        print("Creating DPO dataset from candidates file...")
        with open(candidates_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Processing candidates"):
                item = json.loads(line)
                
                # Expecting item to have 'prompt' (or src info) and 'responses' list with 'score'
                if 'responses' not in item or len(item['responses']) < 2:
                    continue
                
                # Construct prompt if not present
                if 'prompt' in item:
                    prompt = item['prompt']
                else:
                    src_lang = item.get('src_lang')
                    src_text = item.get('src_text')
                    if not src_lang or not src_text:
                        continue
                    prompt = instruction_template.format(src_lang=src_lang, src_text=src_text)

                # Sort responses by score (descending)
                # Assuming response item has 'text' and 'score'
                responses = sorted(item['responses'], key=lambda x: x.get('score', 0), reverse=True)
                
                # Simple strategy: Best vs Worst
                chosen_resp = responses[0]['text']
                rejected_resp = responses[-1]['text']
                
                # Optional: Ensure score difference is significant
                if responses[0].get('score', 0) > responses[-1].get('score', 0):
                    dpo_records.append({
                        'prompt': prompt,
                        'chosen': chosen_resp,
                        'rejected': rejected_resp,
                        'chosen_score': responses[0].get('score'),
                        'rejected_score': responses[-1].get('score')
                    })
    
    else:
        raise ValueError("Must provide either (chosen_file + rejected_file) or candidates_file")

    # Save
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for record in dpo_records:
            f_out.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"DPO dataset saved to {output_file} ({len(dpo_records)} samples)")


def main():
    parser = argparse.ArgumentParser(description="Create RL datasets")
    parser.add_argument("--mode", type=str, choices=['grpo', 'dpo'], required=True,
                       help="Dataset format: grpo or dpo")
    parser.add_argument("--input_file", type=str, required=True,
                       help="Input JSONL file (Pool B for GRPO, or Prompts for DPO)")
    parser.add_argument("--output_file", type=str, required=True,
                       help="Output JSONL file")
    parser.add_argument("--instruction_template", type=str,
                       default="Translate the following {src_lang} text to Chinese:\n\n{src_text}",
                       help="Instruction template")
    
    # DPO-specific arguments
    parser.add_argument("--chosen_file", type=str,
                       help="File with chosen responses (for DPO mode 1)")
    parser.add_argument("--rejected_file", type=str,
                       help="File with rejected responses (for DPO mode 1)")
    parser.add_argument("--candidates_file", type=str,
                       help="File with scored candidates (for DPO mode 2)")
    
    args = parser.parse_args()
    
    if args.mode == 'grpo':
        create_grpo_dataset(
            input_path=args.input_file,
            output_file=Path(args.output_file),
            instruction_template=args.instruction_template
        )
    elif args.mode == 'dpo':
        create_dpo_dataset(
            input_file=Path(args.input_file),
            chosen_file=Path(args.chosen_file) if args.chosen_file else None,
            rejected_file=Path(args.rejected_file) if args.rejected_file else None,
            candidates_file=Path(args.candidates_file) if args.candidates_file else None,
            output_file=Path(args.output_file),
            instruction_template=args.instruction_template
        )


if __name__ == "__main__":
    main()
