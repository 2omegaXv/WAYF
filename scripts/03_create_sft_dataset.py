"""
Script for creating SFT (Supervised Fine-Tuning) reference dataset.
Implements step 1.4 from project procedure.
"""

import argparse
import json
from pathlib import Path
from typing import Optional
from tqdm import tqdm


def get_translation_prompt(src_lang: str, src_text: str, instruction_template: str) -> str:
    """Build translation prompt for SFT."""
    return instruction_template.format(src_lang=src_lang, src_text=src_text)


def create_sft_dataset(input_file: Path,
                      output_file: Path,
                      instruction_template: str):
    """Create SFT dataset from translated data."""
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        lines = f_in.readlines()
        
        for line in tqdm(lines, desc="Creating SFT dataset"):
            item = json.loads(line)
            
            # Extract fields
            item_id = item.get('id')
            src_lang = item.get('src_lang')
            src_text = item.get('src_text')
            zh_ref = item.get('zh_mt')  # Using zh_mt as reference
            
            if not all([src_lang, src_text, zh_ref]):
                continue
            
            # Build SFT record
            prompt = get_translation_prompt(src_lang, src_text, instruction_template)
            
            sft_record = {
                'id': item_id,
                'prompt': prompt,
                'response': zh_ref,
                'metadata': {
                    'src_lang': src_lang,
                    'src_text': src_text
                }
            }
            
            f_out.write(json.dumps(sft_record, ensure_ascii=False) + '\n')
    
    print(f"\nSFT dataset saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Create SFT reference dataset")
    parser.add_argument("--input_file", type=str, required=True, 
                       help="Input JSONL file with translations")
    parser.add_argument("--output_file", type=str, required=True,
                       help="Output JSONL file for SFT")
    parser.add_argument("--instruction_template", type=str,
                       default="Translate the following {src_lang} text to Chinese:\n\n{src_text}",
                       help="Instruction template for prompts")
    
    args = parser.parse_args()
    
    create_sft_dataset(
        input_file=Path(args.input_file),
        output_file=Path(args.output_file),
        instruction_template=args.instruction_template
    )


if __name__ == "__main__":
    main()
