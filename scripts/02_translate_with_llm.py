"""
Script for translating source texts to Chinese using LLM API.
Implements step 1.3 from project procedure.
"""

import argparse
import json
import hashlib
import re
from pathlib import Path
from typing import Dict, Optional
import time
from tqdm import tqdm


# TODO: Replace with actual API client (OpenAI, Anthropic, etc.)
class LLMClient:
    """Placeholder for LLM API client."""
    
    def __init__(self, model_name: str, temperature: float = 0.0, top_p: float = 1.0, max_tokens: int = 2048):
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
    
    def translate(self, system_prompt: str, user_prompt: str) -> str:
        """Call LLM API for translation."""
        # TODO: Implement actual API call
        # This is a placeholder
        raise NotImplementedError("Implement LLM API call")


def get_system_prompt() -> str:
    """Get the system prompt for translation."""
    return """You are a professional translator. Translate the given text to Chinese following these rules:
1. Output ONLY valid JSON with the format: {"translation": "..."}
2. The translation should be natural Chinese, not literal word-by-word translation
3. Rewrite or remove any URLs, email addresses, and social media handles (@mentions)
4. Do not copy Latin/Cyrillic strings directly; paraphrase them in Chinese
5. Use Chinese punctuation conventions (，。！？「」etc.)
6. Ensure the output is proper Chinese without mixed scripts
"""


def get_user_prompt(src_lang: str, src_text: str) -> str:
    """Build user prompt for translation."""
    return f"""Translate the following {src_lang} text to Chinese:

{src_text}

Remember to output only valid JSON: {{"translation": "..."}}"""


def post_process_translation(zh_text: str) -> str:
    """Apply post-processing filters to translation."""
    # Strip whitespace and quotes
    zh_text = zh_text.strip().strip('"\'')
    
    # Remove URLs
    zh_text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', zh_text)
    
    # Remove email addresses
    zh_text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', zh_text)
    
    # Normalize whitespace
    zh_text = re.sub(r'\s+', ' ', zh_text).strip()
    
    return zh_text


def check_cjk_ratio(text: str, min_ratio: float = 0.7) -> bool:
    """Check if text has sufficient CJK characters."""
    cjk_count = sum(1 for c in text if '\u4e00' <= c <= '\u9fff' or 
                    '\u3400' <= c <= '\u4dbf' or
                    '\u20000' <= c <= '\u2a6df')
    total_count = sum(1 for c in text if not c.isspace())
    
    if total_count == 0:
        return False
    
    return (cjk_count / total_count) >= min_ratio


def hash_response(response: str) -> str:
    """Generate hash of raw API response."""
    return hashlib.sha256(response.encode('utf-8')).hexdigest()


def translate_dataset(input_file: Path, 
                      output_file: Path,
                      llm_client: LLMClient,
                      prompt_version: str,
                      field: str = "text",
                      min_cjk_ratio: float = 0.7):
    """Translate a dataset from source languages to Chinese."""
    system_prompt = get_system_prompt()
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        lines = f_in.readlines()
        
        for line in tqdm(lines, desc="Translating"):
            item = json.loads(line)
            
            src_lang = item.get('src_lang', 'unknown')
            src_text = item.get(field, '')
            item_id = item.get('id', hash_text(src_text))
            
            # Build prompt
            user_prompt = get_user_prompt(src_lang, src_text)
            
            # Call LLM API
            try:
                raw_response = llm_client.translate(system_prompt, user_prompt)
                
                # Parse JSON response
                response_data = json.loads(raw_response)
                zh_mt = response_data.get('translation', '')
                
                # Post-process
                zh_mt = post_process_translation(zh_mt)
                
                # Filter by CJK ratio
                if not check_cjk_ratio(zh_mt, min_cjk_ratio):
                    print(f"\nSkipping item {item_id}: insufficient CJK ratio")
                    continue
                
                # Build output record
                output_record = {
                    'id': item_id,
                    'split': item.get('split', 'train'),
                    'pool': item.get('pool', 'unknown'),
                    'src_lang': src_lang,
                    'src_text': src_text,
                    'zh_mt': zh_mt,
                    'audit': {
                        'model': llm_client.model_name,
                        'temperature': llm_client.temperature,
                        'top_p': llm_client.top_p,
                        'max_tokens': llm_client.max_tokens,
                        'prompt_version': prompt_version,
                        'raw_response_hash': hash_response(raw_response)
                    }
                }
                
                f_out.write(json.dumps(output_record, ensure_ascii=False) + '\n')
                
            except Exception as e:
                print(f"\nError translating item {item_id}: {e}")
                continue
            
            # Rate limiting
            time.sleep(0.1)


def hash_text(text: str) -> str:
    """Generate hash for text."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]


def main():
    parser = argparse.ArgumentParser(description="Translate dataset using LLM API")
    parser.add_argument("--input_file", type=str, required=True, help="Input JSONL file")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSONL file")
    parser.add_argument("--model_name", type=str, default="gpt-4", help="LLM model name")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=1.0, help="Nucleus sampling parameter")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Max tokens in response")
    parser.add_argument("--prompt_version", type=str, default="v1.0", help="Prompt version identifier")
    parser.add_argument("--field", type=str, default="text", help="Source text field")
    parser.add_argument("--min_cjk_ratio", type=float, default=0.7, help="Minimum CJK character ratio")
    
    args = parser.parse_args()
    
    # Initialize LLM client
    llm_client = LLMClient(
        model_name=args.model_name,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens
    )
    
    # Translate dataset
    print(f"Translating {args.input_file} to {args.output_file}")
    translate_dataset(
        input_file=Path(args.input_file),
        output_file=Path(args.output_file),
        llm_client=llm_client,
        prompt_version=args.prompt_version,
        field=args.field,
        min_cjk_ratio=args.min_cjk_ratio
    )
    
    print("\nTranslation complete!")


if __name__ == "__main__":
    main()
