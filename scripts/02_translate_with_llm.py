"""
Script for translating source texts to Chinese using LLM API.
Implements step 1.3 from project procedure.
"""

import argparse
import json
import hashlib
import re
from pathlib import Path
from typing import Dict, Optional, List
import time
from tqdm import tqdm
import openai
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


class LLMClient:
    """LLM API client using OpenAI SDK with support for multiple API keys."""
    
    def __init__(self, model_name: str, api_keys: List[str], base_url: str, 
                 temperature: float = 0.0, top_p: float = 1.0, max_tokens: int = 2048):
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.base_url = base_url
        
        # Initialize multiple OpenAI clients for different API keys
        self.clients = [
            openai.OpenAI(api_key=key, base_url=f"{base_url}/v1/")
            for key in api_keys
        ]
        self.current_client_idx = 0
        self.client_lock = threading.Lock()
    
    def get_next_client(self):
        """Get next client in round-robin fashion."""
        with self.client_lock:
            client = self.clients[self.current_client_idx]
            self.current_client_idx = (self.current_client_idx + 1) % len(self.clients)
            return client
    
    def translate(self, system_prompt: str, user_prompt: str, max_retries: int = 3) -> str:
        """Call LLM API for translation with retry logic."""
        for attempt in range(max_retries):
            try:
                # Get next client in round-robin
                client = self.get_next_client()
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=self.max_tokens
                )
                
                return response.choices[0].message.content
                
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 5  # 5, 10, 15 seconds
                    print(f"\nAPI call failed (attempt {attempt + 1}/{max_retries}): {e}")
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    raise Exception(f"API call failed after {max_retries} attempts: {e}")


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


def translate_single_item(item_data, llm_client, system_prompt, field, min_cjk_ratio, pool_name, prompt_version):
    """Translate a single item (for parallel processing)."""
    item, line_idx = item_data
    
    src_lang = item.get('src_lang', 'unknown')
    src_text = item.get(field, '')
    item_id = item.get('id', hash_text(src_text))
    
    # Build prompt
    user_prompt = get_user_prompt(src_lang, src_text)
    
    # Call LLM API
    try:
        raw_response = llm_client.translate(system_prompt, user_prompt, max_retries=3)
        
        # Parse JSON response
        try:
            # Remove markdown code blocks if present
            clean_response = raw_response.strip()
            if clean_response.startswith('```'):
                # Extract content between ```json and ```
                match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', clean_response, re.DOTALL)
                if match:
                    clean_response = match.group(1).strip()
            
            response_data = json.loads(clean_response)
            zh_mt = response_data.get('translation', '')
        except json.JSONDecodeError:
            # If JSON parsing still fails, try to extract translation from response
            match = re.search(r'"translation"\s*:\s*"([^"]*)"', raw_response)
            if match:
                zh_mt = match.group(1)
            else:
                return None, f"Cannot parse JSON for item {item_id}"
        
        # Post-process
        zh_mt = post_process_translation(zh_mt)
        
        # Filter by CJK ratio
        if not check_cjk_ratio(zh_mt, min_cjk_ratio):
            return None, f"Insufficient CJK ratio for item {item_id}"
        
        # Build output record
        output_record = {
            'id': item_id,
            'split': item.get('split', 'train'),
            'pool': pool_name,
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
        
        # Copy additional fields from original item
        for key in ['category', 'article_idx', 'paragraph_idx', 'original_id']:
            if key in item:
                output_record[key] = item[key]
        
        return output_record, None
        
    except Exception as e:
        return None, f"Error translating item {item_id}: {e}"


def translate_dataset(input_file: Path, 
                      output_file: Path,
                      llm_client: LLMClient,
                      prompt_version: str,
                      field: str = "src_text",
                      min_cjk_ratio: float = 0.7,
                      pool_name: str = "unknown",
                      num_workers: int = 2):
    """Translate a dataset from source languages to Chinese with parallel processing."""
    system_prompt = get_system_prompt()
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Read all lines
    with open(input_file, 'r', encoding='utf-8') as f_in:
        lines = f_in.readlines()
    
    # Parse items
    items = [(json.loads(line), idx) for idx, line in enumerate(lines)]
    
    # Translate in parallel
    results = [None] * len(items)
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_idx = {
            executor.submit(translate_single_item, item_data, llm_client, system_prompt, 
                          field, min_cjk_ratio, pool_name, prompt_version): item_data[1]
            for item_data in items
        }
        
        # Process completed tasks
        with tqdm(total=len(items), desc="Translating") as pbar:
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    output_record, error = future.result()
                    if output_record:
                        results[idx] = output_record
                    elif error:
                        print(f"\n{error}")
                except Exception as e:
                    print(f"\nUnexpected error for item {idx}: {e}")
                finally:
                    pbar.update(1)
    
    # Write results in original order
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for result in results:
            if result:
                f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
            
            src_lang = item.get('src_lang', 'unknown')
            src_text = item.get(field, '')
            item_id = item.get('id', hash_text(src_text))



def hash_text(text: str) -> str:
    """Generate hash for text."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]


def main():
    parser = argparse.ArgumentParser(description="Translate dataset using LLM API")
    parser.add_argument("--input_file", type=str, required=True, help="Input JSONL file")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSONL file")
    parser.add_argument("--api_keys", type=str, 
                       default="sk-urQ1kd8QaBdMIrnwW8_0Xw,sk-zGRy0CJMC1mpIooqyIstfQ", 
                       help="API keys (comma-separated for parallel processing)")
    parser.add_argument("--base_url", type=str, default="https://llmapi.paratera.com", help="API base URL")
    parser.add_argument("--model_name", type=str, default="DeepSeek-V3.2", help="LLM model name")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=1.0, help="Nucleus sampling parameter")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Max tokens in response")
    parser.add_argument("--prompt_version", type=str, default="v1.0", help="Prompt version identifier")
    parser.add_argument("--field", type=str, default="src_text", help="Source text field")
    parser.add_argument("--min_cjk_ratio", type=float, default=0.7, help="Minimum CJK character ratio")
    parser.add_argument("--pool_name", type=str, default="unknown", help="Pool name for output records")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of parallel workers")
    
    args = parser.parse_args()
    
    # Parse API keys
    api_keys = [key.strip() for key in args.api_keys.split(',')]
    
    # Initialize LLM client
    llm_client = LLMClient(
        model_name=args.model_name,
        api_keys=api_keys,
        base_url=args.base_url,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens
    )
    
    # Translate dataset
    print(f"Translating {args.input_file} to {args.output_file}")
    print(f"Model: {args.model_name}")
    print(f"Base URL: {args.base_url}")
    print(f"API Keys: {len(api_keys)} keys")
    print(f"Workers: {args.num_workers}")
    translate_dataset(
        input_file=Path(args.input_file),
        output_file=Path(args.output_file),
        llm_client=llm_client,
        prompt_version=args.prompt_version,
        field=args.field,
        min_cjk_ratio=args.min_cjk_ratio,
        pool_name=args.pool_name,
        num_workers=args.num_workers
    )
    
    print("\nTranslation complete!")


if __name__ == "__main__":
    main()
