"""
Script for sampling XL-Sum data and People's Daily News with specific distribution rules.
Consolidates functionality from 01_sample_xlsum.py, 13_download_original_zh.py, and distribute_german_to_pools.py.

Workflow:
1. en, fr, ru, es, ko, ja:
   - Download full XL-Sum dataset.
   - Split into paragraphs.
   - Save to new_data/raw/{lang}.jsonl.
   - Sample 12,000 items (10k A, 1k B, 1k C).
   - Save to new_data/original/(a|b|c)_{lang}.jsonl.

2. de:
   - Read from new_data/raw/german.jsonl.
   - Sample 12,000 items (10k A, 1k B, 1k C).
   - Save to new_data/original/(a|b|c)_de.jsonl.

3. zh:
   - PDN: Download rows 1,710,000-1,982,265 to new_data/raw/zh_pdn.jsonl.
   - XL-Sum: Download full Chinese dataset to new_data/raw/zh_xlsum.jsonl.
   - Sample 6,000 from PDN and 6,000 from XL-Sum.
   - Combine and distribute (10k A, 1k B, 1k C).
   - Save to new_data/original/(a|b|c)_zh.jsonl.
"""

import argparse
import json
import hashlib
import random
import re
import time
import requests
import gzip
import shutil
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Set, Optional
from datetime import datetime
from datasets import load_dataset

# Constants for PDN
PDN_BASE_URL = "https://huggingface.co/datasets/Papersnake/people_daily_news/resolve/main"
USER_AGENT = "WAYF-original-zh-fetch/1.0"

def hash_text(text: str) -> str:
    """Generate hash for deduplication."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def split_into_sentences(text: str, language: str = "english") -> List[str]:
    """Split text into sentences using language-specific patterns."""
    if language in ["japanese"]:
        sentence_pattern = r'[。！？]+'
        sentences = re.split(sentence_pattern, text)
    elif language in ["korean"]:
        sentence_pattern = r'(?<=[가-힣])[.!?]+'
        sentences = re.split(sentence_pattern, text)
    elif language in ["chinese"]:
        sentence_pattern = r'[。！？；]+'
        sentences = re.split(sentence_pattern, text)
    elif language in ["russian"]:
        sentence_pattern = (
            r'(?<=[.!?]["\'”’\)])\s+(?=[А-ЯЁ])'
            r'|(?<=[.!?])\s+(?=[А-ЯЁ])'
            r'|(?<=[.!?])\s+(?=["\'“‘][А-ЯЁ])'
            r'|(?<=[.!?]["\'”’\)])$'
            r'|(?<=[.!?])$'
        )
        sentences = re.split(sentence_pattern, text)
    else:
        sentence_pattern = (
            r'(?<=[.!?]["\'”’\)])\s+(?=[A-ZÀ-Ü])'
            r'|(?<=[.!?])\s+(?=[A-ZÀ-Ü])'
            r'|(?<=[.!?])\s+(?=["\'“‘][A-ZÀ-Ü])'
            r'|(?<=[.!?]["\'”’\)])$'
            r'|(?<=[.!?])$'
        )
        sentences = re.split(sentence_pattern, text)
    
    return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]

def create_paragraphs(text: str, language: str = "english", 
                     min_sentences: int = 6, max_sentences: int = 8) -> List[str]:
    """Split text into paragraphs of 6-8 sentences."""
    sentences = split_into_sentences(text, language)
    
    if len(sentences) < min_sentences:
        return []
    
    paragraphs = []
    i = 0
    
    while i < len(sentences):
        remaining = len(sentences) - i
        if remaining < min_sentences:
            if paragraphs and remaining > 0:
                paragraphs[-1] = paragraphs[-1] + ' ' + ' '.join(sentences[i:])
            break
        
        para_size = min(max_sentences, max(min_sentences, remaining))
        if remaining < max_sentences and remaining >= min_sentences:
            para_size = remaining
        
        paragraph_sentences = sentences[i:i + para_size]
        
        if language in ["japanese", "chinese"]:
            paragraph = ''.join(paragraph_sentences)
        else:
            paragraph = ' '.join(paragraph_sentences)
        
        paragraphs.append(paragraph)
        i += para_size
    
    return paragraphs

def save_jsonl(items: List[Dict], filepath: Path):
    """Save list of dicts to JSONL file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def append_jsonl(items: List[Dict], filepath: Path):
    """Append list of dicts to JSONL file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'a', encoding='utf-8') as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def download_and_process_xlsum(lang: str, raw_dir: Path, hf_config: str = None):
    """Download XL-Sum data, split into paragraphs, and save to raw_dir."""
    print(f"Processing XL-Sum for {lang}...")
    config_name = hf_config if hf_config else lang
    
    try:
        dataset = load_dataset("csebuetnlp/xlsum", config_name, split="train", trust_remote_code=True)
    except Exception as e:
        print(f"Error loading dataset for {lang}: {e}")
        return

    output_file = raw_dir / f"{lang}.jsonl"
    if output_file.exists():
        print(f"  File {output_file} already exists. Skipping download.")
        return

    print(f"  Saving to {output_file}...")
    count = 0
    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, example in enumerate(dataset):
            text = example.get("text", "")
            if not text or len(text.strip()) < 50:
                continue
            
            paragraphs = create_paragraphs(text, language=lang)
            
            for para_idx, paragraph in enumerate(paragraphs):
                item = {
                    'id': f"{lang}_{idx}_{para_idx}",
                    'src_lang': lang,
                    'src_text': paragraph,
                    'original_id': example.get('id', f'{lang}_{idx}'),
                    'source': f"xlsum_{lang}"
                }
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
                count += 1
                
    print(f"  Saved {count} paragraphs.")

def hf_fetch_rows(dataset: str, config: str, split: str, offset: int, length: int, timeout: int = 20) -> Dict:
    params = {
        "dataset": dataset,
        "config": config,
        "split": split,
        "offset": offset,
        "length": length,
    }
    headers = {"User-Agent": USER_AGENT}
    r = requests.get(HF_ROWS_URL, params=params, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.json()

def download_pdn_zh(raw_dir: Path):
    """Download People's Daily News data (2015-2024) via direct file download."""
    output_file = raw_dir / "zh_pdn.jsonl"
    if output_file.exists():
        print(f"  File {output_file} already exists. Skipping download.")
        return

    print("Downloading People's Daily News (zh) 2015-2024...")
    
    # Clear/Create output file
    with open(output_file, 'w', encoding='utf-8') as f_out:
        pass

    headers = {"User-Agent": USER_AGENT}
    
    for year in range(2015, 2025):
        url = f"{PDN_BASE_URL}/{year}.jsonl.gz"
        print(f"  Downloading {year}.jsonl.gz from {url}...")
        
        try:
            with requests.get(url, headers=headers, stream=True) as r:
                r.raise_for_status()
                
                # Decompress on the fly and write to output
                with gzip.open(r.raw, 'rt', encoding='utf-8') as f_in:
                    with open(output_file, 'a', encoding='utf-8') as f_out:
                        for line in f_in:
                            try:
                                row = json.loads(line)
                                text = row.get("text", "")
                                if not text:
                                    continue
                                
                                # Create standardized item
                                item = {
                                    "id": f"people_daily_news_{year}_{hash_text(text)[:10]}",
                                    "src_lang": "chinese",
                                    "src_text": text,
                                    "source": "people_daily_news",
                                    "original_id": f"{year}_{row.get('date', 'unknown')}",
                                    "date": row.get("date"),
                                    "title": row.get("title"),
                                    "page": row.get("page")
                                }
                                f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
                            except json.JSONDecodeError:
                                continue
                                
        except Exception as e:
            print(f"  Error downloading/processing {year}: {e}")
            continue
            
    print("\n  Done fetching PDN.")

def distribute_and_save(items: List[Dict], lang: str, output_dir: Path):
    """Distribute items to pools A, B, C and save."""
    # Deduplicate
    seen_hashes = set()
    unique_items = []
    for item in items:
        h = hash_text(item['src_text'])
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_items.append(item)
    
    items = unique_items
    
    if len(items) < 12000:
        print(f"  Warning: Only {len(items)} items available for {lang} (needed 12000).")
    
    random.shuffle(items)
    
    pool_a = items[:10000]
    pool_b = items[10000:11000]
    pool_c = items[11000:12000]
    
    print(f"  Distributing {lang}: A={len(pool_a)}, B={len(pool_b)}, C={len(pool_c)}")
    
    save_jsonl(pool_a, output_dir / f"a_{lang}.jsonl")
    save_jsonl(pool_b, output_dir / f"b_{lang}.jsonl")
    save_jsonl(pool_c, output_dir / f"c_{lang}.jsonl")

def process_german(raw_dir: Path, output_dir: Path):
    print("Processing German...")
    input_file = raw_dir / "german.jsonl"
    if not input_file.exists():
        print(f"  Error: {input_file} not found.")
        return
    
    items = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line)
                items.append(item)
            except:
                continue
                
    distribute_and_save(items, "de", output_dir)

def random_cn_segment(text: str, min_len: int = 300, max_len: int = 500) -> Optional[str]:
    """Pick a random substring of length in [min_len, max_len]."""
    s = (text or "").strip()
    if len(s) < min_len:
        return None
    length = random.randint(min_len, max_len)
    if len(s) <= length:
        return s[:length]
    start = random.randint(0, len(s) - length)
    return s[start : start + length]

def process_chinese(raw_dir: Path, output_dir: Path):
    print("Processing Chinese...")
    
    # 1. Prepare PDN
    pdn_file = raw_dir / "zh_pdn.jsonl"
    pdn_pool = []
    if pdn_file.exists():
        with open(pdn_file, 'r', encoding='utf-8') as f:
            all_pdn = [json.loads(line) for line in f]
        
        # Process and deduplicate PDN
        seen_hashes = set()
        unique_pdn = []
        
        # Shuffle all_pdn first to ensure random selection from the full dataset
        # before we start filling our unique buffer
        random.shuffle(all_pdn)
        
        for item in all_pdn:
            # Create a copy to avoid modifying original data
            item_copy = item.copy()
            seg = random_cn_segment(item_copy['src_text'])
            if seg:
                item_copy['src_text'] = seg
                
                h = hash_text(item_copy['src_text'])
                if h not in seen_hashes:
                    seen_hashes.add(h)
                    unique_pdn.append(item_copy)
        
        if len(unique_pdn) >= 6000:
            # We already shuffled all_pdn, but shuffling unique_pdn doesn't hurt
            random.shuffle(unique_pdn)
            pdn_pool = unique_pdn[:6000]
        else:
            print(f"  Warning: Only {len(unique_pdn)} unique PDN items found (needed 6000).")
            pdn_pool = unique_pdn
    else:
        print(f"  Error: {pdn_file} not found.")

    # 2. Prepare XLSum
    xlsum_file = raw_dir / "zh_xlsum.jsonl"
    xlsum_pool = []
    if xlsum_file.exists():
        with open(xlsum_file, 'r', encoding='utf-8') as f:
            all_xlsum = [json.loads(line) for line in f]
            
        # Deduplicate XLSum
        seen_hashes = set()
        unique_xlsum = []
        
        random.shuffle(all_xlsum)
        
        for item in all_xlsum:
            h = hash_text(item['src_text'])
            if h not in seen_hashes:
                seen_hashes.add(h)
                unique_xlsum.append(item)
            
        if len(unique_xlsum) >= 6000:
            random.shuffle(unique_xlsum)
            xlsum_pool = unique_xlsum[:6000]
        else:
            print(f"  Warning: Only {len(unique_xlsum)} unique XLSum items found (needed 6000).")
            xlsum_pool = unique_xlsum
    else:
        print(f"  Error: {xlsum_file} not found.")
        
    # Distribute strictly
    # Pool A: 5000 PDN + 5000 XLSum
    # Pool B: 500 PDN + 500 XLSum
    # Pool C: 500 PDN + 500 XLSum
    
    pdn_a = pdn_pool[:5000]
    pdn_b = pdn_pool[5000:5500]
    pdn_c = pdn_pool[5500:6000]
    
    xlsum_a = xlsum_pool[:5000]
    xlsum_b = xlsum_pool[5000:5500]
    xlsum_c = xlsum_pool[5500:6000]
    
    pool_a = pdn_a + xlsum_a
    pool_b = pdn_b + xlsum_b
    pool_c = pdn_c + xlsum_c
    
    random.shuffle(pool_a)
    random.shuffle(pool_b)
    random.shuffle(pool_c)
    
    print(f"  Distributing zh: A={len(pool_a)} (PDN:{len(pdn_a)}, XL:{len(xlsum_a)}), "
          f"B={len(pool_b)} (PDN:{len(pdn_b)}, XL:{len(xlsum_b)}), "
          f"C={len(pool_c)} (PDN:{len(pdn_c)}, XL:{len(xlsum_c)})")
    
    save_jsonl(pool_a, output_dir / "a_zh.jsonl")
    save_jsonl(pool_b, output_dir / "b_zh.jsonl")
    save_jsonl(pool_c, output_dir / "c_zh.jsonl")

def main():
    parser = argparse.ArgumentParser(description="Consolidated data sampling script")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    raw_dir = Path("new_data/raw")
    output_dir = Path("new_data/original")
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # # 1. XL-Sum Languages
    # xlsum_langs = ["english", "french", "russian", "spanish", "korean", "japanese"]
    
    # for lang in xlsum_langs:
    #     download_and_process_xlsum(lang, raw_dir)
        
    #     # Load back and distribute
    #     items = []
    #     if (raw_dir / f"{lang}.jsonl").exists():
    #         with open(raw_dir / f"{lang}.jsonl", 'r', encoding='utf-8') as f:
    #             for line in f:
    #                 items.append(json.loads(line))
    #         distribute_and_save(items, lang, output_dir)
        
    # # 2. German
    # process_german(raw_dir, output_dir)
    
    # # 3. Chinese
    # # Download PDN
    # download_pdn_zh(raw_dir)
    # # Download XLSum ZH
    # download_and_process_xlsum("chinese", raw_dir, hf_config="chinese_simplified")
    # Rename to zh_xlsum.jsonl for consistency with process_chinese expectation
    if (raw_dir / "chinese.jsonl").exists() and not (raw_dir / "zh_xlsum.jsonl").exists():
        (raw_dir / "chinese.jsonl").rename(raw_dir / "zh_xlsum.jsonl")
        
    process_chinese(raw_dir, output_dir)

if __name__ == "__main__":
    main()
