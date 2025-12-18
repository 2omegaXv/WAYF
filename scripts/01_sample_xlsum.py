"""
Script for sampling XL-Sum data with language selection, length matching, and deduplication.
Implements step 1.1 and 1.2 from project procedure.
Splits text articles into 5-7 sentence paragraphs.
"""

import argparse
import json
import hashlib
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Set
import random
import re
from datasets import load_dataset


def hash_text(text: str) -> str:
    """Generate hash for deduplication."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def get_length_bin(text: str, bin_size: int = 50) -> int:
    """Bin text by character length."""
    return len(text) // bin_size


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using regex patterns."""
    # Handle common sentence ending patterns
    # This pattern handles periods, question marks, exclamation marks followed by space or end
    sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])$'
    sentences = re.split(sentence_pattern, text)
    
    # Clean up and filter empty sentences
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return sentences


def deduplicate_data(data: List[Dict], field: str = "text") -> List[Dict]:
    """Remove exact duplicates based on text hash."""
    seen_hashes: Set[str] = set()
    deduplicated = []
    
    for item in data:
        text = item.get(field, "")
        text_hash = hash_text(text)
        
        if text_hash not in seen_hashes:
            seen_hashes.add(text_hash)
            deduplicated.append(item)
    
    return deduplicated


def sample_balanced_paragraphs(data_by_lang: Dict[str, List[Dict]], 
                               paragraphs_per_lang: int,
                               field: str = "text") -> Dict[str, List[Dict]]:
    """
    Sample equal number of paragraphs per language with length matching.
    
    Args:
        data_by_lang: Dictionary of language to paragraph items
        paragraphs_per_lang: Target number of paragraphs per language
        field: Text field to use for length binning
    
    Returns:
        Dictionary with balanced samples per language
    """
    sampled_data = {}
    
    for lang, items in data_by_lang.items():
        print(f"\nSampling {lang}:")
        print(f"  Available paragraphs: {len(items)}")
        
        # Deduplicate first
        items = deduplicate_data(items, field)
        print(f"  After deduplication: {len(items)}")
        
        # Check if we have enough paragraphs
        if len(items) < paragraphs_per_lang:
            print(f"  Warning: Only {len(items)} paragraphs available (requested {paragraphs_per_lang})")
            sampled_data[lang] = items
            continue
        
        # Group by length bins for balanced sampling
        length_bins = defaultdict(list)
        for item in items:
            text = item.get(field, "")
            bin_idx = get_length_bin(text, bin_size=100)  # Use larger bins for paragraphs
            length_bins[bin_idx].append(item)
        
        # Sample evenly across bins
        samples = []
        bins = sorted(length_bins.keys())
        
        if bins:
            samples_per_bin = max(1, paragraphs_per_lang // len(bins))
            
            for bin_idx in bins:
                bin_items = length_bins[bin_idx]
                n_samples = min(len(bin_items), samples_per_bin)
                samples.extend(random.sample(bin_items, n_samples))
            
            # If we need more samples, take randomly from remaining
            if len(samples) < paragraphs_per_lang:
                remaining = [item for item in items if item not in samples]
                additional = min(len(remaining), paragraphs_per_lang - len(samples))
                if additional > 0:
                    samples.extend(random.sample(remaining, additional))
        
        sampled_data[lang] = samples[:paragraphs_per_lang]
        print(f"  Sampled: {len(sampled_data[lang])} paragraphs")
    
    return sampled_data
    """
        languages: List of language names (e.g., ['english', 'french'])
        field: Field to use ('text' for full article)
        min_sentences: Minimum sentences per paragraph
        max_sentences: Maximum sentences per paragraph
        split: Dataset split to use ('train', 'validation', 'test')
    
    Returns:
        Dictionary mapping language to list of paragraph items
    """
    data_by_lang = defaultdict(list)
    
    for lang in languages:
        print(f"Loading {lang}...")
        
        try:
            # Load dataset for this language
            dataset = load_dataset("csebuetnlp/xlsum", lang, split=split, trust_remote_code=True)
            
            paragraph_count = 0
            
            for idx, example in enumerate(dataset):
                article_text = example.get(field, "")
                
                if not article_text or len(article_text.strip()) < 50:
                    continue
                
                # Split article into paragraphs
                paragraphs = create_paragraphs(article_text, min_sentences, max_sentences)
                
                # Create separate items for each paragraph
                for para_idx, paragraph in enumerate(paragraphs):
                    item = {
                        'id': f"{lang}_{idx}_{para_idx}",
                        'src_lang': lang,
                        'src_text': paragraph,
                        'original_id': example.get('id', f'{lang}_{idx}'),
                        'article_idx': idx,
                        'paragraph_idx': para_idx
                    }
                    data_by_lang[lang].append(item)
                    paragraph_count += 1
            
            print(f"  Loaded {len(dataset)} articles → {paragraph_count} paragraphs")
        
        except Exception as e:
            print(f"  Error loading {lang}: {e}")
    
    return data_by_lang


def deduplicate_data(data: List[Dict], field: str) -> List[Dict]:
    """Remove exact duplicates based on text hash."""
    seen_hashes: Set[str] = set()
    deduplicated = []
    
    for item in data:
        text = item.get(field, "")
        text_hash = hash_text(text)
        
        if text_hash not in seen_hashes:
            seen_hashes.add(text_hash)
            deduplicated.append(item)
    
    return deduplicated


def sample_balanced(data_by_lang: Dict[str, List[Dict]], 
                   samples_per_lang: int,
                   field: str = "text") -> Dict[str, List[Dict]]:
    """Sample with length matching across languages."""
    sampled_data = {}
    
    for lang, items in data_by_lang.items():
        # Deduplicate first
        items = deduplicate_data(items, field)
        
        # Group by length bins
        length_bins = defaultdict(list)
        for item in items:
            text = item.get(field, "")
            bin_idx = get_length_bin(text)
            length_bins[bin_idx].append(item)
        
        # Sample evenly across bins
        samples = []
        bins = sorted(length_bins.keys())
        samples_per_bin = max(1, samples_per_lang // len(bins))
        
        for bin_idx in bins:
            bin_items = length_bins[bin_idx]
            n_samples = min(len(bin_items), samples_per_bin)
            samples.extend(random.sample(bin_items, n_samples))
        
        # If we need more samples, take randomly from remaining
        if len(samples) < samples_per_lang:
            remaining = [item for item in items if item not in samples]
            additional = min(len(remaining), samples_per_lang - len(samples))
            samples.extend(random.sample(remaining, additional))
        
        sampled_data[lang] = samples[:samples_per_lang]
    
    return sampled_data


def split_into_pools(data_by_lang: Dict[str, List[Dict]], 
                     pool_ratios: Dict[str, float]) -> Dict[str, Dict[str, List[Dict]]]:
    """Split data into disjoint pools A (classifier), B (RL prompts), C (validation)."""
    pools = {pool_name: {} for pool_name in pool_ratios.keys()}
    
    for lang, items in data_by_lang.items():
        random.shuffle(items)
        n_total = len(items)
        
        start_idx = 0
        for pool_name, ratio in pool_ratios.items():
            n_pool = int(n_total * ratio)
def main():
    parser = argparse.ArgumentParser(description="Sample XL-Sum data with paragraph splitting")
    parser.add_argument("--languages", nargs="+", required=True, 
                       help="List of source languages (e.g., english french japanese)")
    parser.add_argument("--paragraphs_per_lang", type=int, default=1000, 
                       help="Number of paragraphs per language (final total will be equal)")
    parser.add_argument("--field", type=str, default="text", 
                       help="Field to use from XL-Sum (text for full article)")
    parser.add_argument("--min_sentences", type=int, default=5, 
                       help="Minimum sentences per paragraph")
    parser.add_argument("--max_sentences", type=int, default=7, 
                       help="Maximum sentences per paragraph")
    parser.add_argument("--output_dir", type=str, default="data/pools", 
                       help="Output directory")
    parser.add_argument("--pool_a_ratio", type=float, default=0.4, 
                       help="Ratio for Pool A (classifier)")
    parser.add_argument("--pool_b_ratio", type=float, default=0.3, 
                       help="Ratio for Pool B (RL prompts)")
    parser.add_argument("--pool_c_ratio", type=float, default=0.3, 
                       help="Ratio for Pool C (validation)")
    parser.add_argument("--split", type=str, default="train",
                       help="Dataset split to use (train/validation/test)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Load data and split into paragraphs
    print("=" * 60)
    print("Loading XL-Sum data and splitting into paragraphs...")
    print("=" * 60)
    data_by_lang = load_xlsum_data(
        languages=args.languages,
        field=args.field,
        min_sentences=args.min_sentences,
        max_sentences=args.max_sentences,
        split=args.split
    )
    
    # Sample balanced paragraphs (equal count per language)
    print("\n" + "=" * 60)
    print("Sampling balanced paragraphs with length matching...")
    print("=" * 60)
    sampled_data = sample_balanced_paragraphs(
        data_by_lang, 
        args.paragraphs_per_lang,
        field="src_text"
    )
    
    # Split into pools
    print("\n" + "=" * 60)
    print("Splitting into disjoint pools...")
    print("=" * 60)
    pool_ratios = {
        'A': args.pool_a_ratio,
        'B': args.pool_b_ratio,
        'C': args.pool_c_ratio
    }
    pools = split_into_pools(sampled_data, pool_ratios)
    
    # Save pools
    print("\nSaving pools...")
    save_pools(pools, Path(args.output_dir))
    
    # Print statistics
    print("\n" + "=" * 60)
    print("=== Final Statistics ===")
    print("=" * 60)
    for pool_name, pool_data in pools.items():
        total = sum(len(items) for items in pool_data.values())
        print(f"\nPool {pool_name}: {total} paragraphs across {len(pool_data)} languages")
        for lang, items in pool_data.items():
            print(f"  {lang}: {len(items)} paragraphs")
    # Sample balanced data
    print("Sampling balanced data with length matching...")
    sampled_data = sample_balanced(data_by_lang, args.samples_per_lang, args.field)
    
    # Split into pools
    print("Splitting into disjoint pools...")
    pool_ratios = {
        'A': args.pool_a_ratio,
        'B': args.pool_b_ratio,
        'C': args.pool_c_ratio
    }
    pools = split_into_pools(sampled_data, pool_ratios)
    
    # Save pools
    print("Saving pools...")
    save_pools(pools, Path(args.output_dir))
    
    # Print statistics
    print("\n=== Statistics ===")
    for pool_name, pool_data in pools.items():
        total = sum(len(items) for items in pool_data.values())
        print(f"Pool {pool_name}: {total} samples across {len(pool_data)} languages")
        for lang, items in pool_data.items():
            print(f"  {lang}: {len(items)} samples")


if __name__ == "__main__":
    main()
