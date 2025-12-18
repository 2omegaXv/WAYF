"""
Script for sampling XL-Sum data with language selection, length matching, and deduplication.
Implements step 1.1 and 1.2 from project procedure.
"""

import argparse
import json
import hashlib
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Set
import random


def hash_text(text: str) -> str:
    """Generate hash for deduplication."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def get_length_bin(text: str, bin_size: int = 50) -> int:
    """Bin text by character length."""
    return len(text) // bin_size


def load_xlsum_data(data_path: str, languages: List[str], field: str = "summary") -> Dict[str, List[Dict]]:
    """Load XL-Sum data for specified languages."""
    data_by_lang = defaultdict(list)
    
    # TODO: Implement actual XL-Sum loading logic
    # This is a placeholder for the data loading implementation
    print(f"Loading XL-Sum data from {data_path}")
    print(f"Languages: {languages}")
    print(f"Field: {field}")
    
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
                   field: str = "summary") -> Dict[str, List[Dict]]:
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
            end_idx = start_idx + n_pool
            pools[pool_name][lang] = items[start_idx:end_idx]
            start_idx = end_idx
    
    return pools


def save_pools(pools: Dict[str, Dict[str, List[Dict]]], output_dir: Path):
    """Save pools to JSONL files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for pool_name, pool_data in pools.items():
        output_file = output_dir / f"pool_{pool_name.lower()}.jsonl"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for lang, items in pool_data.items():
                for item in items:
                    item['pool'] = pool_name
                    item['src_lang'] = lang
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"Saved {pool_name} to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Sample XL-Sum data for translation project")
    parser.add_argument("--data_path", type=str, required=True, help="Path to XL-Sum dataset")
    parser.add_argument("--languages", nargs="+", required=True, help="List of source languages")
    parser.add_argument("--samples_per_lang", type=int, default=1000, help="Number of samples per language")
    parser.add_argument("--field", type=str, default="summary", help="Field to use (summary or text)")
    parser.add_argument("--output_dir", type=str, default="data/pools", help="Output directory")
    parser.add_argument("--pool_a_ratio", type=float, default=0.4, help="Ratio for Pool A (classifier)")
    parser.add_argument("--pool_b_ratio", type=float, default=0.3, help="Ratio for Pool B (RL prompts)")
    parser.add_argument("--pool_c_ratio", type=float, default=0.3, help="Ratio for Pool C (validation)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Load data
    print("Loading XL-Sum data...")
    data_by_lang = load_xlsum_data(args.data_path, args.languages, args.field)
    
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
