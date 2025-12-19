"""
Download native Chinese data from two sources and split into pools.

Sources:
- Papersnake/people_daily_news via HuggingFace datasets-server API
- csebuetnlp/xlsum (chinese_simplified) via HuggingFace datasets-server API

Selection rules:
- people_daily_news: date in [1990.01.01, 2024.12.31]; sample rows starting at
    row 1,710,000. Always select entries from 2023/2024 with page exactly "17";
    otherwise keep each row with 20% probability. From text, randomly cut a
    300–500 character segment.
- xlsum chinese_simplified: randomly cut 300–500 characters from "text".

Output:
- Append to data/original_zh/pool_a.jsonl (200 per source), pool_b.jsonl (150 per source), pool_c.jsonl (150 per source)
- Record fields: id, src_lang, src_text; plus provenance fields (source, original_id, date/title/page where available).
"""

import argparse
import json
import random
import time
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import requests


HF_ROWS_URL = "https://datasets-server.huggingface.co/rows"
USER_AGENT = "WAYF-original-zh-fetch/1.0"
PDN_MIN_OFFSET = 1_710_000
PDN_MAX_OFFSET = 1_982_265


def parse_date(date_str: str) -> Optional[datetime]:
    """Parse date in format YYYY.MM.DD; return None if invalid."""
    try:
        return datetime.strptime((date_str or "").strip(), "%Y.%m.%d")
    except Exception:
        return None


def is_in_date_range(date_str: str, start: str, end: str) -> bool:
    d = parse_date(date_str)
    s = parse_date(start)
    e = parse_date(end)
    if not d or not s or not e:
        return False
    return s <= d <= e


def random_cn_segment(text: str, min_len: int = 300, max_len: int = 500) -> Optional[str]:
    """Pick a random substring of length in [min_len, max_len]; return None if too short."""
    s = (text or "").strip()
    if len(s) < min_len:
        return None
    length = random.randint(min_len, max_len)
    if len(s) <= length:
        return s[:length]
    start = random.randint(0, len(s) - length)
    return s[start : start + length]


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


def collect_people_daily_news(n_target: int,
                              start_date: str,
                              end_date: str,
                              batch: int = 100) -> List[Dict]:
    """Collect n_target items from Papersnake/people_daily_news according to rules."""
    dataset = "Papersnake/people_daily_news"
    config = "default"
    split = "train"

    meta = hf_fetch_rows(dataset, config, split, offset=0, length=1)
    total = int(meta.get("num_rows_total", 0))
    max_offset = PDN_MAX_OFFSET
    min_offset = PDN_MIN_OFFSET 

    seen = set()
    out: List[Dict] = []

    while len(out) < n_target:
        offset = random.randint(min_offset, max_offset)
        try:
            page = hf_fetch_rows(dataset, config, split, offset=offset, length=batch)
        except Exception:
            time.sleep(1.0)
            continue

        for row in page.get("rows", []):
            if len(out) >= n_target:
                break
            row_idx = row.get("row_idx")
            if row_idx in seen:
                continue
            seen.add(row_idx)

            data = row.get("row", {})
            date = data.get("date", "")
            title = data.get("title", "")
            author = data.get("author", "")
            page_info = data.get("page", "")
            text = data.get("text", "")

            if not is_in_date_range(date, start_date, end_date) or not text:
                continue

            page_s = (page_info or "").strip()

            date_year = date.strip()[:4]
            is_target_year = date_year in {"2023", "2024"}
            must_select = is_target_year and page_s == "17"
            if not must_select and random.random() > 0.20:
                continue

            segment = random_cn_segment(text)
            if not segment:
                continue

            item = {
                "id": f"people_daily_news_{row_idx}",
                "src_lang": "chinese",
                "src_text": segment,
                "source": "people_daily_news",
                "original_id": str(row_idx),
                "date": date,
                "title": title,
                "author": author,
                "page": page_info,
            }
            out.append(item)

        time.sleep(0.2)

    return out


def collect_xlsum_chs(n_target: int, batch: int = 100) -> List[Dict]:
    """Collect n_target items from csebuetnlp/xlsum chinese_simplified using datasets-server."""
    dataset = "csebuetnlp/xlsum"
    config = "chinese_simplified"
    split = "train"

    meta = hf_fetch_rows(dataset, config, split, offset=0, length=1)
    total = int(meta.get("num_rows_total", 0))

    seen = set()
    out: List[Dict] = []

    while len(out) < n_target:
        offset = random.randint(0, max(0, total - batch))
        try:
            page = hf_fetch_rows(dataset, config, split, offset=offset, length=batch)
        except Exception:
            time.sleep(1.0)
            continue

        for row in page.get("rows", []):
            if len(out) >= n_target:
                break
            row_idx = row.get("row_idx")
            if row_idx in seen:
                continue
            seen.add(row_idx)

            data = row.get("row", {})
            text = data.get("text", "")
            original_id = data.get("id", str(row_idx))

            if not text:
                continue

            segment = random_cn_segment(text)
            if not segment:
                continue

            item = {
                "id": f"xlsum_{original_id}",
                "src_lang": "chinese",
                "src_text": segment,
                "source": "xlsum_chinese_simplified",
                "original_id": str(original_id),
            }
            out.append(item)

        time.sleep(0.2)

    return out


def split_and_save(items: List[Dict], out_dir: Path, a_count: int, b_count: int, c_count: int):
    """Shuffle and append items into pool_a/b/c.jsonl with given counts."""
    random.shuffle(items)
    out_dir.mkdir(parents=True, exist_ok=True)

    pools = [
        ("pool_a.jsonl", a_count),
        ("pool_b.jsonl", b_count),
        ("pool_c.jsonl", c_count),
    ]

    start = 0
    for fname, count in pools:
        subset = items[start : start + count]
        start += count
        with open(out_dir / fname, "a", encoding="utf-8") as f:
            for it in subset:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Download native Chinese data and split into pools")
    parser.add_argument("--pdn_count", type=int, default=500, help="Count from people_daily_news")
    parser.add_argument("--xlsum_count", type=int, default=500, help="Count from xlsum chinese_simplified")
    parser.add_argument("--start_date", type=str, default="1990.01.01", help="Start date for PDN")
    parser.add_argument("--end_date", type=str, default="2024.12.31", help="End date for PDN")
    parser.add_argument("--out_dir", type=str, default="data/original_zh", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)

    print("Collecting people_daily_news...")
    pdn_items = collect_people_daily_news(args.pdn_count, args.start_date, args.end_date)
    print(f"Collected PDN: {len(pdn_items)}")
    split_and_save(pdn_items, Path(args.out_dir), 200, 150, 150)

    print("Collecting xlsum chinese_simplified...")
    xlsum_items = collect_xlsum_chs(args.xlsum_count)
    print(f"Collected XLSum: {len(xlsum_items)}")
    split_and_save(xlsum_items, Path(args.out_dir), 200, 150, 150)

    print(f"Done. Wrote to {args.out_dir}/pool_a.jsonl, pool_b.jsonl, pool_c.jsonl")


if __name__ == "__main__":
    main()