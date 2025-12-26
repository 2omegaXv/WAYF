import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time
from tqdm import tqdm
import openai
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import yaml
from sklearn.metrics import accuracy_score, f1_score
import jieba.posseg as pseg

DEFAULT_ALLOWED_LABELS = [
    "chinese",
    "english",
    "french",
    "german",
    "japanese",
    "korean",
    "russian",
    "spanish",
]

ZERO_SHOT = """This Chinese sequence may be translated from a foreign language, or it may be originally written in Chinese.
Please determine which language this sequence is translated from.
Respond with only ONE label from this list: ['chinese', 'english', 'french', 'german', 'japanese', 'korean', 'russian', 'spanish'].
If you think it is originally written in Chinese, respond with 'chinese'."""

# --- Masking Logic ---
ENTITY_MAP = {
    'nr': '[unused1]',  # Person Name
    'ns': '[unused2]',  # Place Name
    'nt': '[unused3]',  # Organization
    'nz': '[unused4]',  # Other Proper Noun
}

def mask_entities(text: str) -> str:
    """
    Mask person names (nr), place names (ns), organization names (nt), 
    and other proper nouns (nz) with specific [unused] tokens.
    """
    words = pseg.cut(text)
    masked_text = ""
    for word, flag in words:
        # nr: Person name
        # ns: Place name
        # nt: Organization
        # nz: Other proper noun
        if flag.startswith('nr'):
            masked_text += ENTITY_MAP['nr']
        elif flag.startswith('ns'):
            masked_text += ENTITY_MAP['ns']
        elif flag.startswith('nt'):
            masked_text += ENTITY_MAP['nt']
        elif flag.startswith('nz'):
            masked_text += ENTITY_MAP['nz']
        else:
            masked_text += word
    return masked_text
# ---------------------

class LLMClient:
    """LLM API client using OpenAI SDK with support for multiple API keys."""
    
    def __init__(self, model_name: str, api_keys: List[str], base_url: str, temperature: float = 0.0, top_p: float = 1.0, max_tokens: int = 1024):
        self.model_name = model_name
        self.base_url = base_url
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        
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
    
    def classify(self, system_prompt: str, user_prompt: str, max_retries: int = 3) -> str:
        """Call LLM API for classification with retry logic."""
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
                    response_format={"type": "json_object"},
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=self.max_tokens
                )
                
                return response.choices[0].message.content
                
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2  # 2, 4, 6 seconds
                    print(f"\nAPI call failed (attempt {attempt + 1}/{max_retries}): {e}")
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    raise Exception(f"API call failed after {max_retries} attempts: {e}")


def get_system_prompt() -> str:
    return "You are a careful and consistent text classifier."


def get_user_prompt(zh_text: str, allowed_labels: List[str]) -> str:
    allowed = ", ".join([f"'{x}'" for x in allowed_labels])
    return (
        f"{ZERO_SHOT}\n\n"
        f"Chinese sequence:\n{zh_text}\n\n"
        f"Output ONLY valid JSON like: {{\"label\": \"english\"}}\n"
        f"Allowed labels: [{allowed}]"
    )


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def hash_response(response: str) -> str:
    """Generate hash of raw API response."""
    return hashlib.sha256(response.encode('utf-8')).hexdigest()


def _parse_json_label(raw_response: str) -> Optional[str]:
    """Parse {"label": "..."} from a model response.

    Keeps parsing intentionally strict; returns None if invalid.
    """
    try:
        payload = json.loads(raw_response.strip())
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    label = payload.get("label")
    if not isinstance(label, str):
        return None
    label = label.strip().lower()
    return label or None


def classify_single_item(
    item_data: Tuple[Dict, int, str],
    llm_client: LLMClient,
    allowed_labels: List[str],
    max_attempts: int,
    max_retries: int,
) -> Dict:
    item, line_idx, dataset = item_data
    gold = (item.get("src_lang", "") or "").strip().lower()
    zh_text = item.get("zh_mt", "")
    if not zh_text and gold == "chinese":
        zh_text = item.get("src_text", "")
    
    # Apply masking
    if zh_text:
        zh_text = mask_entities(zh_text)
    
    item_id = item.get("id") or hash_text(zh_text)

    system_prompt = get_system_prompt()
    user_prompt = get_user_prompt(zh_text, allowed_labels)

    backoff = 2
    last_error: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        try:
            raw_response = llm_client.classify(system_prompt, user_prompt, max_retries=max_retries)
            pred = _parse_json_label(raw_response)
            if pred not in allowed_labels:
                pred = "unknown"

            return {
                "id": item_id,
                "line_idx": line_idx,
                "dataset": dataset,
                "gold": gold,
                "pred": pred,
                "audit": {
                    "model": llm_client.model_name,
                    "temperature": llm_client.temperature,
                    "top_p": llm_client.top_p,
                    "max_tokens": llm_client.max_tokens,
                    "raw_response_hash": hash_response(raw_response),
                },
            }
        except Exception as e:
            last_error = e
            if attempt < max_attempts:
                wait_time = min(backoff, 60)
                time.sleep(wait_time)
                backoff = min(backoff * 2, 60)
            else:
                return {
                    "id": item_id,
                    "line_idx": line_idx,
                    "dataset": dataset,
                    "gold": gold,
                    "pred": "error",
                    "error": str(last_error),
                }


def run_llm_baseline(
    test_files: List[Path],
    output_dir: Path,
    llm_client: LLMClient,
    num_workers: int,
    allowed_labels: List[str],
    max_attempts: int,
    max_retries: int,
    limit: Optional[int] = None,
) -> Dict[str, float]:
    output_dir.mkdir(parents=True, exist_ok=True)
    preds_path = output_dir / "predictions.jsonl"
    metrics_path = output_dir / "metrics.json"

    # Load test examples
    items: List[Tuple[Dict, int, str]] = []
    for test_file in test_files:
        dataset = test_file.stem  # Use filename (without .jsonl) as dataset
        with test_file.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if limit is not None and len(items) >= limit:
                    break
                obj = json.loads(line)
                items.append((obj, idx, dataset))

    # Truncate predictions file
    preds_path.write_text("", encoding="utf-8")

    writer_lock = threading.Lock()
    gold_labels: List[str] = []
    pred_labels: List[str] = []
    datasets: List[str] = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(
                classify_single_item,
                item_data,
                llm_client,
                allowed_labels,
                max_attempts,
                max_retries,
            )
            for item_data in items
        ]

        with tqdm(total=len(items), desc="LLM baseline (classifying)") as pbar:
            for fut in as_completed(futures):
                rec = fut.result()
                with writer_lock:
                    with preds_path.open("a", encoding="utf-8") as out:
                        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if rec.get("gold") is not None and rec.get("pred") is not None:
                    gold_labels.append(rec["gold"])
                    pred_labels.append(rec["pred"])
                    datasets.append(rec["dataset"])
                pbar.update(1)

    # Metrics (ignore empty)
    labels_union = sorted(set(gold_labels) | set(pred_labels))
    accuracy = float(accuracy_score(gold_labels, pred_labels)) if gold_labels else 0.0
    macro_f1 = float(f1_score(gold_labels, pred_labels, average="macro", labels=labels_union, zero_division=0)) if gold_labels else 0.0

    # --- Per-language breakdown ---
    lang_stats = {}
    dataset_stats = {}
    for g, p, d in zip(gold_labels, pred_labels, datasets):
        if g not in lang_stats:
            lang_stats[g] = {"correct": 0, "total": 0}
        lang_stats[g]["total"] += 1
        if g == p:
            lang_stats[g]["correct"] += 1
        
        if d not in dataset_stats:
            dataset_stats[d] = {"correct": 0, "total": 0}
        dataset_stats[d]["total"] += 1
        if g == p:
            dataset_stats[d]["correct"] += 1

    print("\nResults Breakdown by Dataset:")
    print("-" * 40)
    print(f"{'Dataset':<15} | {'Accuracy':<10} | {'Count':<10}")
    print("-" * 40)
    for ds in sorted(dataset_stats.keys()):
        stats = dataset_stats[ds]
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        print(f"{ds:<15} | {acc:<10.4f} | {stats['total']:<10}")
    print("-" * 40)
    print(f"{'OVERALL':<15} | {accuracy:<10.4f} | {len(gold_labels):<10}")
    print("=" * 40 + "\n")
    # ------------------------------

    metrics = {
        "n": int(len(gold_labels)),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_language": {
            l: {"accuracy": s["correct"] / s["total"], "count": s["total"]} 
            for l, s in lang_stats.items()
        },
        "per_dataset": {
            d: {"accuracy": s["correct"] / s["total"], "count": s["total"]} 
            for d, s in dataset_stats.items()
        }
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def _ensure_list(value) -> List[Path]:
    if value is None:
        return []
    if isinstance(value, list):
        return [Path(v) for v in value]
    return [Path(value)]


def main():
    parser = argparse.ArgumentParser(description="LLM baseline for source-language classification (Masked)")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/classifier_baseline_config.yaml",
        help="YAML config for LLM baseline",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for quick runs")
    
    args = parser.parse_args()

    # Load config
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    experiment_cfg = cfg.get("experiment") or {}
    
    # Hardcode for this specific request: DeepSeek on DeepSeek Pool C
    trans_model = "DeepSeek-V3.2"
        
    print(f"Auto-resolving test files for Translation Model: {trans_model}")
    input_dir = Path("test_data/train_split")
    if not input_dir.exists():
        raise FileNotFoundError(f"Translation directory not found: {input_dir}")
    
    # Find all files matching {pool}_*_zh.jsonl (e.g., c_english_zh.jsonl)
    found_files = sorted(list(input_dir.glob(f"spanish_books_zh_val.jsonl")))
    
    test_files = found_files
    print(f"  Found {len(test_files)} files.")

    # Output Directory
    output_dir = Path("models/baselines/llm/DeepSeek-V3.2_noconf_masked")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # LLM Config
    llm_cfg = cfg.get("llm", {})
    base_url = str(llm_cfg.get("base_url") or "https://llmapi.paratera.com")
    model_name = "DeepSeek-V3.2" # Force DeepSeek-V3.2
    num_workers = int(llm_cfg.get("num_workers", 8))
    max_attempts = int(llm_cfg.get("max_attempts", 2))
    max_retries = int(llm_cfg.get("max_retries", 3))
    temperature = float(llm_cfg.get("temperature", 0.0))
    top_p = float(llm_cfg.get("top_p", 1.0))
    max_tokens = int(llm_cfg.get("max_tokens", 1024))

    api_keys: List[str] = []
    if llm_cfg.get("api_keys") is not None:
        api_keys = [k for k in _ensure_list(llm_cfg.get("api_keys"))]
    else:
        api_keys_env = str(llm_cfg.get("api_keys_env") or "API_KEYS")
        env_val = os.environ.get(api_keys_env) or os.environ.get("LLM_API_KEYS") or os.environ.get("OPENAI_API_KEY")
        if env_val:
            # allow comma-separated in env
            api_keys = [k.strip() for k in str(env_val).split(",") if k.strip()]

    if not api_keys:
        raise ValueError(
            "No API keys provided. Set `llm.api_keys` in the YAML, or set an env var "
            "(default: API_KEYS) and point `llm.api_keys_env` to it."
        )

    prompt_cfg = cfg.get("prompt", {}) or {}
    allowed_labels = _ensure_list(prompt_cfg.get("allowed_labels"))
    allowed_labels = [str(x).strip().lower() for x in allowed_labels if str(x).strip()]
    if not allowed_labels:
        allowed_labels = DEFAULT_ALLOWED_LABELS

    llm_client = LLMClient(
        model_name=model_name,
        api_keys=api_keys,
        base_url=base_url,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    print(f"Test file: {test_files}")
    print(f"Output dir: {output_dir}")
    print(f"Model: {model_name}")
    print(f"Base URL: {base_url}")
    print(f"Workers: {num_workers}")

    metrics = run_llm_baseline(
        test_files=test_files,
        output_dir=output_dir,
        llm_client=llm_client,
        num_workers=num_workers,
        allowed_labels=allowed_labels,
        max_attempts=max_attempts,
        max_retries=max_retries,
        limit=args.limit,
    )

    print("\n=== LLM Baseline Metrics (Masked) ===")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
