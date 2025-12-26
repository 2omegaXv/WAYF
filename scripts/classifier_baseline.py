"""scripts/classifier_baseline.py

LLM-based baseline for the source-language classifier.

This script runs zero-shot classification over the same JSONL schema used by
`scripts/06_train_classifier.py` (expects `zh_mt` text and `src_lang` label),
and reports accuracy + macro-F1.
"""

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
    item_data: Tuple[Dict, int],
    llm_client: LLMClient,
    allowed_labels: List[str],
    max_attempts: int,
    max_retries: int,
) -> Dict:
    item, line_idx = item_data
    gold = (item.get("src_lang", "") or "").strip().lower()
    zh_text = item.get("zh_mt", "")
    if not zh_text and gold == "chinese":
        zh_text = item.get("src_text", "")
    
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
    items: List[Tuple[Dict, int]] = []
    for test_file in test_files:
        with test_file.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if limit is not None and len(items) >= limit:
                    break
                obj = json.loads(line)
                items.append((obj, idx))

    # Truncate predictions file
    preds_path.write_text("", encoding="utf-8")

    writer_lock = threading.Lock()
    gold_labels: List[str] = []
    pred_labels: List[str] = []

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
                pbar.update(1)

    # Metrics (ignore empty)
    labels_union = sorted(set(gold_labels) | set(pred_labels))
    accuracy = float(accuracy_score(gold_labels, pred_labels)) if gold_labels else 0.0
    macro_f1 = float(f1_score(gold_labels, pred_labels, average="macro", labels=labels_union, zero_division=0)) if gold_labels else 0.0

    metrics = {
        "n": int(len(gold_labels)),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
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
    parser = argparse.ArgumentParser(description="LLM baseline for source-language classification")
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
    
    trans_model = experiment_cfg.get("translation_model")
    pool = experiment_cfg.get("pool")
        
    if trans_model and pool:
        print(f"Auto-resolving test files for Translation Model: {trans_model}, Pool: {pool}")
        input_dir = Path("new_data/translated") / trans_model
        if not input_dir.exists():
            raise FileNotFoundError(f"Translation directory not found: {input_dir}")
        
        # Find all files matching {pool}_*_zh.jsonl (e.g., c_english_zh.jsonl)
        found_files = sorted(list(input_dir.glob(f"{pool}_*_zh.jsonl")))
        if not found_files:
            raise FileNotFoundError(f"No files found matching pattern '{pool}_*_zh.jsonl' in {input_dir}")
        
        test_files = found_files
        print(f"  Found {len(test_files)} files.")
    else:
        test_files = []

    if not test_files:
        raise ValueError("No test files provided in config or found via auto-resolution.")

    # 2. Resolve Output Directory
    paths_cfg = cfg.get("paths", {}) or {}
    output_dir_str = paths_cfg.get("output_dir")
    if not output_dir_str:
        classifier_model = cfg.get("llm", {}).get("model_name", "unknown_model")
        pool = experiment_cfg.get("pool", "unknown_pool")
        trans_model = experiment_cfg.get("translation_model", "unknown_trans")
        
        # Sanitize model name (e.g. replace / with _)
        safe_classifier_name = classifier_model.replace("/", "_")
        safe_trans_name = trans_model.replace("/", "_")
        
        output_dir_str = f"models/baselines/llm/{safe_classifier_name}_on_{safe_trans_name}_pool_{pool}"
        print(f"Auto-resolved output directory: {output_dir_str}")
    
    output_dir = Path(output_dir_str)
    output_dir.mkdir(parents=True, exist_ok=True)
    # -----------------------------------------------------

    # LLM Config
    llm_cfg = cfg.get("llm", {})
    base_url = str(llm_cfg.get("base_url") or "https://llmapi.paratera.com")
    model_name = str(llm_cfg.get("model_name") or "DeepSeek-V3.2")
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

    print("\n=== LLM Baseline Metrics ===")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
