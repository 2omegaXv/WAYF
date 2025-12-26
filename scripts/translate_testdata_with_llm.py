"""
Script to batch translate text files in test_data/original/ using LLM API.
Each input .txt file is named [lang]_[suffix].txt, each line is a text to translate.
Output is written as JSONL to test_data/translated/[lang]_[suffix]_zh.jsonl
"""

import os
import json
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
import importlib.util
import sys
from pathlib import Path

# Dynamically import 02_translate_with_llm.py as a module
llm_module_path = Path(__file__).parent / "02_translate_with_llm.py"
spec = importlib.util.spec_from_file_location("llm_mod", llm_module_path)
llm_mod = importlib.util.module_from_spec(spec)
sys.modules["llm_mod"] = llm_mod
spec.loader.exec_module(llm_mod)

LLMClient = llm_mod.LLMClient
get_system_prompt = llm_mod.get_system_prompt
get_user_prompt = llm_mod.get_user_prompt
post_process_translation = llm_mod.post_process_translation
check_cjk_ratio = llm_mod.check_cjk_ratio
hash_text = llm_mod.hash_text

# Resolve paths relative to repository root (two levels up from this script)
repo_root = Path(__file__).resolve().parents[1]
INPUT_DIR = repo_root / 'test_data' / 'spanish'
OUTPUT_DIR = repo_root / 'test_data' / 'translated'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

API_KEYS = []
BASE_URL = "https://llmapi.paratera.com"
MODEL_NAME = "DeepSeek-V3.2"
TEMPERATURE = 0.0
TOP_P = 1.0
MAX_TOKENS = 2048
MIN_CJK_RATIO = 0.3
PROMPT_VERSION = "v1.0"
NUM_WORKERS = 800


def parse_lang_and_suffix(filename):
    name = filename.stem
    if '_' in name:
        lang, suffix = name.split('_', 1)
    else:
        lang, suffix = name, ''
    return lang, suffix


def translate_lines(lines, lang, llm_client, pool_name, output_file=None, num_workers=NUM_WORKERS):
    """Translate lines and optionally stream each result to `output_file` as JSONL.
    If `output_file` is provided, each successful translation is appended and flushed immediately.
    Returns the number of successful translations written.
    """
    system_prompt = get_system_prompt()

    # Prepare jobs (index + text) and count non-empty
    jobs = [(i, line.strip()) for i, line in enumerate(lines) if line.strip()]
    total_jobs = len(jobs)
    if total_jobs == 0:
        return 0

    writer_lock = threading.Lock()
    written = 0

    def worker(job):
        idx, src_text = job
        max_attempts = 5
        backoff = 2
        last_err = None
        raw_response = None
        for attempt in range(1, max_attempts + 1):
            try:
                user_prompt = get_user_prompt(lang, src_text)
                raw_response = llm_client.translate(system_prompt, user_prompt, max_retries=3)
                clean_response = raw_response.strip()
                # If response is fenced in a markdown code block, extract inner content
                if clean_response.startswith('```'):
                    import re
                    match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', clean_response, re.DOTALL)
                    if match:
                        clean_response = match.group(1).strip()

                # Try strict JSON parse first; if it fails, attempt to sanitize common errors
                try:
                    response_data = json.loads(clean_response)
                except Exception:
                    # If the model returned a JSON-like object but with unescaped quotes
                    # inside the translation value (common), try to fix that by escaping
                    # internal quotes within the "translation" value.
                    def try_fix_translation_field(text: str):
                        key = '"translation"'
                        idx = text.find(key)
                        if idx == -1:
                            return None
                        # find colon after key
                        colon = text.find(':', idx)
                        if colon == -1:
                            return None
                        # find bounding '}' for this object (fallback to end)
                        brace = text.find('}', colon)
                        if brace == -1:
                            brace = len(text) - 1
                        # find first quote starting the value
                        first_q = text.find('"', colon)
                        if first_q == -1 or first_q >= brace:
                            return None
                        # find last quote before the closing brace
                        last_q = text.rfind('"', colon, brace)
                        if last_q == -1 or last_q <= first_q:
                            return None
                        val_raw = text[first_q+1:last_q]
                        # Escape backslashes first, then double quotes
                        val_escaped = val_raw.replace('\\', '\\\\').replace('"', '\\"')
                        fixed = text[:first_q+1] + val_escaped + text[last_q:]
                        return fixed

                    fixed_text = try_fix_translation_field(clean_response)
                    if fixed_text is not None:
                        try:
                            response_data = json.loads(fixed_text)
                        except Exception:
                            response_data = None
                    else:
                        response_data = None

                    # Final fallback: extract first '{...}' substring and parse
                    if response_data is None:
                        start = clean_response.find('{')
                        end = clean_response.rfind('}')
                        if start != -1 and end != -1 and end > start:
                            try:
                                response_data = json.loads(clean_response[start:end+1])
                            except Exception:
                                raise
                        else:
                            raise
                zh_mt = response_data.get('translation', '')
                zh_mt = post_process_translation(zh_mt)
                if not check_cjk_ratio(zh_mt, MIN_CJK_RATIO):
                    raise ValueError(f"Insufficient CJK ratio for line {idx}")

                output_record = {
                    'id': hash_text(src_text),
                    'src_lang': lang,
                    'src_text': src_text,
                    'zh_mt': zh_mt,
                    'audit': {
                        'model': llm_client.model_name,
                        'temperature': llm_client.temperature,
                        'top_p': llm_client.top_p,
                        'max_tokens': llm_client.max_tokens,
                        'prompt_version': PROMPT_VERSION
                    }
                }
                return (idx, output_record, None, raw_response)
            except Exception as e:
                last_err = e
                if attempt < max_attempts:
                    wait_time = min(backoff * 2, 60)
                    print(f"Line {idx} attempt {attempt} failed: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    return (idx, None, last_err, raw_response)

    # Open output file for appending; we'll write as tasks complete
    out_f = None
    if output_file:
        out_f = open(output_file, 'a', encoding='utf-8')

    try:
        with ThreadPoolExecutor(max_workers=min(num_workers, total_jobs)) as executor:
            future_to_job = {executor.submit(worker, job): job for job in jobs}

            with tqdm(total=total_jobs, desc=f"Translating {pool_name}") as pbar:
                for future in as_completed(future_to_job):
                    idx, record, err, raw = future.result()
                    if err:
                        print(f"Error translating line {idx}: {err}")
                        if raw:
                            snippet = raw if len(raw) <= 2000 else raw[:2000] + '\n... [truncated]'
                            print(f"Raw response (len={len(raw)}):\n{snippet}")
                    else:
                        if out_f:
                            with writer_lock:
                                out_f.write(json.dumps(record, ensure_ascii=False) + '\n')
                                out_f.flush()
                        written += 1
                    pbar.update(1)
    finally:
        if out_f:
            out_f.close()

    return written


def main():
    llm_client = LLMClient(
        model_name=MODEL_NAME,
        api_keys=API_KEYS,
        base_url=BASE_URL,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_TOKENS
    )
    print(f"Working directory: {Path.cwd()}")
    print(f"Resolved input dir: {INPUT_DIR}")

    txt_files = sorted(INPUT_DIR.glob('*.txt')) if INPUT_DIR.exists() else []
    if not txt_files:
        print(f"No input .txt files found in {INPUT_DIR}. Run the script from the project root or place files there.")
        return

    for txt_file in txt_files:
        lang, suffix = parse_lang_and_suffix(txt_file)
        pool_name = f"{lang}_{suffix}" if suffix else lang
        output_file = OUTPUT_DIR / f"{pool_name}_zh.jsonl"
        print(f"Translating {txt_file} -> {output_file}")
        # Truncate/create output file so we start fresh for this input
        with open(output_file, 'w', encoding='utf-8'):
            pass

        with open(txt_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        written = translate_lines(lines, lang, llm_client, pool_name, output_file=output_file)
        print(f"Done: {output_file} ({written} translated lines written)")

if __name__ == "__main__":
    main()
