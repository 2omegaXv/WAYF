from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
except Exception:  # pragma: no cover
    PeftModel = None


def _load_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"input_data not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        records: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records

    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
            return data["data"]
        raise ValueError(f"Unsupported JSON structure in {path}")

    if suffix == ".csv":
        import pandas as pd

        df = pd.read_csv(path)
        return df.to_dict(orient="records")

    raise ValueError(f"Unsupported input format: {path} (expected .jsonl/.json/.csv)")


def _chunked(items: List[Any], batch_size: int) -> Iterable[List[Any]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def _is_peft_adapter_dir(path: Path) -> bool:
    return path.is_dir() and (path / "adapter_config.json").exists()


def _build_prompt(src_text: str) -> str:
    return f"Translate the following japanese text to Chinese:\n\n{src_text}"


def _format_inputs(tokenizer, prompts: List[str]) -> List[str]:
    # Prefer chat template when available (matches `chat_with_model.py`).
    if getattr(tokenizer, "chat_template", None):
        rendered: List[str] = []
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            rendered.append(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        return rendered
    return prompts


@torch.no_grad()
def _generate_batch(
    model,
    tokenizer,
    prompts: List[str],
    *,
    max_prompt_length: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    compute_token_logprobs: bool,
) -> Tuple[List[str], Optional[List[float]]]:
    texts = _format_inputs(tokenizer, prompts)
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_prompt_length,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    generate_kwargs: Dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
    }
    if do_sample:
        generate_kwargs.update({"temperature": temperature, "top_p": top_p})

    if compute_token_logprobs:
        generate_kwargs.update({"return_dict_in_generate": True, "output_scores": True})
        out = model.generate(**inputs, **generate_kwargs)
        sequences = out.sequences
        scores = out.scores  # list[tensor(batch, vocab)]
    else:
        sequences = model.generate(**inputs, **generate_kwargs)
        scores = None

    # Per-sample input lengths (ignore padding)
    input_lens = inputs["attention_mask"].sum(dim=1).tolist()

    decoded: List[str] = []
    token_logprobs: Optional[List[float]] = [] if compute_token_logprobs else None

    for i in range(sequences.shape[0]):
        gen_ids = sequences[i, int(input_lens[i]) :]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        decoded.append(text)

    if compute_token_logprobs and scores is not None:
        # Align generated token ids per step.
        # scores[t] corresponds to the logits for token at time-step t.
        # generated token ids are the tokens appended after the prompt.
        gen_token_ids = []
        for i in range(sequences.shape[0]):
            gen_token_ids.append(sequences[i, int(input_lens[i]) : int(input_lens[i]) + len(scores)])
        gen_token_ids_t = torch.stack(gen_token_ids, dim=0)  # (batch, steps)

        # Compute mean log-prob per generated token for each sample.
        logp_per_token = []
        for t, step_scores in enumerate(scores):
            step_logp = torch.log_softmax(step_scores.float(), dim=-1)  # (batch, vocab)
            token_ids = gen_token_ids_t[:, t].unsqueeze(1)  # (batch, 1)
            chosen = step_logp.gather(1, token_ids).squeeze(1)  # (batch,)
            logp_per_token.append(chosen)
        logp_per_token_t = torch.stack(logp_per_token, dim=1)  # (batch, steps)

        # Ignore padding-like zero tokens if any (can happen if generation ended early).
        # We treat negative/positive logps equally; just mask tokens == 0 beyond EOS.
        mask = (gen_token_ids_t != 0).float()
        denom = mask.sum(dim=1).clamp_min(1.0)
        mean_logp = (logp_per_token_t * mask).sum(dim=1) / denom
        token_logprobs = mean_logp.detach().cpu().tolist()

    return decoded, token_logprobs


def _resolve_classifier_files(classifier_path: Path) -> Tuple[Path, Path]:
    """Return (checkpoint_pt, label_map_json)."""

    if classifier_path.is_dir():
        ckpt = classifier_path / "best_model.pt"
        label_map = classifier_path / "label_map.json"
        if not ckpt.exists():
            raise FileNotFoundError(f"Classifier dir missing best_model.pt: {classifier_path}")
        if not label_map.exists():
            raise FileNotFoundError(f"Classifier dir missing label_map.json: {classifier_path}")
        return ckpt, label_map

    if classifier_path.is_file():
        ckpt = classifier_path
        label_map = classifier_path.parent / "label_map.json"
        if not label_map.exists():
            raise FileNotFoundError(
                f"label_map.json not found next to classifier checkpoint: {label_map}. "
                "Pass a directory created by 06_train_classifier.py or place label_map.json alongside the .pt file."
            )
        return ckpt, label_map

    raise FileNotFoundError(f"classifier_path not found: {classifier_path}")


def _invert_label_map(label_map: Dict[str, int]) -> List[str]:
    # Ensure stable index order.
    id2label = [None] * (max(label_map.values()) + 1)
    for k, v in label_map.items():
        if v < 0:
            raise ValueError(f"Invalid label index for {k}: {v}")
        if v >= len(id2label):
            id2label.extend([None] * (v - len(id2label) + 1))
        id2label[v] = k
    if any(x is None for x in id2label):
        raise ValueError(f"label_map indices are not contiguous: {label_map}")
    return id2label  # type: ignore[return-value]


@torch.no_grad()
def _classifier_probs(
    classifier_model,
    classifier_tokenizer,
    texts: List[str],
    *,
    device: torch.device,
    batch_size: int,
    max_length: int = 512,
) -> np.ndarray:
    """Return probs for each sample: shape (n, num_labels)."""

    probs_list: List[np.ndarray] = []
    for batch in _chunked(texts, batch_size):
        inputs = classifier_tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        logits = classifier_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        # `SourceLanguageClassifier.forward` returns logits tensor (batch, num_labels).
        probs = torch.softmax(logits.float(), dim=-1).detach().cpu().numpy()
        probs_list.append(probs)

    return np.concatenate(probs_list, axis=0) if probs_list else np.zeros((0, 0), dtype=np.float32)


def _maybe_set_pad_token(tokenizer) -> None:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


def _load_translation_model(
    *,
    base_model: str,
    rl_model: Optional[str],
    torch_dtype: str,
) -> Tuple[Any, Any, str]:
    """Load a translation model (+ tokenizer). Returns (model, tokenizer, model_kind)."""

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    _maybe_set_pad_token(tokenizer)

    dtype = torch_dtype
    if dtype == "auto":
        dtype_arg = "auto"
    elif dtype == "float16":
        dtype_arg = torch.float16
    elif dtype == "bfloat16":
        dtype_arg = torch.bfloat16
    else:
        raise ValueError("--torch_dtype must be one of: auto|float16|bfloat16")

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        torch_dtype=dtype_arg,
        trust_remote_code=True,
    )

    if rl_model is None:
        return model, tokenizer, "base"

    rl_path = Path(rl_model)
    if rl_path.exists() and _is_peft_adapter_dir(rl_path):
        if PeftModel is None:
            raise RuntimeError("peft is not available but rl_model looks like an adapter directory")
        model = PeftModel.from_pretrained(model, rl_model)
        return model, tokenizer, "peft_adapter"

    # Otherwise treat as a standalone full model path or HF id.
    model = AutoModelForCausalLM.from_pretrained(
        rl_model,
        device_map="auto",
        torch_dtype=dtype_arg,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(rl_model, trust_remote_code=True, fix_mistral_regex=True)
    _maybe_set_pad_token(tokenizer)
    return model, tokenizer, "full_model"


def _evaluate_one_model(
    *,
    name: str,
    model,
    tokenizer,
    src_texts: List[str],
    classifier_model,
    classifier_tokenizer,
    classifier_device: torch.device,
    gen_batch_size: int,
    cls_batch_size: int,
    max_prompt_length: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    compute_token_logprobs: bool,
    output_jsonl: Optional[Path],
    id2label: List[str],
) -> Dict[str, Any]:
    num_labels = len(id2label)
    prob_sum = np.zeros((num_labels,), dtype=np.float64)
    n_total = 0

    mean_logp_sum = 0.0
    mean_logp_count = 0

    writer = None
    if output_jsonl is not None:
        output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        writer = output_jsonl.open("w", encoding="utf-8")

    try:
        for batch_idx, batch_src in enumerate(
            tqdm(list(_chunked(src_texts, gen_batch_size)), desc=f"Translating ({name})")
        ):
            prompts = [_build_prompt(x) for x in batch_src]
            translations, token_logprobs = _generate_batch(
                model,
                tokenizer,
                prompts,
                max_prompt_length=max_prompt_length,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                compute_token_logprobs=compute_token_logprobs,
            )

            probs = _classifier_probs(
                classifier_model,
                classifier_tokenizer,
                translations,
                device=classifier_device,
                batch_size=cls_batch_size,
            )
            prob_sum += probs.sum(axis=0)
            n_total += probs.shape[0]

            if token_logprobs is not None:
                mean_logp_sum += float(np.sum(token_logprobs))
                mean_logp_count += len(token_logprobs)

            if writer is not None:
                for i, (src, mt) in enumerate(zip(batch_src, translations)):
                    rec: Dict[str, Any] = {
                        "src_text": src,
                        "translation": mt,
                        "classifier_probs": {id2label[j]: float(probs[i, j]) for j in range(num_labels)},
                    }
                    if token_logprobs is not None:
                        rec["mean_token_logprob"] = float(token_logprobs[i])
                    writer.write(json.dumps(rec, ensure_ascii=False) + "\n")

    finally:
        if writer is not None:
            writer.close()

    avg_probs = (prob_sum / max(n_total, 1)).tolist()
    avg_probs_dict = {id2label[i]: float(avg_probs[i]) for i in range(num_labels)}

    result: Dict[str, Any] = {
        "num_samples": int(n_total),
        "avg_language_probs": avg_probs_dict,
    }
    if compute_token_logprobs:
        result["avg_mean_token_logprob"] = float(mean_logp_sum / max(mean_logp_count, 1))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare base vs RL translations and report classifier language distributions"
    )
    parser.add_argument("--base_model", type=str, required=True, help="Base LLM path or HF id")
    parser.add_argument(
        "--rl_model",
        type=str,
        required=True,
        help="RL model path (either a PEFT adapter dir with adapter_config.json, or a full model path/id)",
    )
    parser.add_argument("--input_data", type=str, required=True, help="Dataset file (.jsonl/.json/.csv)")
    parser.add_argument(
        "--classifier_path",
        type=str,
        required=True,
        help="Classifier checkpoint (.pt) or directory containing best_model.pt + label_map.json",
    )
    parser.add_argument(
        "--classifier_backbone",
        type=str,
        default="models/hf_backbones/chinese-roberta-wwm-ext",
        help="Backbone name/id used by the classifier architecture",
    )
    parser.add_argument("--max_samples", type=int, default=0, help="If >0, evaluate only first N")
    parser.add_argument("--gen_batch_size", type=int, default=1)
    parser.add_argument("--cls_batch_size", type=int, default=32)
    parser.add_argument("--max_prompt_length", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument(
        "--compute_token_logprobs",
        action="store_true",
        help="Also compute mean per-token logprob for each generated translation (slower)",
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="auto",
        help="Translation model dtype: auto|float16|bfloat16",
    )
    parser.add_argument("--output_json", type=str, default="", help="Write summary JSON to this path")
    parser.add_argument(
        "--output_jsonl",
        type=str,
        default="",
        help="Write per-sample records as JSONL (will be overwritten)",
    )

    args = parser.parse_args()

    input_path = Path(args.input_data)
    records = _load_records(input_path)
    if args.max_samples and args.max_samples > 0:
        records = records[: args.max_samples]
    if not records:
        raise ValueError(f"No records loaded from: {input_path}")

    if "src_text" not in records[0]:
        raise ValueError("input_data records must contain a 'src_text' field")

    src_texts = [str(r.get("src_text", "")) for r in records]

    # Load classifier (from this repo's scripts/ folder)
    import sys

    sys.path.append(str(Path(__file__).parent))
    from classifier_model import SourceLanguageClassifier  # type: ignore

    ckpt_path, label_map_path = _resolve_classifier_files(Path(args.classifier_path))
    with label_map_path.open("r", encoding="utf-8") as f:
        label_map = json.load(f)
    id2label = _invert_label_map(label_map)

    classifier_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier_model = SourceLanguageClassifier(
        num_languages=len(label_map),
        model_name=args.classifier_backbone,
        hidden_dim=512,
    ).to(classifier_device)
    classifier_state = torch.load(ckpt_path, map_location="cpu")
    classifier_model.load_state_dict(classifier_state)
    classifier_model.eval()
    classifier_tokenizer = classifier_model.get_tokenizer()

    # Evaluate base model (no adapter)
    print("\n=== Loading base model ===")
    base_model, base_tokenizer, base_kind = _load_translation_model(
        base_model=args.base_model,
        rl_model=None,
        torch_dtype=args.torch_dtype,
    )
    base_model.eval()
    print(f"Base model kind: {base_kind}")

    per_sample_jsonl = Path(args.output_jsonl) if args.output_jsonl else None
    base_jsonl = None
    rl_jsonl = None
    if per_sample_jsonl is not None:
        base_jsonl = per_sample_jsonl.with_name(per_sample_jsonl.stem + ".base.jsonl")
        rl_jsonl = per_sample_jsonl.with_name(per_sample_jsonl.stem + ".rl.jsonl")

    base_result = _evaluate_one_model(
        name="base",
        model=base_model,
        tokenizer=base_tokenizer,
        src_texts=src_texts,
        classifier_model=classifier_model,
        classifier_tokenizer=classifier_tokenizer,
        classifier_device=classifier_device,
        gen_batch_size=args.gen_batch_size,
        cls_batch_size=args.cls_batch_size,
        max_prompt_length=args.max_prompt_length,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        compute_token_logprobs=args.compute_token_logprobs,
        output_jsonl=base_jsonl,
        id2label=id2label,
    )

    # Free base model before loading RL model to avoid OOM
    del base_model
    torch.cuda.empty_cache()

    # Evaluate RL model
    print("\n=== Loading RL model ===")
    rl_model, rl_tokenizer, rl_kind = _load_translation_model(
        base_model=args.base_model,
        rl_model=args.rl_model,
        torch_dtype=args.torch_dtype,
    )
    rl_model.eval()
    print(f"RL model kind: {rl_kind}")

    rl_result = _evaluate_one_model(
        name="rl",
        model=rl_model,
        tokenizer=rl_tokenizer,
        src_texts=src_texts,
        classifier_model=classifier_model,
        classifier_tokenizer=classifier_tokenizer,
        classifier_device=classifier_device,
        gen_batch_size=args.gen_batch_size,
        cls_batch_size=args.cls_batch_size,
        max_prompt_length=args.max_prompt_length,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        compute_token_logprobs=args.compute_token_logprobs,
        output_jsonl=rl_jsonl,
        id2label=id2label,
    )

    summary: Dict[str, Any] = {
        "input_data": str(input_path),
        "num_samples": len(src_texts),
        "prompt": "Translate the following japanese text to Chinese:\n\n{src_text}",
        "base": base_result,
        "rl": rl_result,
        "labels": id2label,
    }

    print("\n=== Average language probability distribution ===")
    print("\n[Base]")
    for k, v in sorted(base_result["avg_language_probs"].items(), key=lambda kv: -kv[1]):
        print(f"{k}: {v:.6f}")

    print("\n[RL]")
    for k, v in sorted(rl_result["avg_language_probs"].items(), key=lambda kv: -kv[1]):
        print(f"{k}: {v:.6f}")

    if args.compute_token_logprobs:
        print("\n=== Mean generation token logprob (avg over samples) ===")
        print(f"Base: {base_result.get('avg_mean_token_logprob', float('nan')):.6f}")
        print(f"RL:   {rl_result.get('avg_mean_token_logprob', float('nan')):.6f}")

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\nWrote summary JSON: {out_path}")

    if per_sample_jsonl is not None:
        print(f"Wrote per-sample JSONL: {base_jsonl}")
        print(f"Wrote per-sample JSONL: {rl_jsonl}")


if __name__ == "__main__":
    main()
