"""
Utility functions for model operations.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from pathlib import Path
from typing import List, Optional, Dict
import json


def load_model_and_tokenizer(base_model: str,
                             adapter_path: Optional[str] = None,
                             device_map: str = "auto",
                             torch_dtype = torch.float16):
    """
    Load model and tokenizer with optional adapter.
    
    Args:
        base_model: Base model name or path
        adapter_path: Path to LoRA adapter (optional)
        device_map: Device map for model loading
        torch_dtype: Torch dtype for model
    
    Returns:
        Tuple of (model, tokenizer)
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch_dtype,
        device_map=device_map
    )
    
    # Load adapter if provided
    if adapter_path and Path(adapter_path).exists():
        model = PeftModel.from_pretrained(model, adapter_path)
    
    model.eval()
    
    return model, tokenizer


def generate_batch(model,
                  tokenizer,
                  prompts: List[str],
                  max_new_tokens: int = 512,
                  num_beams: int = 1,
                  do_sample: bool = False,
                  temperature: float = 1.0,
                  top_p: float = 1.0) -> List[str]:
    """
    Generate translations for a batch of prompts.
    
    Args:
        model: The model to use
        tokenizer: The tokenizer
        prompts: List of prompts
        max_new_tokens: Maximum new tokens to generate
        num_beams: Number of beams for beam search
        do_sample: Whether to use sampling
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
    
    Returns:
        List of generated texts
    """
    # Tokenize
    inputs = tokenizer(
        prompts,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=512
    ).to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            top_p=top_p if do_sample else 1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode
    generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    # Extract only the generated part (after prompt)
    results = []
    for prompt, full_output in zip(prompts, generated_texts):
        # Remove prompt from output
        if full_output.startswith(prompt):
            generated = full_output[len(prompt):].strip()
        else:
            generated = full_output.strip()
        results.append(generated)
    
    return results


def save_model_with_metadata(model,
                             tokenizer,
                             output_dir: Path,
                             metadata: Dict):
    """
    Save model, tokenizer, and metadata.
    
    Args:
        model: Model to save
        tokenizer: Tokenizer to save
        output_dir: Output directory
        metadata: Metadata dictionary to save
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    if hasattr(model, 'save_pretrained'):
        model.save_pretrained(output_dir)
    else:
        torch.save(model.state_dict(), output_dir / "model.pt")
    
    # Save tokenizer
    tokenizer.save_pretrained(output_dir)
    
    # Save metadata
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)


def count_parameters(model) -> Dict[str, int]:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
    
    Returns:
        Dictionary with parameter counts
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total,
        'trainable': trainable,
        'frozen': total - trainable,
        'trainable_percentage': (trainable / total * 100) if total > 0 else 0
    }


def get_model_device(model) -> torch.device:
    """Get the device of a model."""
    return next(model.parameters()).device


def clear_gpu_memory():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
