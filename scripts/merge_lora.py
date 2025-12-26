import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter with base model")
    parser.add_argument("--base_model", type=str, required=True, help="Path to base model")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to LoRA adapter")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save merged model")
    parser.add_argument("--sft_adapter", type=str, default=None, help="Path to SFT adapter if RL was trained on top of SFT")
    
    args = parser.parse_args()
    
    print(f"Loading base model from {args.base_model}")
    # Use bfloat16 if available (Ampere+), else float16
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    
    if args.sft_adapter:
        print(f"Loading and merging SFT adapter from {args.sft_adapter}")
        model = PeftModel.from_pretrained(model, args.sft_adapter)
        model = model.merge_and_unload()
        
    print(f"Loading and merging RL adapter from {args.adapter_path}")
    model = PeftModel.from_pretrained(model, args.adapter_path)
    model = model.merge_and_unload()
    
    print(f"Saving merged model to {args.output_path}")
    model.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)
    print("Done!")

if __name__ == "__main__":
    main()
