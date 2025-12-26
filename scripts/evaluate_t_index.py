import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm
import os

# Paths
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(project_root, "models/hf_backbones/Qwen3-4B-Instruct-2507")
input_file = os.path.join(project_root, "t_index/data/wild/pointwise.jsonl")
output_file = os.path.join(project_root, "t_index/data/wild/qwen.jsonl")

def main():
    # Load model and tokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        device_map="auto", 
        torch_dtype="auto"
    )

    # Read input
    print(f"Reading input from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f"Translating {len(lines)} items...")
    
    # Ensure output directory exists if it has a path
    if os.path.dirname(output_file):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f_out:
        for line in tqdm(lines):
            try:
                data = json.loads(line)
                source = data.get('source', '')
                
                prompt = f"Translate the following english text to Chinese:\n\n{source}"
                messages = [
                    {"role": "user", "content": prompt}
                ]
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

                with torch.no_grad():
                    generated_ids = model.generate(
                        **model_inputs,
                        max_new_tokens=512,
                        do_sample=False
                    )
                
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                ]
                translation = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                
                output_record = {
                    "source": source,
                    "translation": translation.strip()
                }
                f_out.write(json.dumps(output_record, ensure_ascii=False) + '\n')
            except Exception as e:
                print(f"Error processing line: {e}")
                continue

    print(f"Done. Saved to {output_file}")

if __name__ == "__main__":
    main()
