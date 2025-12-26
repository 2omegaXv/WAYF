from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
from copy import deepcopy

def main():
    parser = argparse.ArgumentParser(description="Chat with a local model")
    parser.add_argument("--model_path", type=str, default="models/hf_backbones/Qwen3-4B-Instruct-2507", help="Path to the base model")
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, 
        device_map="auto", 
        torch_dtype="auto"
    )

    user_input = "Translate the following japanese text to Chinese:\n\n"

    messages = [{"role": "user", "content": user_input}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=1024,
    )
    output_ids = generated_ids[0][len(model_inputs["input_ids"][0]):].tolist()
    response = tokenizer.decode(output_ids, skip_special_tokens=True)
    print(f"\nModel: {response}\n")


if __name__ == "__main__":
    main()