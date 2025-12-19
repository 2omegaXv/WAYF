#!/bin/bash
# Batch translation script for all pools with both models

SCRIPT_DIR="/root/WAYF/scripts"
DATA_DIR="/root/WAYF/data"

API_KEYS="sk-urQ1kd8QaBdMIrnwW8_0Xw,sk-zGRy0CJMC1mpIooqyIstfQ"
BASE_URL="https://llmapi.paratera.com"
NUM_WORKERS=4  # 并行工作线程数

echo "========================================"
echo "WAYF Pool Translation - Batch Processing"
echo "========================================"
echo ""

# Function to translate with a specific model
translate_with_model() {
    local model=$1
    local pool=$2
    
    # Create model-specific directory
    local model_dir="$DATA_DIR/translations/${model}"
    mkdir -p "$model_dir"
    
    local input_file="$DATA_DIR/pools/pool_${pool}.jsonl"
    local output_file="${model_dir}/pool_${pool}_zh.jsonl"
    
    echo "--------------------------------------"
    echo "Translating pool_${pool} with ${model}"
    echo "--------------------------------------"
    echo "Input:  $input_file"
    echo "Output: $output_file"
    echo ""
    
    python3 "$SCRIPT_DIR/02_translate_with_llm.py" \
        --input_file "$input_file" \
        --output_file "$output_file" \
        --api_keys "$API_KEYS" \
        --base_url "$BASE_URL" \
        --model_name "$model" \
        --temperature 0.0 \
        --top_p 1.0 \
        --max_tokens 2048 \
        --prompt_version "v1.0" \
        --field "src_text" \
        --min_cjk_ratio 0.3 \
        --pool_name "$pool" \
        --num_workers "$NUM_WORKERS"
    
    if [ $? -eq 0 ]; then
        echo "✓ Completed: pool_${pool} with ${model}"
        local count=$(wc -l < "$output_file" 2>/dev/null || echo "0")
        echo "  Translated entries: $count"
    else
        echo "✗ Failed: pool_${pool} with ${model}"
    fi
    echo ""
}

# Main processing
echo "Starting batch translation..."
echo ""

# Translate all pools with DeepSeek-V3.2
for pool in a b c; do
    translate_with_model "Qwen3-Next-80B-A3B-Instruct" "$pool"
    translate_with_model "DeepSeek-V3.2" "$pool"
done


echo ""
echo "========================================"
echo "Batch translation completed!"
echo "========================================"
echo ""
echo "Results:"
ls -lh "$DATA_DIR/translations/"*/*.jsonl 2>/dev/null || echo "No output files found"
