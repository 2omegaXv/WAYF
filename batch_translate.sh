#!/bin/bash
# Batch translation script for all pools with both models

SCRIPT_DIR="/root/WAYF/scripts"
BASE_DIR="/root/WAYF"
INPUT_DIR="$BASE_DIR/new_data/original"
OUTPUT_DIR="$BASE_DIR/new_data/translated"

API_KEYS=""
BASE_URL="https://llmapi.paratera.com"
NUM_WORKERS=800  # 并行工作线程数

echo "========================================"
echo "WAYF Pool Translation - Batch Processing"
echo "========================================"
echo ""

# Function to translate with a specific model
translate_file() {
    local model=$1
    local input_file=$2
    local filename=$(basename "$input_file")
    local basename="${filename%.*}"
    
    # Create model-specific directory
    local model_dir="$OUTPUT_DIR/${model}"
    mkdir -p "$model_dir"
    
    local output_file="${model_dir}/${basename}_zh.jsonl"
    
    echo "--------------------------------------"
    echo "Translating ${filename} with ${model}"
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
        --pool_name "${basename}" \
        --num_workers "$NUM_WORKERS"
    
    if [ $? -eq 0 ]; then
        echo "✓ Completed: ${filename} with ${model}"
        local count=$(wc -l < "$output_file" 2>/dev/null || echo "0")
        echo "  Translated entries: $count"
    else
        echo "✗ Failed: ${filename} with ${model}"
    fi
    echo ""
}

# Main processing
echo "Starting batch translation..."
echo ""

POOLS=("c" "b")
model="DeepSeek-V3.2"
# Languages to translate (excluding chinese)
LANGUAGES=("german" "japanese" "korean" "russian" "spanish" "french" "english")

for pool in "${POOLS[@]}"; do
    for lang in "${LANGUAGES[@]}"; do
        input_file="${INPUT_DIR}/${pool}_${lang}.jsonl"
        
        if [ ! -f "$input_file" ]; then
            echo "Skipping missing file: ${pool}_${lang}.jsonl"
            continue
        fi

        translate_file "$model" "$input_file"
    done
done

echo ""
echo "========================================"
echo "Batch translation completed!"
echo "========================================"
echo ""
echo "Results:"
ls -lh "$OUTPUT_DIR/"*/*.jsonl 2>/dev/null || echo "No output files found"
