# Translation Scripts Usage


## 批量翻译所有池

翻译所有三个池（a, b, c）使用两个模型：

```bash
/root/WAYF/batch_translate.sh
```

这将创建以下目录结构：

```
WAYF/data/translations/
├── DeepSeek-V3.2/
│   ├── pool_a_zh.jsonl
│   ├── pool_b_zh.jsonl
│   └── pool_c_zh.jsonl
└── Qwen3-Next-80B-A3B-Instruct/
    ├── pool_a_zh.jsonl
    ├── pool_b_zh.jsonl
    └── pool_c_zh.jsonl
```

## 单独翻译某个池

```bash
python3 /root/WAYF/scripts/02_translate_with_llm.py \
    --input_file /root/WAYF/data/pools/pool_a.jsonl \
    --output_file /root/WAYF/data/translations/DeepSeek-V3.2/pool_a_zh.jsonl \
    --model_name "DeepSeek-V3.2" \
    --pool_name "a"
```
