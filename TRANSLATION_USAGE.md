# Translation Scripts Usage

## 快速测试（推荐先运行）

测试3个样本，验证两个模型都能正常工作：

```bash
/root/WAYF/test_translation.sh
```

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

## 数据统计

查看翻译进度：

```bash
echo "DeepSeek-V3.2:"
wc -l /root/WAYF/data/translations/DeepSeek-V3.2/*.jsonl

echo -e "\nQwen3-Next-80B-A3B-Instruct:"
wc -l /root/WAYF/data/translations/Qwen3-Next-80B-A3B-Instruct/*.jsonl
```

## 注意事项

1. **错误重试**：API失败会自动重试3次，间隔递增（5s, 10s, 15s）
2. **JSON解析**：自动处理markdown代码块格式
3. **数据分离**：两个模型的翻译结果存储在不同目录
4. **进度跟踪**：使用tqdm显示翻译进度
5. **质量过滤**：自动过滤CJK字符比例低于70%的翻译

## 预计时间

- 每个池约2800条（pool_a）或2100条（pool_b/c）
- 每条翻译约2-5秒
- Pool A: 约2-4小时
- Pool B/C: 约1.5-3小时每个
- 总计：约10-20小时（取决于API速度）
