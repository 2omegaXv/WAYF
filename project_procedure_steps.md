# Project Procedure (Step-by-Step)

## 1) Data Preparing (classification, RL, validation)

### 1.1 Choose languages + sampling
- Select K source languages from XL-Sum.
- Choose source field: start with `text`.
- Balance per language: equal N samples/language.
- Length matching: bin by length and sample evenly.
- Deduplicate:
  - Exact duplicates by `hash(src_text)`.
  - Near-duplicates via MinHash/SimHash or embedding cosine.

### 1.2 Create three disjoint pools (no overlap)
- Split by XL-Sum `id` (or `hash(src_text)` if LLM-generated):
  - **Pool A (Classifier)**: for training source-language classifier.
  - **Pool B (RL Prompts)**: used only as prompts during RL.
  - **Pool C (Validation/Eval)**: final reporting set.

### 1.3 LLM API translation dataset (JSONL)
- Use one fixed LLM API configuration for translation dataset generation:
  - fixed model name
  - fixed system prompt
  - deterministic decoding (e.g., `temperature=0`, fixed `top_p`)
- For each XL-Sum example, request strict JSON output and store one JSON object per line.

#### 1.3.1 JSON schema (minimum)
- `id`
- `split` (pre-decided: `train`/`valid`/`test`)
- `src_lang`
- `src_text`
- `zh_mt`
- `audit`:
  - `model`
  - decoding params (`temperature`, `top_p`, `max_tokens`, etc.)
  - `prompt_version`
  - `raw_response_hash`

#### 1.3.2 Prompt constraints (must include)
- Output Chinese only.
- Return JSON only (no extra text).
- Rewrite/remove URLs, emails, @handles.
- Avoid copying Latin/Cyrillic strings; paraphrase if needed.
- Normalize punctuation to Chinese conventions.

#### 1.3.3 Post-filters (apply to `zh_mt`)
- Strip whitespace and surrounding quotes.
- Remove remaining URLs.
- Unicode normalization for punctuation.
- Reject samples with high non-CJK ratio.

### 1.4 RL reference dataset for SFT (optional but recommended)
- Generate `zh_ref` using LLM API with a stricter “high-quality translation” prompt.
- Store SFT pairs:
  - `prompt` (translation instruction + `src_lang` + `src_text`)
  - `response` (`zh_ref`)
  - metadata: `id`, `src_lang`, `src_text`

### 1.5 RL dataset formats (prepare all needed)
- **On-policy RL (GRPO/PPO)** JSONL:
  - `prompt`
  - metadata: `id`, `src_lang`, `src_text` (optional), `zh_ref` (optional)
- **Preference RL (DPO-from-reward)** JSONL:
  - `prompt`
  - `chosen`
  - `rejected`


## 2) Classifier Training and Evaluation

### 2.1 Model + components
- Backbone encoder: `hfl/chinese-roberta-wwm-ext`
- Tokenizer: `AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")`
- Backbone: `AutoModel.from_pretrained("hfl/chinese-roberta-wwm-ext")`
- Feature: `[CLS]` hidden state
  - `cls = last_hidden_state[:, 0, :]`
- Classification head (MLP):
  - `Dropout(0.1)`
  - `Linear(hidden_size → 512)`
  - `GELU`
  - `Dropout(0.1)`
  - `Linear(512 → num_languages)`
- Loss: `CrossEntropyLoss`

### 2.2 Trainer integration (custom module)
- Implement a `torch.nn.Module` with `forward(input_ids, attention_mask, labels=None, ...)`.
- Return `loss` and `logits` (dict or `ModelOutput`).
- Train on Pool A (Classifier) using `zh_mt` → `src_lang` labels.

### 2.3 Baselines (train + report)
- Majority baseline.
- Stratified random baseline.
- Character n-gram (3–6) + Logistic Regression or Linear SVM.
- fastText supervised (char n-grams).
- Frozen backbone + linear head.

### 2.4 Metrics + artifacts
- Metrics on classifier valid/test:
  - Accuracy
  - Macro-F1
  - AUC (one-vs-rest)
- Save:
  - trained reward classifier checkpoint
  - tokenizer
  - label mapping


## 3) RL

### 3.1 Policy model
- Translator model: `Qwen/Qwen3-4B-Instruct-2507`
- Finetuning method: LoRA or QLoRA (`peft`, optional `bitsandbytes`).

### 3.2 Prompt template (fixed)
- Build one prompt template for all RL runs:
  - includes `src_lang`
  - includes `src_text`
  - instructs “output Chinese translation only”

### 3.3 Warm start (SFT)
- Train on SFT dataset (`prompt` → `zh_ref`) using LoRA/QLoRA.
- Save SFT checkpoint (this is the RL reference policy).

### 3.4 Reward model
- Use the trained classifier as reward model.
- For a generated Chinese translation `zh_out` with true source label `y*`:
  - `p = softmax(classifier_logits(zh_out))`
  - `r_cls = -log(max(ε, p[y*]))`

### 3.5 RL objective
- Total reward:
  - `R = w_cls * r_cls + w_qual * r_qual - w_kl * KL(policy || ref)`
- Choose `r_qual` (one):
  - reference similarity (chrF/BLEU vs `zh_ref`) or
  - quality scoring model (optional)

### 3.6 RL training data usage (avoid bias)
- Use Pool B (RL Prompts) only.
- Do not use Pool A samples as RL prompts.

### 3.7 TRL training modes
- **GRPO**:
  - for each `prompt`, sample K candidates
  - compute rewards
  - update policy
- **DPO-from-reward (offline alternative)**:
  - generate K candidates per prompt
  - compute `R`
  - construct `chosen/rejected` pairs
  - run DPO on preference JSONL
- **PPO (fallback)**:
  - on-policy rollouts
  - reward + KL constraint


## 4) Evaluate (Improvement After RL)

### 4.1 Translation quality (before vs after RL)
- Evaluate on Pool C (Validation/Eval) and/or FLORES-200 (for selected languages).
- Metrics:
  - chrF
  - BLEU
  - COMET (optional)
- Report:
  - SFT baseline scores
  - RL model scores

### 4.2 Translationese reduction (before vs after RL)
- Train an **independent probe classifier** (different seed and ideally different backbone) on Pool A translations.
- Run probe on:
  - SFT model outputs on Pool C
  - RL model outputs on Pool C
- Metrics:
  - Probe Accuracy / Macro-F1 / AUC
- Report:
  - Probe performance decrease after RL

### 4.3 Tradeoff sweep
- Sweep `w_cls` and log per setting:
  - quality metrics (chrF/BLEU/COMET)
  - translationese metrics (probe AUC/accuracy)
- Select final checkpoint based on the sweep results.
