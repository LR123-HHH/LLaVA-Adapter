# LLaVA-Sentiment

Multimodal (image + text) sentiment analysis built on LLaVA-v1.5. Three modules are added on top of the original LLaVA stack:

- **SAPP** (Semantically-aware Adaptive Patch Partitioning): SAM masks guide CLIP patch merging.
- **Bi-CMAA** (Bidirectional Cross-Modal Attention Alignment): bidirectional cross-attention between vision and text tokens.
- **DHCCA** (Dual-Head Confidence-Calibrated Aggregation): classification head + generation head, fused dynamically with confidence threshold τ.

Pipeline: `Image → CLIP → SAPP → Bi-CMAA → LLaMA → gen head + cls head → DHCCA fusion`

---

## 1. Project layout


```
LLaVA-sentiment/
├── LLaVA-main/                 # Modified LLaVA source + training + eval
│   ├── llava/model/llava_arch_new_v01.py   # SAPP + Bi-CMAA
│   ├── llava/model/language_model/llava_llama.py  # DHCCA cls head + joint loss
│   ├── llava/train/train_mem.py            # Training entry
│   ├── llava/eval/eval_llava_from_file*.py # Eval scripts
│   └── scripts/v1_5/                       # Training shell scripts
├── notebooks/                  # Dataset config/utils + Bi-CMAA attention comparison
├── notebooks2/                 # Paper experiments (SAPP/DHCCA/Bi-CMAA ablations, significance)
├── datasets/                   # Four datasets (extract from datasets.zip)
├── checkpoints/                # SAM weights (sam_vit_h_4b8939.pth, etc.)
├── clip-vit-large-patch14-336/ # CLIP vision tower
└── requirements.txt
```

---

## 2. Environment setup

```bash
# 1. Create env (Python 3.10 recommended; CUDA 11.8 / 12.1)
conda create -n llava-sent python=3.10 -y
conda activate llava-sent

# 2. Install dependencies (pick one)
pip install -r requirements.txt
#   or official LLaVA install (includes [train])
cd LLaVA-main && pip install -e ".[train]" && cd ..
```

**Models / weights you need on disk:**

| Asset | Location in repo |
|-------|------------------|
| Base LLaVA | `llava-v1.5-7b/` or `llava-7b/` |
| Vision tower | `clip-vit-large-patch14-336/` |
| SAM | `checkpoints/sam_vit_h_4b8939.pth` |

---

## 3. Data preparation

```bash
# From project root — unpack into datasets/
unzip datasets.zip -d /path/to/LLaVA-sentiment/
```

Each sample is LLaVA instruction-tuning JSON:

```json
{
  "id": "uuid",
  "image": "MVSA/data/22152.jpg",
  "conversations": [
    {"from": "human", "value": "<image> Provide a sentiment analysis (Positive, Negative, Neutral) for the provided image. Reference content: <tweet text>"},
    {"from": "gpt", "value": "Positive"}
  ]
}
```

| Dataset | JSON splits | Image directory |
|---------|-------------|-----------------|
| MVSA-Single | `datasets/MVSA-Single/{train_split,val,test}.json` | `datasets/MVSA-Single/data/` |
| MVSA-Multiple | `datasets/MVSA-Multiple/{train_split,val,test}.json` | `datasets/MVSA-Multiple/data/` |
| TWITTER-15 | `LLaVA-main/playground/data/t15_{train_split,val,test}.json` | `LLaVA-main/playground/data/` |
| TWITTER-17 | `LLaVA-main/playground/data/t17_{train_split,val,test}.json` | `LLaVA-main/playground/data/` |

Label mapping: `negative=0`, `neutral=1`, `positive=2`

---

## 4. Training

> **Path note:** Some `scripts/v1_5/*.sh` files hard-code `/root/autodl-tmp/LLaVA-main/`. This repo uses `LLaVA-sentiment/LLaVA-main/`. Edit `--model_name_or_path`, `--data_path`, `--vision_tower`, `--output_dir`, etc. before running.

### Route A: Generative fine-tuning (DeepSpeed)

```bash
cd LLaVA-main

# Full fine-tuning
bash scripts/v1_5/finetune_task_t15.sh          # TWITTER-15
bash scripts/v1_5/finetune_task_t17.sh          # TWITTER-17

# LoRA (r=128)
bash scripts/v1_5/finetune_task_lora_t15.sh
bash scripts/v1_5/finetune_task_lora_t17.sh
bash scripts/v1_5/finetune_task_lora_mvsa_multi.sh   # MVSA
```

Entry: `llava/train/train_mem.py` → `llava/train/train.py`, DeepSpeed config `scripts/zero3.json`.  
Checkpoints default to `checkpoints/llava-v1.5-7b-sentiment[-lora]`.

## 5. Evaluation

```bash
cd LLaVA-main

# Full model
python llava/eval/eval_llava_from_file_t15.py \
  --input-file playground/data/t15_test.json \
  --output-file outputs/t15_pred.json \
  --model-path checkpoints/llava-v1.5-7b-sentiment

# LoRA (base + adapter)
python llava/eval/eval_llava_lora_MVSA.py \
  --input-file playground/data/MVSA.json \
  --model-path ../llava-7b \
  --lora-path checkpoints/llava-v1.5-7b-sentiment-lora
```

| Script | Dataset |
|--------|---------|
| `eval_llava_from_file_t15.py` | TWITTER-15 |
| `eval_llava_from_file_t17.py` | TWITTER-17 |
| `eval_llava_from_file_MVSA.py` | MVSA |
| `eval_llava_lora_MVSA.py` | MVSA (LoRA) |
| `convert_lora_to_full.py` | Merge LoRA into full weights |

Flow: load JSON → `model.generate()` → parse Positive/Negative/Neutral → write predictions.  
Paper metrics (including DHCCA dual-head fusion) are computed in `notebooks2/exp3`–`exp7`.

---

## 6. Quick reference

| Task | Command / location |
|------|-------------------|
| Install | `pip install -r requirements.txt` |
| Data | `unzip datasets.zip` at repo root |
| Full fine-tuning | `LLaVA-main/scripts/v1_5/finetune_task_*.sh` |
| LoRA fine-tuning | `LLaVA-main/scripts/v1_5/finetune_task_lora_*.sh` |
| Evaluation | `LLaVA-main/llava/eval/` |
| Optional paper notebooks | `notebooks/`, `notebooks2/` |
