# Download Instructions

---

## Step 1 — Download from Baidu Netdisk

> Link: https://pan.baidu.com/s/1rZY6cvW7UXHXXrxo6uygow Extraction code: sbxb

| File | Size | Description |
|---|---|---|
| `checkpoints.zip` | ~2.6 GB | SAM weights (ViT-H and ViT-B) |
| `datasets.zip` | ~2.4 GB | Four datasets: images + JSON + precomputed cache |
| `playground_data.tar.gz` | ~2.1 GB | Training images + JSON (for fine-tuning) |
| `llava-v1.5-7b-sentiment.tar.gz` | ~652 MB | Fine-tuned LoRA adapter (project output) |

---

## Step 2 — Download from HuggingFace

```bash
git lfs install

# Base model (~13 GB)
git clone https://huggingface.co/liuhaotian/llava-v1.5-7b

# CLIP vision tower (~1.7 GB)
git clone https://huggingface.co/openai/clip-vit-large-patch14-336
```

---

## Step 3 — Place Files

After downloading everything, organize files under the project root `LLaVA-sentiment/` as follows:

```
LLaVA-sentiment/
├── llava-7b/                        ← rename cloned llava-v1.5-7b to llava-7b
├── clip-vit-large-patch14-336/      ← cloned from HuggingFace
├── checkpoints/                     ← extracted from checkpoints.zip
│   ├── sam_vit_h_4b8939.pth
│   └── sam_vit_b_01ec64.pth
├── datasets/                        ← extracted from datasets.zip
│   ├── MVSA-Multiple/
│   ├── MVSA-Single/
│   ├── TWITTER-15/
│   └── TWITTER-17/
└── LLaVA-main/
    ├── checkpoints/
    │   └── llava-v1.5-7b-sentiment/ ← extracted from llava-v1.5-7b-sentiment.tar.gz
    └── playground/data/             ← extracted from playground_data.tar.gz
```

---

## Step 4 — Extract All Files

Run the following commands from the project root:

```bash
cd ~/autodl-tmp/LLaVA-sentiment

unzip checkpoints.zip
unzip datasets.zip -d datasets/
tar xzf llava-v1.5-7b-sentiment.tar.gz -C LLaVA-main/checkpoints/
tar xzf playground_data.tar.gz -C LLaVA-main/playground/data/

# Rename base model directory
mv llava-v1.5-7b llava-7b
```

---

## What Each File Is Used For

| File | Required for |
|---|---|
| `llava-7b/` | Base model (needed by LoRA) |
| `clip-vit-large-patch14-336/` | Visual encoder |
| `sam_vit_h_4b8939.pth` | SAPP module (inference + training) |
| `sam_vit_b_01ec64.pth` | SAM ViT-B (backup) |
| `llava-v1.5-7b-sentiment/` | Fine-tuned LoRA adapter (for evaluation) |
| `datasets/` | Evaluation datasets |
| `playground/data/` | Training datasets |

HuggingFace：
- llava-v1.5-7b: `https://huggingface.co/liuhaotian/llava-v1.5-7b`
- CLIP: `https://huggingface.co/openai/clip-vit-large-patch14-336`
- SAM: `https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth`
