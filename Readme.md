# English → Hindi Neural Machine Translator

A sequence-to-sequence Transformer trained from scratch on the [IITB English-Hindi dataset](https://huggingface.co/datasets/cfilt/iitb-english-hindi). Built in PyTorch with SentencePiece BPE tokenization and a Flask REST API backend.

---

## How it works

The model follows the encoder-decoder Transformer architecture from *Attention Is All You Need* (Vaswani et al., 2017). An English sentence is tokenized with SentencePiece BPE, passed through 6 encoder layers with multi-head self-attention, and then decoded token-by-token in Hindi using 6 decoder layers with cross-attention over the encoder output. Greedy decoding is used at inference time.

---

## Project structure

```
Translation Transformer/
├── frontend/               # HTML/CSS/JS frontend
│   ├── index.html
│   ├── index.css
│   └── main.js
├── model.py                # Transformer architecture
├── dataset.py              # BilingualDataset + causal mask
├── train.py                # Training loop
├── inference.py            # Inference + auto GDrive weight download
├── api_server.py           # Flask REST API
├── config.py               # Hyperparameters + checkpoint paths
├── tokenizer_en.model      # SentencePiece English tokenizer
├── tokenizer_hi.model      # SentencePiece Hindi tokenizer
└── requirements.txt
```

---

## Setup

```bash
git clone https://github.com/ershavait/en-hi-translator-transformer
cd en-hi-translator-transformer
pip install -r requirements.txt
```

Model weights (~786 MB) are stored on Google Drive and **download automatically** on first run. No manual step needed.

> Make sure you have ~1 GB free disk space before running.

---

## Running the app

**Start the backend:**
```bash
python api_server.py
```

On first startup it will download the weights from Google Drive, load the model, then serve at `http://localhost:5000`.

**Open the frontend:**

Just open `frontend/index.html` in your browser. The frontend hits the Flask API at port 5000.

---

## Training config

| Parameter | Value |
|---|---|
| Architecture | Encoder-Decoder Transformer |
| Layers (N) | 6 |
| Heads (h) | 8 |
| d_model | 512 |
| d_ff | 2048 |
| Vocab size | 16,000 (BPE) |
| seq_len | 150 |
| Dataset | IITB English-Hindi (5% sample ~330K pairs) |
| Epochs | 20 |
| Batch size | 32 |
| Optimizer | Adam (lr=1e-4, eps=1e-9) |
| Loss | CrossEntropy + label smoothing 0.1 |
| Hardware | Lightning AI T4 GPU |

---

## API

**POST** `/api/translate`

```json
// Request
{ "text": "The cat sat on the mat." }

// Response
{
  "source": "The cat sat on the mat.",
  "translation": "बिल्ली चटाई पर बैठी थी।",
  "time_ms": 312
}
```

**GET** `/api/health` — returns model load status and device.

---

## Model weights

Stored on Google Drive (too large for GitHub):
[Download weights](https://drive.google.com/drive/folders/1Q9jO0l9Kl1ikzI1aCIjxqrTWTgwSLcW3?usp=drive_link)

`inference.py` handles the download automatically via `gdown`. If you want to download manually, drop the `.pt` file into `cfilt/iitb-english-hindi_weights/`.

---

## Requirements

```
torch
sentencepiece
datasets
flask
flask-cors
torchmetrics
tensorboard
tqdm
gdown
```

---

## Reference

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). **Attention Is All You Need**. *Advances in Neural Information Processing Systems*, 30.

Paper: [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
