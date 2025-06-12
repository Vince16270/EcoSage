<h1 align="center">EcoSage</h1>
<h3 align="center">LLM-Powered Chatbot for Sustainable Policy</h3>

<p align="center">
  <img src="frontend/logo_full.png" alt="EcoSage banner" width="400">
</p>

EcoSage makes European and Dutch energy-transition policy understandable for everyone. It ingests official PDF documents, applies state-of-the-art Retrieval-Augmented Generation (RAG) and a Large Language Model, and serves concise, plain-language answers through a friendly chat interface.

---

## Why EcoSage?

* **Clarity for citizens.** No more wading through 200-page regulatory texts.
* **Proven tech.** Built on open-source frameworks: Flask, FAISS, Transformers.
* **Bring-your-own PDFs.** Just drop files in the `data/` folder—EcoSage takes care of the rest.
* **Runs anywhere.** Laptop, workstation, server, NVIDIA CUDA, or Apple Silicon.

---

## Language Switch & Translator

Select **English** or **Nederlands** in the chat header.

* When **English** is active, answers are returned directly from **NousResearch/Hermes-3-Llama-3.2-3B**.
* When **Nederlands** is active, EcoSage:

  1. Generates the answer in English.
  2. Instantly translates it to Dutch with **`Helsinki-NLP/opus-mt-en-nl`** (MarianMT).

> The translator loads once at start-up and requires the lightweight *SentencePiece* library (already included in `requirements.txt`).

---

## Quick Start

### 1. Prerequisites

| Requirement | Version (or newer)                          |
| ----------- | ------------------------------------------- |
| Python      | 3.8 – 3.12                                  |
| Git         | Latest stable                               |

#### GPU acceleration (optional)

| Hardware                     | Setup                                                                                                                                                                                                                 |
| ---------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **NVIDIA GPU** (≥ 6 GB VRAM) | 1. Install CUDA 11.8 & cuDNN 8.<br>2. `pip install torch --index-url https://download.pytorch.org/whl/cu118`<br>3. `pip install bitsandbytes`                                                                         |
| **Apple Silicon (M-series)** | 1. Install the Metal-optimized wheels:<br>`pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu`<br>2. Set `export PYTORCH_ENABLE_MPS_FALLBACK=1` if you mix CPU & GPU ops |

### 2. Clone the repo

```bash
git clone https://github.com/Vince16270/EcoSage
cd EcoSage
```

### 3. Create & activate a virtual environment

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 4. Install Python dependencies

```bash
pip install -r requirements.txt
```

> **Note**   GPU users can swap `faiss-cpu` for `faiss-gpu` in *requirements.txt*.

### 5. Pre-process documents (first run only)

```bash
python -m src.preprocessing data/*.pdf
```

This will:

1. Split PDFs into manageable text chunks.
2. Embed each chunk.
3. Save `models/chunks.json` and the FAISS index `models/index.faiss`.

### 6. Launch the API

```bash
python -m src.api
```

* Flask serves at **`http://127.0.0.1:5000`**.
* Static files in `frontend/` are auto-served.

### 7. Open the chat UI

```
http://127.0.0.1:5000
```

Ask a question, watch the typing dots, and then get an answer.

### 8. Environment variables (common)

| Variable            | Description                  | Default                              |
| ------------------- | ---------------------------- | ------------------------------------ |
| `PORT`              | Flask port                   | `5000`                               |
| `DEVICE`            | `cpu`, `cuda`, `mps`, `auto` | `auto`                               |
| `MODEL_NAME`        | HF ID for generator model    | `NousResearch/Hermes-3-Llama-3.2-3B` |
| `TRANSLATION_MODEL` | HF ID for EN→NL translator   | `Helsinki-NLP/opus-mt-en-nl`         |

Example:

```bash
export PORT=8080
export DEVICE=mps
python -m src.api
```

---

## Testing & Evaluation

EcoSage includes an end-to-end evaluation script that queries the running chat API, compares model replies against a JSON of test questions, and computes several metrics (Exact Match, Word Overlap, BLEU, ROUGE-1).

### 1. Testset JSON

Make sure you have a `testset.json` in the `tests/` folder, structured as follows:

```json
{
  "questions_and_answers": [
    {
      "id": 1,
      "question": "What is the European Green Deal and what are its main goals?",
      "answer": "The European Green Deal is a comprehensive plan by the European Union to make Europe the first climate-neutral continent by 2050. Its main goals include achieving net-zero greenhouse gas emissions, decoupling economic growth from resource use, and ensuring a just and inclusive transition for all regions and citizens."
    },
    {
      "id": 2,
      "question": "How can I reduce my carbon footprint at home?",
      "answer": "You can reduce your carbon footprint by improving home insulation, using energy-efficient appliances, switching to LED lighting, reducing meat consumption, and choosing public transport or an electric vehicle."
    }
  ]
}
```

There are more question with answers, in total we have 30 questions. 

### 2. End-to-End Evaluation Script

The script `tests/end_to_end_eval.py` will:

1. Load all question items from `tests/testset.json`.
2. Randomly sample *N* questions if you supply `--num N` (otherwise it runs all).
3. Send each question to the `/chat` API with `"lang": "en"`.
4. Compute metrics:

   * **Exact Match** (1 or 0)
   * **Word Overlap Ratio**
   * **BLEU-4** (with smoothing)
   * **ROUGE-1 F1**
5. Save a detailed CSV and a summary CSV in `tests/evaluation_results/`.

#### 2.1 Install additional dependencies

This evaluation uses NLTK for BLEU. Install or verify:

```bash
pip install nltk
```

When you first import `nltk.translate.bleu_score`, NLTK will download what it needs automatically. If you encounter an NLTK resource error, run:

```python
import nltk
nltk.download("punkt")
```

#### 2.2 Run the evaluation

1. Ensure the Flask server is running (e.g. `python -m src.api`).

2. In another terminal (from project root):

   ```bash
   # To evaluate all questions
   python tests/end_to_end_eval.py

   # To evaluate 5 random questions
   python tests/end_to_end_eval.py --num 5
   ```

   Each run will pick a different random subset if you specify `--num`. No seed is fixed by default.

3. After completion, two CSV files will appear under `tests/evaluation_results/`:

   * **`detailed_results.csv`**
     Columns:

     ```
     id,question,expected_answer,model_answer,exact_match,word_overlap_ratio,bleu_score,rouge1_f1
     ```

   * **`summary_stats.csv`**
     Contains:

     ```
     metric,value
     total_questions,<int>
     avg_exact_match,<0–1>
     avg_word_overlap_ratio,<0–1>
     avg_bleu_score,<0–1>
     avg_rouge1_f1,<0–1>
     ```

Open these CSVs in Excel, Google Sheets, or a text editor to inspect per-question metrics and overall performance.

---

## Updating

```bash
git pull
pip install -r requirements.txt --upgrade
```

---

© 2025 EcoSage AI • MIT License