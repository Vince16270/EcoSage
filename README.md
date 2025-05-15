<h1 align="center">EcoSage</h1>
<h3 align="center">LLM‑Powered Chatbot for Sustainable Policy</h3>

![EcoSage banner](frontend/logo_full.png)

EcoSage makes European and Dutch energy‑transition policy understandable for everyone. It ingests official PDF documents, applies state‑of‑the‑art Retrieval‑Augmented Generation (RAG) and a Large Language Model, and serves concise, plain‑language answers through a friendly chat interface.

---

## Why EcoSage?

* **Clarity for citizens.** No more wading through 200‑page regulatory texts.
* **Proven tech.** Built on open‑source frameworks: Flask, FAISS, Transformers.
* **Bring‑your‑own PDFs.** Just drop files in the `data/` folder—EcoSage takes care of the rest.
* **Runs anywhere.** Laptop, workstation, server, NVIDIA CUDA, or Apple Silicon.

---

## Language Switch & Translator

Select **English** or **Nederlands** in the chat header.

* When **English** is active, answers are returned directly from **NousResearch/Hermes-3-Llama-3.2-3B**.
* When **Nederlands** is active, EcoSage:

  1. Generates the answer in English.
  2. Instantly translates it to Dutch with **`Helsinki‑NLP/opus‑mt‑en‑nl`** (MarianMT).

> The translator loads once at start‑up and requires the lightweight *SentencePiece* library (already included in `requirements.txt`).

---

## Quick Start

### 1. Prerequisites

| Requirement | Version (or newer)                          |
| ----------- | ------------------------------------------- |
| Python      | 3.8 – 3.12                                  |
| Git         | Latest stable                               |
| Node.js ✱   | *Optional* (serve HTML from another server) |

✱ Not needed if you simply let Flask host the static html.

#### GPU acceleration (optional)

| Hardware                     | Setup                                                                                                                                                                                                                 |
| ---------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **NVIDIA GPU** (≥ 6 GB VRAM) | 1. Install CUDA 11.8 & cuDNN 8.<br>2. `pip install torch --index-url https://download.pytorch.org/whl/cu118`<br>3. `pip install bitsandbytes`                                                                         |
| **Apple Silicon (M‑series)** | 1. Install the Metal‑optimized wheels:<br>`pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu`<br>2. Set `export PYTORCH_ENABLE_MPS_FALLBACK=1` if you mix CPU & GPU ops |

### 2. Clone the repo

```bash
git clone https://github.com/Vince16270/EcoSage
cd EcoSage
```

### 3. Create & activate a virtual environment

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 4. Install Python dependencies

```bash
pip install -r requirements.txt
```

> **Tip**   GPU users can swap `faiss‑cpu` for `faiss‑gpu` in *requirements.txt*.

### 5. Pre‑process documents (first run only)

```bash
python -m src.preprocessing data/*.pdf
```

This will:

1. Split PDFs into manageable text chunks.
2. Embed each chunk.
3. Save `models/chunks.json` and the FAISS index `models/index.faiss`.

### 6. Launch the API

```bash
python -m src.api
```

* Flask serves at **`http://127.0.0.1:5000`**.
* Static files in `frontend/` are auto‑served.

### 7. Open the chat UI

```
http://127.0.0.1:5000
```

Ask a question, watch the typing dots, and then get an answer.

### 8. Environment variables (common)

| Variable            | Description                  | Default                                   |
| ------------------- | ---------------------------- | ----------------------------------------- |
| `PORT`              | Flask port                   | `5000`                                    |
| `DEVICE`            | `cpu`, `cuda`, `mps`, `auto` | `auto`                                    |
| `MODEL_NAME`        | HF ID for generator model    | `NousResearch/Hermes-3-Llama-3.2-3B`      |
| `TRANSLATION_MODEL` | HF ID for EN→NL translator   | `Helsinki-NLP/opus-mt-en-nl`              |

Example:

```bash
export PORT=8080
export DEVICE=mps
python -m src.api
```

---

## Updating

```bash
git pull
pip install -r requirements.txt --upgrade
```

---

© 2025 EcoSage AI • MIT License