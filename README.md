# 🌌 Nova — Neural Optimized Virtual Assistant

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![UI](https://img.shields.io/badge/UI-CustomTkinter-lightblue.svg)](https://github.com/TomSchimansky/CustomTkinter)

Nova is a Hindi-speaking AI desktop assistant that answers questions from uploaded PDFs using local LLMs. It supports voice input through Whisper and speaks responses via Piper TTS.

---

## ⚙️ Features

* Answers only in **Hindi** (no English words)
* Upload PDFs and query them contextually
* **Voice input + spoken output**
* Works fully **offline** with Ollama, Whisper, and Piper
* Modern **CustomTkinter** UI with animated splash screen

---

## 🧩 Requirements

| Tool         | Purpose            | Link                                                                         |
| ------------ | ------------------ | ---------------------------------------------------------------------------- |
| Python 3.11 | Runtime            | [python.org](https://www.python.org)                                         |
| Ollama       | LLM backend        | [ollama.ai](https://ollama.ai)                                               |
| Whisper.cpp  | Speech recognition | [github.com/ggerganov/whisper.cpp](https://github.com/ggerganov/whisper.cpp) |
| Piper        | Text-to-speech     | [github.com/rhasspy/piper](https://github.com/rhasspy/piper)                 |

---

## 🚀 Setup

```bash
git clone https://github.com/yourusername/nova-assistant.git
cd nova-assistant
uv sync
```

Create a `.env` file from the provided `.env.example` and configure paths:

---

## ▶️ Run

```bash
uv run main.py
```

1. Add your PDFs
2. Type or speak your question
3. Nova answers in Hindi and reads it aloud

---

## 📁 Structure

```
nova/
├── .env.example
├── .novadocs
│   └── assets
├── main.py
├── pyproject.toml
├── README.md
└── uv.lock

```

---

## 📜 License

MIT © 2025 — Md Altaf Husssain
