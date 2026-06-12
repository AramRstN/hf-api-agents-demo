# HF API Agents Demo
A teaching repository for an introductory hands-on lesson on:

- **Reusing ML models**
- **Exploring Hugging Face**
- **Building a tiny ML application**
- **Adding a UI with Gradio**
- **Calling GenAI through an API**
- **Understanding the basic idea of AI agents**

This repository is used during class as a **live demo codebase**. Students can follow along, run the examples, and use the code as a reference after the lesson.

---

## Repository structure

```text
hf-api-agents-demo/
│
├── README.md
├── requirements.txt
|── 00_setup.ipynb
├── .gitignore
├── .env.example
│
├── notebooks/
│   ├── 01_reuse_ml_and_hf.ipynb
│   ├── 02_prediction.ipynb
│   ├── 03_openai_api_and_tiny_agent.ipynb
│   └── 04_gradio_ui.ipynb
│
├── src/
│   ├── __init__.py
│   ├── hf_demo.py
│   ├── model.py
│   ├── llm_api.py
│   └── tiny_agent.py
│
└── app/
    └── gradio_app.py
```
---

## What each part contains
`notebooks/`

Jupyter notebooks used during the lesson.

`app/`

A standalone Gradio application.


---
## Quick start
1. Install Ollama from `ollama.com`
2. Download at least one local model

```bash
ollama pull llama3.2
```

or

```bash
ollama pull qwen3:4b
```

3. Clone this repository
4. Run:

```bash
00_setup.ipynb
```

and make sure all checks pass successfully.
---
## Related lab repository

A separate lab starter repository is provided for in-class exercises:

`hf-api-agents-lab`

That repository contains TODOs for students to complete.

---
## Disclaimer

This repository is designed for teaching and experimentation.
It is intentionally simple and focuses on clarity over production-level engineering.
