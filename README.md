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
├── .gitignore
├── .env.example
│
├── notebooks/
│   ├── 01_reuse_ml_and_hf.ipynb
│   ├── 02_titanic_prediction.ipynb
│   ├── 03_gradio_ui.ipynb
│   └── 04_openai_api_and_tiny_agent.ipynb
│
├── src/
│   ├── __init__.py
│   ├── hf_demo.py
│   ├── titanic_model.py
│   ├── llm_api.py
│   └── tiny_agent.py
│
├── app/
│   └── gradio_app.py
│
├── data/
│   └── titanic_sample.csv
│
└── assets/
    └── architecture.png
```
---

## What each part contains
`notebooks/`

Jupyter notebooks used during the lesson.

`src/`

Reusable Python code for the notebooks and app.

`app/`

A standalone Gradio application.

`data/`

Small sample datasets used for demo purposes.

`assets/`

Images or diagrams used in the lesson or README.

---
## Quick start
1. Clone the repository.
2. Create a virtual environment
3. Install dependencies: 
```shell 
pip install -r requirements.txt
```
---
## Related lab repository

A separate lab starter repository is provided for in-class exercises:

`hf-api-agents-lab`

That repository contains starter code and TODOs for students to complete.

---
## Disclaimer

This repository is designed for teaching and experimentation.
It is intentionally simple and focuses on clarity over production-level engineering.