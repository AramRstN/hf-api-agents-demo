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

## Learning goals

By the end of this lesson, students should be able to:

- explain what it means to **reuse an ML model**
- understand Hugging Face as a platform for **models, datasets, and demos**
- use a small ML model inside a simple application
- wrap a prediction function in a **Gradio UI**
- understand the basic role of a **GenAI API**
- understand the high-level idea of an **AI agent**

---

## Lesson flow

The lesson follows this progression:

1. **Reuse of ML models**  
   Why we do not always train models from scratch

2. **Hugging Face ecosystem**  
   Finding reusable datasets, models, and demos

3. **Tiny model reuse demo**  
   Running a pretrained model in a few lines

4. **Tiny Titanic prediction app**  
   Turning model logic into an application

5. **Gradio UI**  
   Making the app interactive

6. **OpenAI API introduction**  
   Showing how applications can call hosted GenAI models

7. **AI agent idea**  
   Understanding how an application can decide what to do next using models and tools

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