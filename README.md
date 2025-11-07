# docALR
# Scientific Paper Analysis Pipeline

A modular and extensible pipeline for **automated retrieval**, **named entity recognition (NER)**, **summarization**, and **question-answering** from scientific papers using **Pygetpapers**, **spaCy/transformers**, and **LLM/RAG-based models**.

---

## Features

- **Retrieve scientific papers** from open-access sources using [Pygetpapers](https://github.com/petermr/pygetpapers)
- **Extract named entities** using pre-trained spaCy or transformer-based models
- **Summarize full texts** or abstracts using transformer-based summarization models (e.g., BART, T5)
- **Ask questions** and get answers with RAG-based pipelines or custom LLMs

---
# ============================================
# 🧪 Project Setup & Run Instructions
# ============================================
# 1️⃣ Create a new Conda environment with Python 3.11
      '''conda create -n phytochem python=3.11 -y'''
#
# 2️⃣ Activate the environment
  '''conda activate phytochem'''
#
# 3️⃣ Install all required libraries
  '''pip install -r requirements.txt'''
#
# 4️⃣ Run the Streamlit app
  '''streamlit run app_final.py'''


