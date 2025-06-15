# pdfsummarizer/summarizer.py

import os
import fitz
import nltk
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

nltk.download("punkt", quiet=True)

def remove_references_section(text):
    ref_keywords = ["references", "bibliography", "works cited"]
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        if any(keyword in line.strip().lower() for keyword in ref_keywords):
            break
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = " ".join([page.get_text() for page in doc])
    doc.close()
    return text

def summarize_text(text, model_name, max_output_length, min_length, length_penalty):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    max_input_length = tokenizer.model_max_length

    sentences = nltk.tokenize.sent_tokenize(remove_references_section(text))
    chunks = []
    chunk = ""
    length = 0

    for sentence in sentences:
        token_len = len(tokenizer.tokenize(sentence))
        if length + token_len <= max_input_length:
            chunk += sentence + " "
            length += token_len
        else:
            chunks.append(chunk.strip())
            chunk = sentence + " "
            length = token_len
    if chunk:
        chunks.append(chunk.strip())

    summaries = []
    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors="pt", truncation=True, padding=True, max_length=max_input_length)
        outputs = model.generate(
            **inputs,
            max_length=max_output_length,
            min_length=min_length,
            length_penalty=length_penalty,
            early_stopping=True,
            no_repeat_ngram_size=3
        )
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        cleaned = re.sub(r'\b(?:figure|fig)\s*\d+', '', decoded)
        cleaned = re.sub(r'\[\d+(?:,\d+)*\]', '', cleaned)
        cleaned = re.sub(r'\d+(\.\d+)+', '', cleaned)
        summaries.append(cleaned.strip())

    return "\n".join(summaries)
