# summarization.py

import os
import fitz  # PyMuPDF
import nltk
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

nltk.download("punkt")


def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    return "".join([page.get_text() for page in doc])


def remove_references_section(text):
    ref_keywords = ["references", "bibliography", "works cited"]
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        if any(keyword in line.strip().lower() for keyword in ref_keywords):
            break
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def chunk_text(text, tokenizer):
    max_input_length = tokenizer.model_max_length
    sentences = nltk.tokenize.sent_tokenize(text)
    chunks = []
    chunk = ""
    length = 0
    for sentence in sentences:
        sentence_length = len(tokenizer.tokenize(sentence))
        if length + sentence_length <= max_input_length:
            chunk += sentence + " "
            length += sentence_length
        else:
            chunks.append(chunk.strip())
            chunk = sentence + " "
            length = sentence_length
    if chunk:
        chunks.append(chunk.strip())
    return chunks


def summarize_chunks(chunks, tokenizer, model, max_output_length, min_length, length_penalty):
    summaries = []
    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors="pt", truncation=True, padding=True)
        output = model.generate(
            **inputs,
            max_length=max_output_length,
            min_length=min_length,
            length_penalty=length_penalty,
            early_stopping=True,
            no_repeat_ngram_size=3
        )
        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        cleaned = re.sub(r'\b(?:figure|fig)\s*\d+', '', decoded)
        summaries.append(cleaned.strip())
    return " ".join(summaries)


def summarize_file(file_path, model_name, max_output_length, min_length, length_penalty):
    text = extract_text_from_pdf(file_path)
    cleaned = remove_references_section(text)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    chunks = chunk_text(cleaned, tokenizer)
    summary = summarize_chunks(chunks, tokenizer, model, max_output_length, min_length, length_penalty)
    return summary
