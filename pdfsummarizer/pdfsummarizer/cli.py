# pdfsummarizer/cli.py

import os
import argparse
import csv
from .summarizer import extract_text_from_pdf, summarize_text

def main():
    parser = argparse.ArgumentParser(description="Summarize PDFs using a transformer model")
    parser.add_argument('--pdf_dir', type=str, required=True, help='Path to folder containing PDFs')
    parser.add_argument('--model_name', type=str, default="sshleifer/distilbart-cnn-12-6", help='Hugging Face model name')
    parser.add_argument('--max_output_length', type=int, default=200, help='Maximum summary length')
    parser.add_argument('--min_length', type=int, default=10, help='Minimum summary length')
    parser.add_argument('--length_penalty', type=float, default=0.8, help='Length penalty')
    parser.add_argument('--output', type=str, default="summaries.csv", help='Output file path (CSV)')

    args = parser.parse_args()

    pdf_paths = [os.path.join(args.pdf_dir, f) for f in os.listdir(args.pdf_dir) if f.endswith(".pdf")]

    print(f"Found {len(pdf_paths)} PDF files in {args.pdf_dir}")

    with open(args.output, "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Filename", "Summary"])

        for pdf in pdf_paths:
            print(f"Summarizing: {os.path.basename(pdf)}")
            try:
                text = extract_text_from_pdf(pdf)
                summary = summarize_text(
                    text,
                    model_name=args.model_name,
                    max_output_length=args.max_output_length,
                    min_length=args.min_length,
                    length_penalty=args.length_penalty
                )
                writer.writerow([os.path.basename(pdf), summary])
            except Exception as e:
                print(f"Error summarizing {pdf}: {e}")
                writer.writerow([os.path.basename(pdf), f"Error: {e}"])

    print(f"\n✅ All summaries saved to {args.output}")
