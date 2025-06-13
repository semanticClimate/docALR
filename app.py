import streamlit as st
import os
import tempfile
import shutil
import pandas as pd
from summarization import summarize_file

st.set_page_config(page_title="DocAlr - Summarize PDFs", layout="centered")
st.title("📄 DocAlr: PDF Summarizer")

# Model configuration
model_name = st.text_input("🤖 Hugging Face Model", "sshleifer/distilbart-cnn-12-6")
max_output_length = st.slider("Max Output Length", 50, 512, 200)
min_length = st.slider("Min Output Length", 10, 100, 10)
length_penalty = st.slider("Length Penalty", 0.1, 2.0, 0.8)

# File input
uploaded_files = st.file_uploader("📎 Upload one or more PDF files", type="pdf", accept_multiple_files=True)

# Output format
output_format = st.radio("📤 Choose Output Format", ["CSV", "TXT"])

if st.button("Summarize"):
    if uploaded_files:
        results = []

        with st.spinner("Summarizing documents..."):
            for file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(file.read())
                    tmp_path = tmp.name

                try:
                    summary = summarize_file(
                        tmp_path,
                        model_name=model_name,
                        max_output_length=max_output_length,
                        min_length=min_length,
                        length_penalty=length_penalty,
                    )
                    results.append({"filename": file.name, "summary": summary})

                except Exception as e:
                    results.append({"filename": file.name, "summary": f"Error: {e}"})

                os.remove(tmp_path)

        if output_format == "CSV":
            df = pd.DataFrame(results)
            csv = df.to_csv(index=False)
            st.download_button("Download CSV", csv, file_name="summarized_papers.csv", mime="text/csv")
            st.dataframe(df)

        elif output_format == "TXT":
            full_txt = "\n\n".join([f"{r['filename']}\n{r['summary']}" for r in results])
            st.download_button("Download TXT", full_txt, file_name="summarized_papers.txt", mime="text/plain")
            st.text_area("Summaries", full_txt, height=400)

    else:
        st.warning("Please upload at least one PDF file.")