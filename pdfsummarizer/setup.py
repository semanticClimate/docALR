from setuptools import setup, find_packages

setup(
    name='pdfsummarizer',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'transformers[sentencepiece]',
        'nltk',
        'pymupdf',
    ],
    entry_points={
        'console_scripts': [
            'pdfsummarizer=pdfsummarizer.cli:main',
        ],
    },
    author='Shabnam Barbhuiya',
    description='Summarize academic PDFs from the command line using HuggingFace models',
    keywords='NLP summarization PDF transformer CLI',
)
