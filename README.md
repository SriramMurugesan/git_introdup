# Financial Statement Analyzer

A powerful tool for analyzing financial documents and extracting relevant financial entities using advanced NLP techniques.

## Features

- Financial Entity Recognition (Organizations, Money, Dates, Percentages, Metrics, Regulators)
- Document Processing (PDF, DOCX support)
- Advanced NLP using FinBERT models
- Sentiment Analysis for financial text
- RAG (Retrieval Augmented Generation) implementation
- Vector storage using ChromaDB
- Web interface for easy interaction

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download required spaCy model:
   ```bash
   python -m spacy download en_core_web_sm
   ```

## Usage

1. Start the web server:
   ```bash
   python main.py
   ```
2. Open your browser and navigate to `http://localhost:5000`
3. Enter financial text in the input box and click "Analyze Text"

## Project Structure

- `main.py`: Flask web server and basic entity recognition
- `financial_processor.py`: Advanced NLP processing with ML models
- `templates/index.html`: Web interface
- `requirements.txt`: Project dependencies

## Dependencies

- Flask
- PyTorch
- Transformers (HuggingFace)
- SentenceTransformers
- ChromaDB
- spaCy
- PyPDF2
- python-docx
- And more (see requirements.txt)

## License

MIT License