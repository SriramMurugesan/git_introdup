import os
import PyPDF2
import docx
from typing import Dict, Any, List
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
import spacy
import pandas as pd
from tqdm import tqdm

class FinancialDocumentProcessor:
    def __init__(self):
        # Initialize financial NER pipeline with a model specifically trained for financial entities
        self.ner_pipeline = pipeline(
            "token-classification",
            model="yiyanghkust/finbert",
            aggregation_strategy="simple"
        )
        
        # Initialize sentence transformer for embeddings
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize financial sentiment classifier
        self.sentiment_model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
        self.sentiment_tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
        
        # Initialize Chroma vector store (persistent)
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.chroma_client.create_collection("financial_documents")
        
        # Initialize text splitter for RAG
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        
        # Load spaCy model for additional NLP tasks
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
        
        # Financial taxonomy
        self.taxonomy = {
            'financial_metrics': ['revenue', 'profit', 'margin', 'ebitda', 'eps', 'roa', 'roe'],
            'risk_factors': ['risk', 'uncertainty', 'liability', 'debt', 'exposure'],
            'market_terms': ['market', 'industry', 'sector', 'competition', 'trend'],
            'regulatory': ['compliance', 'regulation', 'law', 'requirement', 'policy']
        }

    def process_text(self, text: str) -> List[str]:
        # Process text through NER pipeline
        ner_results = self.ner_pipeline(text)
        
        # Map label indices to meaningful entity types
        entity_map = {
            'LABEL_0': 'ORG',  # Organization
            'LABEL_1': 'MONEY',  # Monetary value
            'LABEL_2': 'DATE',  # Date
            'LABEL_3': 'PERCENT'  # Percentage
        }
        
        # Format results in desired way
        output_lines = []
        for entity in ner_results:
            if entity['score'] > 0.5:  # Filter low confidence predictions
                entity_type = entity_map.get(entity['entity'], entity['entity'])
                output_line = f"Text: {entity['word']}, Entity: {entity_type}, Score: {entity['score']:.2f}"
                output_lines.append(output_line)
        
        return output_lines

    def process_document(self, file_path: str) -> Dict[str, Any]:
        """Process a document and extract financial information using RAG."""
        try:
            text = self._extract_text(file_path)
        except Exception as e:
            return {'error': str(e)}
        
        chunks = self.text_splitter.split_text(text)
        
        # Store chunks in vector store
        try:
            embeddings = [self.sentence_model.encode(chunk) for chunk in tqdm(chunks)]
            self.collection.add(
                embeddings=embeddings,
                documents=chunks,
                ids=[f"chunk_{i}" for i in range(len(chunks))]
            )
        except Exception as e:
            return {'error': str(e)}
        
        # Extract financial entities
        entities = []
        for chunk in chunks:
            ner_results = self.ner_pipeline(chunk)
            entities.extend(ner_results)
        
        # Classify document sections
        classified_sections = self._classify_sections(chunks)
        
        # Extract key financial metrics
        metrics = self._extract_financial_metrics(text)
        
        # Perform sentiment analysis
        sentiment = self._analyze_sentiment(text)
        
        return {
            'entities': entities,
            'classified_sections': classified_sections,
            'metrics': metrics,
            'sentiment': sentiment,
            'chunks': len(chunks)
        }
    
    def query_document(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Query the document using RAG."""
        try:
            query_embedding = self.sentence_model.encode(query)
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k
            )
            return results.get('documents', [])  # Ensure it returns a list
        except Exception as e:
            return []
    
    def _extract_text(self, file_path: str) -> str:
        """Extract text from PDF or DOCX files."""
        if file_path.endswith('.pdf'):
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ' '.join([page.extract_text() or '' for page in reader.pages])
        elif file_path.endswith('.docx'):
            doc = docx.Document(file_path)
            text = ' '.join([paragraph.text for paragraph in doc.paragraphs])
        else:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
        return text
    
    def _classify_sections(self, chunks: List[str]) -> List[Dict[str, Any]]:
        """Classify document sections based on the financial taxonomy."""
        classified_sections = []
        for chunk in chunks:
            section_types = {}
            doc = self.nlp(chunk.lower())
            
            for category, terms in self.taxonomy.items():
                matches = sum(1 for term in terms if term in doc.text)
                if matches > 0:
                    section_types[category] = matches
            
            if section_types:
                classified_sections.append({
                    'text': chunk,
                    'classifications': section_types
                })
        
        return classified_sections
