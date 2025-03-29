import os
import PyPDF2
import docx
from typing import Dict, Any, List, Optional
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
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('financial_processor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModelLoadError(Exception):
    """Custom exception for model loading errors"""
    pass

class DocumentProcessingError(Exception):
    """Custom exception for document processing errors"""
    pass

class FinancialDocumentProcessor:
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the Financial Document Processor.
        
        Args:
            model_path: Optional path to custom model directory
        
        Raises:
            ModelLoadError: If any of the required models fail to load
        """
        try:
            # Initialize financial NER pipeline
            self.ner_pipeline = pipeline(
                "token-classification",
                model=model_path or "yiyanghkust/finbert",
                aggregation_strategy="simple"
            )
            logger.info("Successfully loaded NER pipeline")
            
            # Initialize sentence transformer
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Successfully loaded sentence transformer")
            
            # Initialize financial sentiment classifier
            self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(
                model_path or "yiyanghkust/finbert-tone"
            )
            self.sentiment_tokenizer = AutoTokenizer.from_pretrained(
                model_path or "yiyanghkust/finbert-tone"
            )
            logger.info("Successfully loaded sentiment model")
            
            # Create data directory if it doesn't exist
            data_dir = Path("./data")
            data_dir.mkdir(exist_ok=True)
            
            # Initialize Chroma vector store
            chroma_path = data_dir / "chroma_db"
            self.chroma_client = chromadb.PersistentClient(path=str(chroma_path))
            try:
                self.collection = self.chroma_client.get_collection("financial_documents")
                logger.info("Using existing Chroma collection")
            except:
                self.collection = self.chroma_client.create_collection("financial_documents")
                logger.info("Created new Chroma collection")
            
            # Initialize text splitter
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=512,
                chunk_overlap=50,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
            )
            
            # Load spaCy model
            try:
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("Successfully loaded spaCy model")
            except OSError:
                logger.warning("Downloading spaCy model...")
                os.system("python -m spacy download en_core_web_sm")
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("Successfully downloaded and loaded spaCy model")
            
        except Exception as e:
            logger.error(f"Failed to initialize models: {str(e)}")
            raise ModelLoadError(f"Failed to initialize models: {str(e)}")
        
        # Financial taxonomy
        self.taxonomy = {
            'financial_metrics': ['revenue', 'profit', 'margin', 'ebitda', 'eps', 'roa', 'roe'],
            'risk_factors': ['risk', 'uncertainty', 'liability', 'debt', 'exposure'],
            'market_terms': ['market', 'industry', 'sector', 'competition', 'trend'],
            'regulatory': ['compliance', 'regulation', 'law', 'requirement', 'policy']
        }

    def process_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Process text through NER pipeline.
        
        Args:
            text: Input text to process
            
        Returns:
            List of extracted entities with their types and confidence scores
            
        Raises:
            ValueError: If text is empty or not a string
        """
        if not isinstance(text, str):
            raise ValueError("Input must be a string")
        if not text.strip():
            raise ValueError("Input text cannot be empty")
            
        try:
            ner_results = self.ner_pipeline(text)
            
            entity_map = {
                'LABEL_0': 'ORG',
                'LABEL_1': 'MONEY',
                'LABEL_2': 'DATE',
                'LABEL_3': 'PERCENT'
            }
            
            entities = []
            for entity in ner_results:
                if entity['score'] > 0.5:
                    entity_type = entity_map.get(entity['entity'], entity['entity'])
                    entities.append({
                        "text": entity['word'].strip(),
                        "entity": entity_type,
                        "score": round(entity['score'], 3)
                    })
            
            return entities
            
        except Exception as e:
            logger.error(f"Error processing text: {str(e)}")
            raise DocumentProcessingError(f"Error processing text: {str(e)}")

    def process_document(self, file_path: str) -> Dict[str, Any]:
        """
        Process a document and extract financial information using RAG.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dictionary containing extracted information
            
        Raises:
            FileNotFoundError: If file doesn't exist
            DocumentProcessingError: If processing fails
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        try:
            text = self._extract_text(file_path)
            logger.info(f"Successfully extracted text from {file_path}")
            
            chunks = self.text_splitter.split_text(text)
            logger.info(f"Split text into {len(chunks)} chunks")
            
            # Store chunks in vector store
            try:
                embeddings = [
                    self.sentence_model.encode(chunk) 
                    for chunk in tqdm(chunks, desc="Generating embeddings")
                ]
                self.collection.add(
                    embeddings=embeddings,
                    documents=chunks,
                    ids=[f"chunk_{i}" for i in range(len(chunks))]
                )
                logger.info("Successfully stored embeddings in Chroma")
            except Exception as e:
                logger.error(f"Error storing embeddings: {str(e)}")
                raise
            
            # Process document
            entities = []
            for chunk in tqdm(chunks, desc="Extracting entities"):
                chunk_entities = self.process_text(chunk)
                entities.extend(chunk_entities)
            
            classified_sections = self._classify_sections(chunks)
            metrics = self._extract_financial_metrics(text)
            sentiment = self._analyze_sentiment(text)
            
            return {
                'entities': entities,
                'classified_sections': classified_sections,
                'metrics': metrics,
                'sentiment': sentiment,
                'chunks': len(chunks)
            }
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            raise DocumentProcessingError(f"Error processing document: {str(e)}")
    
    def query_document(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Query the document using RAG.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of relevant document chunks
            
        Raises:
            ValueError: If query is empty or k is invalid
        """
        if not query.strip():
            raise ValueError("Query cannot be empty")
        if k < 1:
            raise ValueError("k must be positive")
            
        try:
            query_embedding = self.sentence_model.encode(query)
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k
            )
            return results.get('documents', [])
            
        except Exception as e:
            logger.error(f"Error querying document: {str(e)}")
            return []
    
    def _extract_text(self, file_path: str) -> str:
        """Extract text from various file formats."""
        try:
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
                    
            if not text.strip():
                raise DocumentProcessingError("Extracted text is empty")
                
            return text
            
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {str(e)}")
            raise DocumentProcessingError(f"Error extracting text: {str(e)}")
    
    def _classify_sections(self, chunks: List[str]) -> List[Dict[str, Any]]:
        """Classify document sections based on the financial taxonomy."""
        try:
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
            
        except Exception as e:
            logger.error(f"Error classifying sections: {str(e)}")
            return []
            
    def _extract_financial_metrics(self, text: str) -> Dict[str, Any]:
        """Extract key financial metrics."""
        try:
            doc = self.nlp(text.lower())
            metrics = {}
            
            for metric in self.taxonomy['financial_metrics']:
                if metric in doc.text:
                    # Find sentences containing the metric
                    relevant_sentences = [
                        sent.text for sent in doc.sents 
                        if metric in sent.text.lower()
                    ]
                    if relevant_sentences:
                        metrics[metric] = relevant_sentences
                        
            return metrics
            
        except Exception as e:
            logger.error(f"Error extracting metrics: {str(e)}")
            return {}
            
    def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of financial text."""
        try:
            inputs = self.sentiment_tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512
            )
            
            with torch.no_grad():
                outputs = self.sentiment_model(**inputs)
                predictions = torch.softmax(outputs.logits, dim=1)
                
            labels = ['negative', 'neutral', 'positive']
            scores = {
                label: round(float(score), 3) 
                for label, score in zip(labels, predictions[0])
            }
            
            return {
                'sentiment': max(scores.items(), key=lambda x: x[1])[0],
                'scores': scores
            }
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {str(e)}")
            return {
                'sentiment': 'neutral',
                'scores': {'negative': 0.0, 'neutral': 1.0, 'positive': 0.0}
            }
