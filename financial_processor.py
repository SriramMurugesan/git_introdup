import os
import PyPDF2
import docx
import nltk
from typing import Dict, Any, List
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import torch

# Download required NLTK data
nltk.download('punkt')

class FinancialDocumentProcessor:
    def __init__(self):
        # Initialize NER pipeline with FinBERT
        self.ner_pipeline = pipeline("ner", model="ProsusAI/finbert")
        
        # Initialize sentence transformer for embeddings
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load financial concepts and create FAISS index
        self.concepts = self._load_financial_concepts()
        self.concept_embeddings = self.sentence_model.encode(self.concepts)
        self.index = faiss.IndexFlatL2(self.concept_embeddings.shape[1])
        self.index.add(self.concept_embeddings)

    def _load_financial_concepts(self) -> List[str]:
        """Load predefined financial concepts"""
        return [
            "revenue", "profit margin", "operating income",
            "net income", "earnings per share", "EBITDA",
            "cash flow", "balance sheet", "income statement",
            "assets", "liabilities", "equity",
            "market capitalization", "dividend yield", "P/E ratio",
            "debt-to-equity", "working capital", "ROI",
            "ROE", "ROA", "gross margin",
            "operating margin", "net margin", "current ratio",
            "quick ratio", "inventory turnover", "accounts receivable",
            "accounts payable", "capital expenditure", "depreciation"
        ]

    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from a PDF file"""
        text = ""
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text

    def extract_text_from_docx(self, file_path: str) -> str:
        """Extract text from a DOCX file"""
        doc = docx.Document(file_path)
        return "\n".join([paragraph.text for paragraph in doc.paragraphs])

    def extract_text(self, file_path: str) -> str:
        """Extract text from a document based on its file type"""
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.pdf':
            return self.extract_text_from_pdf(file_path)
        elif ext == '.docx':
            return self.extract_text_from_docx(file_path)
        elif ext == '.txt':
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    def split_into_sections(self, text: str, max_length: int = 512) -> List[str]:
        """Split text into manageable sections"""
        sentences = nltk.sent_tokenize(text)
        sections = []
        current_section = ""
        
        for sentence in sentences:
            if len(current_section) + len(sentence) < max_length:
                current_section += sentence + " "
            else:
                if current_section:
                    sections.append(current_section.strip())
                current_section = sentence + " "
        
        if current_section:
            sections.append(current_section.strip())
        
        return sections

    def process_section(self, text: str) -> Dict[str, Any]:
        """Process a single section of text"""
        # Named Entity Recognition
        ner_results = self.ner_pipeline(text)
        
        # Convert numpy float32 to regular Python float for JSON serialization
        ner_results = [{
            **{k: float(v) if isinstance(v, (torch.Tensor, np.float32)) else v 
               for k, v in entity.items()}
        } for entity in ner_results]
        
        # Get document embedding and find similar concepts
        document_embedding = self.sentence_model.encode([text], convert_to_tensor=True)
        _, indices = self.index.search(document_embedding.cpu().detach().numpy(), k=3)
        relevant_concepts = [self.concepts[i] for i in indices[0]]
        
        return {
            'text': text,
            'named_entities': ner_results,
            'relevant_concepts': relevant_concepts
        }

    def process_text(self, text: str) -> Dict[str, Any]:
        """Process a single piece of text"""
        return self.process_section(text)

    def process_document(self, file_path: str) -> Dict[str, Any]:
        """Process an entire document"""
        # Extract text from document
        text = self.extract_text(file_path)
        
        # Split into sections
        sections = self.split_into_sections(text)
        
        # Process each section
        processed_sections = [self.process_section(section) for section in sections]
        
        # Collect unique concepts and total entities
        all_concepts = set()
        total_entities = 0
        for section in processed_sections:
            all_concepts.update(section['relevant_concepts'])
            total_entities += len(section['named_entities'])
        
        # Create document summary
        summary = {
            'total_sections': len(sections),
            'total_entities': total_entities,
            'unique_concepts': list(all_concepts)
        }
        
        return {
            'summary': summary,
            'sections': processed_sections
        }
