import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from sentence_transformers import SentenceTransformer
import faiss
import json
import numpy as np

# Load Pretrained Transformer Model for NER
model_name = "yiyanghkust/finbert-pretrain"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)

# Load Sentence Transformer for RAG
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

# Example Financial Document
financial_text = "Apple Inc. reported a revenue of $394 billion in 2023 according to its balance sheet."

# Named Entity Recognition (NER)
ner_results = ner_pipeline(financial_text)
print("NER Results:", ner_results)

# Financial Concept Embeddings for RAG
concepts = ["Net Revenue", "Net Profit Margin", "Balance Sheet", "Cash Flow"]
concept_embeddings = sentence_model.encode(concepts, convert_to_tensor=True)

# Build FAISS Index for Retrieval
index = faiss.IndexFlatL2(concept_embeddings.shape[1])
index.add(concept_embeddings.cpu().detach().numpy())

# Query Example
document_embedding = sentence_model.encode([financial_text], convert_to_tensor=True)
_, indices = index.search(document_embedding.cpu().detach().numpy(), k=2)
retrieved_concepts = [concepts[i] for i in indices[0]]

print("Retrieved Financial Concepts:", retrieved_concepts)
tokens = tokenizer(financial_text, return_tensors="pt")
print("Tokens:", tokens)

# Prototype JSON Output
output = {
    "original_text": financial_text,
    "named_entities": [{
        **{k: float(v) if isinstance(v, (torch.Tensor, np.float32)) else v 
           for k, v in entity.items()}
    } for entity in ner_results],
    "retrieved_concepts": retrieved_concepts
}
print(json.dumps(output, indent=4))
