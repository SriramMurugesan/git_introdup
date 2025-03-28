from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional
from financial_processor import FinancialDocumentProcessor
import os
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import uvicorn

app = FastAPI(title="Financial Document Analysis")

# Create directories if they don't exist
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)
os.makedirs("uploads", exist_ok=True)

# Mount static files and setup templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Initialize the financial processor
processor = FinancialDocumentProcessor()

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx'}

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load models globally
model_name = "yiyanghkust/finbert-pretrain"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

# Financial concepts for RAG
concepts = ["Net Revenue", "Net Profit Margin", "Balance Sheet", "Cash Flow"]
concept_embeddings = sentence_model.encode(concepts, convert_to_tensor=True)

# Build FAISS Index for Retrieval
index = faiss.IndexFlatL2(concept_embeddings.shape[1])
index.add(concept_embeddings.cpu().detach().numpy())

class TextRequest(BaseModel):
    text: str

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze_text")
async def analyze_text(text_request: TextRequest):
    text = text_request.text
    
    if not text:
        raise HTTPException(status_code=400, detail="No text provided")
    
    try:
        # Perform NER
        ner_results = ner_pipeline(text)
        
        # Get document embedding and search for relevant concepts
        document_embedding = sentence_model.encode([text], convert_to_tensor=True)
        _, indices = index.search(document_embedding.cpu().detach().numpy(), k=2)
        retrieved_concepts = [concepts[i] for i in indices[0]]
        
        return {
            "original_text": text,
            "named_entities": [{
                **{k: float(v) if isinstance(v, (torch.Tensor, np.float32)) else v 
                   for k, v in entity.items()}
            } for entity in ner_results],
            "retrieved_concepts": retrieved_concepts
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze_document")
async def analyze_document(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file provided")
    
    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="File type not allowed")
    
    try:
        # Save the uploaded file
        file_path = os.path.join("uploads", file.filename)
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process the document based on its type
        if file.filename.endswith('.pdf'):
            text = processor.extract_text_from_pdf(file_path)
        elif file.filename.endswith('.docx'):
            text = processor.extract_text_from_docx(file_path)
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        
        # Clean up the uploaded file
        os.remove(file_path)
        
        # Analyze the extracted text
        return await analyze_text(TextRequest(text=text))
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
