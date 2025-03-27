from flask import Flask, render_template, request, jsonify, send_from_directory
from financial_processor import FinancialDocumentProcessor
import os
from werkzeug.utils import secure_filename
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize the financial processor
processor = FinancialDocumentProcessor()

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx'}

def allowed_file(filename):
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

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze_text', methods=['POST'])
def analyze_text():
    data = request.get_json()
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    try:
        # Named Entity Recognition
        ner_results = ner_pipeline(text)
        
        # Convert numpy float32 to regular Python float for JSON serialization
        ner_results = [{
            **{k: float(v) if isinstance(v, (torch.Tensor, np.float32)) else v 
               for k, v in entity.items()}
        } for entity in ner_results]
        
        # Get document embedding and find similar concepts
        document_embedding = sentence_model.encode([text], convert_to_tensor=True)
        _, indices = index.search(document_embedding.cpu().detach().numpy(), k=2)
        retrieved_concepts = [concepts[i] for i in indices[0]]
        
        # Process the text
        result = processor.process_section(text)
        result['named_entities'] = ner_results
        result['retrieved_concepts'] = retrieved_concepts
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analyze_document', methods=['POST'])
def analyze_document():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400

    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process the document
        result = processor.process_document(filepath)
        
        # Clean up the uploaded file
        os.remove(filepath)
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
