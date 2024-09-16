from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.llms import CTransformers
import numpy as np
import faiss
import fitz
import os
import tempfile

# Initialize global variables
docs = []
doc_vectors = None
index = None

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

def split_text(text, max_len=500):
    """Splits text into smaller chunks with a max_len limit."""
    return [text[i:i + max_len] for i in range(0, len(text), max_len)]

def simple_vectorizer(text, max_len=10):
    """Converts text into a simple numerical vector."""
    vector = [hash(word) % 1000 for word in text.split()]
    vector = vector[:max_len] + [0] * (max_len - len(vector))
    return np.array(vector, dtype='float32')

def create_index(documents):
    """Creates a FAISS index from document vectors."""
    global doc_vectors, index
    doc_vectors = np.array([simple_vectorizer(doc) for doc in documents])
    index = faiss.IndexFlatL2(doc_vectors.shape[1])
    index.add(doc_vectors)

def retrieve(query, k=1):
    """Retrieves the most relevant document chunk based on the query."""
    query_vector = simple_vectorizer(query)
    distances, indices = index.search(np.array([query_vector]), k=k)
    return [docs[i] for i in indices[0]]

class LLaMALLM:
    """Wrapper for the LLaMA model."""
    def __init__(self):
        self.llm = CTransformers(
            model="TheBloke/Llama-2-7B-Chat-GGML",
            model_type="llama",
            max_new_tokens=100, 
            temperature=0.2,
        )

    def generate(self, prompt):
        """Generates a response from the model based on the prompt."""
        return self.llm(prompt).strip()

llm = LLaMALLM()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8080"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

class Query(BaseModel):
    """Schema for query input."""
    question: str

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    """Uploads and processes a PDF file."""
    global docs, doc_vectors, index
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(await file.read())
        tmp_pdf_path = tmp_file.name

    pdf_text = extract_text_from_pdf(tmp_pdf_path)
    docs = split_text(pdf_text, max_len=500) 
    create_index(docs)

    os.remove(tmp_pdf_path)

    return {"message": "File uploaded and processed successfully."}

@app.post("/chat/")
async def chat(query: Query):
    """
    Handles user queries and provides answers based on document content or general knowledge.
    Ensures that responses are safe, unbiased, and accurate.
    """
    if index is None:
        # General knowledge response
        prompt = f"""
        You are a helpful assistant. Answer as concisely and to the point as possible while being safe and unbiased.
        
        Question: {query.question}
        Answer:
        """
    else:
        # Document-based response, retrieve top 1 chunk
        relevant_chunks = retrieve(query.question, k=1)
        context = "\n".join(relevant_chunks)
        prompt = f"""
        You are a helpful assistant. Answer should only be based on the document content provided, as concisely as possible.
        
        Relevant Document Content:
        {context}
        
        Question: {query.question}
        Answer:
        """
    
    response = llm.generate(prompt)
    return {"response": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
