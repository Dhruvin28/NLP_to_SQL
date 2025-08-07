import os
import json
import uuid
import faiss
import numpy as np
from typing import List, Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import mammoth
from sentence_transformers import SentenceTransformer
import groq
from pathlib import Path

# Initialize FastAPI app
app = FastAPI(title="Schema to SQL API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Open source embedding model
groq_client = groq.Groq(api_key="test replace from readme")  # Replace with your Groq API key

# Configuration
STORAGE_DIR = Path("storage")
FAISS_DIR = STORAGE_DIR / "faiss_indexes"
METADATA_FILE = STORAGE_DIR / "user_metadata.json"

# Create directories
STORAGE_DIR.mkdir(exist_ok=True)
FAISS_DIR.mkdir(exist_ok=True)

# Pydantic models
class QueryRequest(BaseModel):
    user_id: str
    query: str

class QueryResponse(BaseModel):
    sql_query: str
    matched_chunks: List[str]

class UploadResponse(BaseModel):
    user_id: str
    message: str
    chunks_count: int

# Utility functions
def load_user_metadata() -> Dict:
    """Load user metadata from JSON file"""
    if METADATA_FILE.exists():
        with open(METADATA_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_user_metadata(metadata: Dict):
    """Save user metadata to JSON file"""
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=2)

def extract_text_from_docx(file_content: bytes) -> str:
    """Extract text from DOCX file"""
    try:
        import io
        
        # Convert bytes to file-like object for mammoth
        file_like = io.BytesIO(file_content)
        
        # Try mammoth first for better formatting
        result = mammoth.extract_raw_text(file_like)
        text = result.value
        
        # If mammoth returns None or empty, try alternative approach
        if not text or not text.strip():
            # Alternative: Try using python-docx as backup
            try:
                from zipfile import ZipFile
                import xml.etree.ElementTree as ET
                
                # Reset file pointer for alternative method
                file_like.seek(0)
                
                # Manual extraction as backup
                with ZipFile(file_like) as docx:
                    # Try to extract from document.xml
                    xml_content = docx.read('word/document.xml')
                    root = ET.fromstring(xml_content)
                    
                    # Extract all text nodes
                    text_elements = []
                    for elem in root.iter():
                        if elem.text and elem.text.strip():
                            text_elements.append(elem.text.strip())
                    
                    text = ' '.join(text_elements)
            except Exception as backup_error:
                print(f"Backup extraction failed: {backup_error}")
                pass
        
        # Final check
        if not text or not text.strip():
            raise HTTPException(
                status_code=400, 
                detail="No readable text found in the DOCX file. Please ensure the document contains text content."
            )
        
        return text.strip()
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=400, 
            detail=f"Error extracting text from DOCX: {str(e)}. Please ensure you uploaded a valid DOCX file."
        )

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk.strip())
    
    return chunks

def create_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    """Create FAISS index from embeddings"""
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))
    return index

def search_similar_chunks(query: str, user_id: str, top_k: int = 5) -> List[str]:
    """Search for similar chunks using FAISS"""
    try:
        # Load user metadata
        metadata = load_user_metadata()
        if user_id not in metadata:
            raise HTTPException(status_code=404, detail="User ID not found")
        
        # Load FAISS index and chunks
        index_path = FAISS_DIR / f"{user_id}.index"
        chunks_path = FAISS_DIR / f"{user_id}_chunks.json"
        
        if not index_path.exists() or not chunks_path.exists():
            raise HTTPException(status_code=404, detail="User data not found")
        
        # Load index and chunks
        index = faiss.read_index(str(index_path))
        with open(chunks_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        # Validate chunks data
        if not chunks or len(chunks) == 0:
            raise HTTPException(status_code=404, detail="No chunks found for this user")
        
        # Embed query and search
        query_embedding = embedding_model.encode([query])
        distances, indices = index.search(query_embedding.astype('float32'), min(top_k, len(chunks)))
        
        # Return matched chunks, filtering out invalid indices
        matched_chunks = []
        for i in indices[0]:
            if 0 <= i < len(chunks):
                matched_chunks.append(chunks[i])
        
        return matched_chunks
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching chunks: {str(e)}")

def generate_sql_query(natural_query: str, schema_context: str) -> str:
    """Generate SQL query using Groq API"""
    try:
        prompt = f"""
Based on the following database schema information, convert the natural language query to SQL:

Database Schema Context:
{schema_context}

Natural Language Query: {natural_query}

Please provide only the SQL query without any explanations or formatting. Make sure the SQL is syntactically correct and uses the tables and columns from the provided schema.
Note : If query not related to Database Schema Context then return "Please enter correct query"
SQL Query:"""

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # Use Llama 3.2 model
            messages=[
                {"role": "system", "content": "You are a SQL expert. Convert natural language queries to SQL based on provided database schema."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.1
        )
        
        sql_query = response.choices[0].message.content.strip()
        return sql_query
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating SQL: {str(e)}")

# API Endpoints
@app.post("/upload-schema", response_model=UploadResponse)
async def upload_schema(file: UploadFile = File(...)):
    """Upload database schema in DOCX format and process it"""
    try:
        # Validate file type
        if not file.filename or not file.filename.lower().endswith('.docx'):
            raise HTTPException(status_code=400, detail="Only DOCX files are supported")
        
        # Check file size (limit to 10MB)
        file_content = await file.read()
        if len(file_content) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        
        if len(file_content) > 10 * 1024 * 1024:  # 10MB limit
            raise HTTPException(status_code=400, detail="File size too large. Maximum 10MB allowed.")
        
        # Generate unique user ID
        user_id = str(uuid.uuid4())
        
        # Extract text from DOCX
        text = extract_text_from_docx(file_content)
        
        # Additional validation
        if len(text.strip()) < 10:
            raise HTTPException(
                status_code=400, 
                detail="Document contains insufficient text. Please upload a document with meaningful schema content."
            )
        
        # Chunk the text
        chunks = chunk_text(text)
        
        if not chunks:
            raise HTTPException(
                status_code=400, 
                detail="No text chunks could be created from the document. Please check the document content."
            )
        
        # Filter out very short chunks
        chunks = [chunk for chunk in chunks if len(chunk.strip()) > 20]
        
        if not chunks:
            raise HTTPException(
                status_code=400, 
                detail="No meaningful text chunks found. Please ensure your document contains substantial schema information."
            )
        
        # Create embeddings
        try:
            embeddings = embedding_model.encode(chunks)
            if embeddings is None or len(embeddings) == 0:
                raise HTTPException(status_code=500, detail="Failed to create embeddings from text chunks")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error creating embeddings: {str(e)}")
        
        # Create FAISS index
        try:
            index = create_faiss_index(embeddings)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error creating vector index: {str(e)}")
        
        # Save FAISS index and chunks
        try:
            index_path = FAISS_DIR / f"{user_id}.index"
            chunks_path = FAISS_DIR / f"{user_id}_chunks.json"
            
            faiss.write_index(index, str(index_path))
            with open(chunks_path, 'w', encoding='utf-8') as f:
                json.dump(chunks, f, indent=2, ensure_ascii=False)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error saving index files: {str(e)}")
        
        # Update user metadata
        try:
            metadata = load_user_metadata()
            metadata[user_id] = {
                "filename": file.filename,
                "chunks_count": len(chunks),
                "file_size": len(file_content),
                "created_at": str(uuid.uuid1().time)
            }
            save_user_metadata(metadata)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error saving metadata: {str(e)}")
        
        return UploadResponse(
            user_id=user_id,
            message="Schema uploaded and processed successfully",
            chunks_count=len(chunks)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error processing file: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process natural language query and return SQL with matched schema chunks"""
    try:
        # Search for similar chunks
        matched_chunks = search_similar_chunks(request.query, request.user_id)
        
        if not matched_chunks:
            raise HTTPException(status_code=404, detail="No relevant schema information found")
        
        # Combine chunks for context
        schema_context = "\n\n".join(matched_chunks)
        
        # Generate SQL query
        sql_query = generate_sql_query(request.query, schema_context)
        
        return QueryResponse(
            sql_query=sql_query,
            matched_chunks=matched_chunks
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/users")
async def list_users():
    """List all users and their metadata"""
    metadata = load_user_metadata()
    return {"users": metadata}

@app.delete("/user/{user_id}")
async def delete_user(user_id: str):
    """Delete user data"""
    try:
        metadata = load_user_metadata()
        
        if user_id not in metadata:
            raise HTTPException(status_code=404, detail="User ID not found")
        
        # Delete FAISS files
        index_path = FAISS_DIR / f"{user_id}.index"
        chunks_path = FAISS_DIR / f"{user_id}_chunks.json"
        
        if index_path.exists():
            index_path.unlink()
        if chunks_path.exists():
            chunks_path.unlink()
        
        # Remove from metadata
        del metadata[user_id]
        save_user_metadata(metadata)
        
        return {"message": "User data deleted successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting user: {str(e)}")

@app.get("/")
async def serve_ui():
    """Serve the web UI"""
    return FileResponse('index.html')

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)