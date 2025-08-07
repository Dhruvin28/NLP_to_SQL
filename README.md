# FastAPI Schema to SQL Application

A standalone FastAPI application that converts natural language queries to SQL based on uploaded database schema documents.

## Features

- **Schema Upload**: Upload database schema in DOCX format
- **Text Processing**: Automatic chunking and embedding of schema content
- **Vector Search**: FAISS-based similarity search for relevant schema chunks
- **SQL Generation**: Natural language to SQL conversion using Groq's Llama 3.2 model
- **User Management**: Unique ID system with JSON-based metadata storage

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Get Groq API Key

1. Visit [Groq Console](https://console.groq.com/)
2. Create an account and get your API key
3. Replace `"your-groq-api-key-here"` in the main code with your actual API key

### 3. Run the Application

```bash
python main.py
```

The API will be available at `http://localhost:8000`

## API Endpoints

### 1. Upload Schema
- **Endpoint**: `POST /upload-schema`
- **Description**: Upload database schema in DOCX format
- **Input**: DOCX file
- **Output**: Unique user ID and processing status

```bash
curl -X POST "http://localhost:8000/upload-schema" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@Database schema.docx"
```

### 2. Process Query
- **Endpoint**: `POST /query`
- **Description**: Convert natural language to SQL
- **Input**: User ID and natural language query
- **Output**: SQL query and matched schema chunks

```bash
curl -X POST "http://localhost:8000/query" \
     -H "accept: application/json" \
     -H "Content-Type: application/json" \
     -d '{
       "user_id": "your-user-id",
       "query": "Show me all users with age greater than 25"
     }'
```

### 3. List Users
- **Endpoint**: `GET /users`
- **Description**: List all users and their metadata

### 4. Delete User
- **Endpoint**: `DELETE /user/{user_id}`
- **Description**: Delete user data and associated files

## Usage Example

1. **Upload Schema**: Upload a DOCX file containing your database schema
2. **Get User ID**: Save the returned unique user ID
3. **Query**: Send natural language queries with the user ID to get SQL responses

## File Structure

```
├── main.py                 # Main FastAPI application
├── requirements.txt        # Python dependencies
├── test_client.py         # Test script
├── storage/               # Storage directory (created automatically)
│   ├── faiss_indexes/     # FAISS index files
│   └── user_metadata.json # User metadata
```

## Schema Document Format

Your DOCX file should contain database schema information such as:

- Table names and descriptions
- Column names, types, and constraints
- Relationships between tables
- Indexes and keys
- Sample data or examples

Example schema content:
```
Users Table:
- id (INTEGER, PRIMARY KEY)
- username (VARCHAR(50), UNIQUE)
- email (VARCHAR(100))
- age (INTEGER)
- created_at (TIMESTAMP)

Orders Table:
- order_id (INTEGER, PRIMARY KEY)
- user_id (INTEGER, FOREIGN KEY references Users.id)
- total_amount (DECIMAL(10,2))
- order_date (DATE)
```

## Configuration

- **Embedding Model**: Uses `all-MiniLM-L6-v2` (open source)
- **LLM Model**: Groq's `llama-3.2-90b-text-preview`
- **Chunk Size**: 500 words with 50-word overlap
- **Vector Database**: FAISS (Facebook AI Similarity Search)
- **Storage**: JSON files (no external database required)

## Testing

Run the test client:

```bash
python test_client.py
```

Make sure you have a sample DOCX file named `sample_schema.docx` in the same directory.

## API Documentation

Once running, visit `http://localhost:8000/docs` for interactive API documentation (Swagger UI).

## Troubleshooting

1. **DOCX Reading Issues**: Ensure your DOCX file is valid and contains readable text
2. **Groq API Errors**: Check your API key and quota
3. **FAISS Errors**: Ensure numpy version compatibility
4. **File Permissions**: Make sure the application can create directories and files

## Notes

- The application creates a `storage` directory to store FAISS indexes and metadata
- Each user gets a unique ID that maps to their schema data
- Schema chunks are embedded and stored for efficient similarity search
- The system uses in-memory processing with file-based persistence