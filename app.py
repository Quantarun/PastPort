from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import asyncio
import os
import tempfile
from dotenv import load_dotenv
import sys
import uvicorn
import json

# Load environment variables
load_dotenv()

# Add the current directory to the path so we can import the RAG module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the RAG chatbot class
from rag_module import RAGChatbot

app = FastAPI(title="Henry George Chatbot API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development - restrict this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize the chatbot
chatbot = RAGChatbot()

# Define request and response models
class QueryRequest(BaseModel):
    query: str
    namespaces: List[str] = []

class DocumentResponse(BaseModel):
    id: str
    name: str

class DocumentsResponse(BaseModel):
    documents: List[DocumentResponse]

# Routes
@app.get("/")
async def index():
    return FileResponse("static/index.html")

@app.post("/api/query")
async def query(request: Request):
    try:
        # Parse the request body manually
        body = await request.json()
        
        # Log the received request for debugging
        print(f"Received query request: {body}")
        
        user_query = body.get("query", "")
        namespaces = body.get("namespaces", [])
        
        if not user_query:
            raise HTTPException(status_code=400, detail="No query provided")

        if not namespaces:
            # Get all available namespaces if none specified
            namespaces = chatbot.vector_db.list_namespaces()
            print(f"Using all namespaces: {namespaces}")

        # Run the async query
        response, search_results, structured_response = await chatbot.query_async(namespaces, user_query)
        
        # Log the response structure (without the full content)
        print(f"Response generated. Type: {type(response)}")
        print(f"Search results count: {len(search_results)}")
        print(f"Structured response keys: {structured_response.keys() if isinstance(structured_response, dict) else 'Not a dict'}")
        
        # Ensure the response is JSON serializable
        if search_results and isinstance(search_results, list):
            # Convert any non-serializable objects in search_results
            for i, result in enumerate(search_results):
                if isinstance(result, dict):
                    for key, value in result.items():
                        if not isinstance(value, (str, int, float, bool, list, dict, type(None))):
                            result[key] = str(value)

        # Return the response in the format expected by the frontend
        return JSONResponse({
            "response": response,
            "search_results": search_results,
            "structured_response": structured_response
        })
        
    except Exception as e:
        import traceback
        print(f"Error processing query: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/api/documents")
async def get_documents():
    try:
        namespaces = chatbot.vector_db.list_namespaces()
        display_names = [ns.replace("book_", "").replace("_", " ").title() for ns in namespaces]

        documents = [{"id": ns, "name": display} for ns, display in zip(namespaces, display_names)]
        
        print(f"Retrieved documents: {documents}")
        return {"documents": documents}
    except Exception as e:
        import traceback
        print(f"Error getting documents: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error getting documents: {str(e)}")

@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        if not file:
            raise HTTPException(status_code=400, detail="No file part")

        if file.filename == '':
            raise HTTPException(status_code=400, detail="No selected file")

        print(f"Received file upload: {file.filename}")

        if file and file.filename.endswith('.pdf'):
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp:
                content = await file.read()
                temp.write(content)
                temp_path = temp.name

            # Generate a namespace from the filename
            namespace = "book_" + os.path.splitext(file.filename)[0].lower().replace(' ', '_')
            
            print(f"Processing PDF: {file.filename} with namespace: {namespace}")

            # Process the PDF
            result = chatbot.process_pdf(temp_path, namespace)

            # Clean up the temporary file
            os.unlink(temp_path)
            
            print(f"Upload successful: {result}")

            return {"success": True, "document": result}
        else:
            raise HTTPException(status_code=400, detail="Invalid file type. Only PDF files are supported.")
    except Exception as e:
        import traceback
        print(f"Error uploading document: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error uploading document: {str(e)}")

@app.get("/api/health")
async def health_check():
    return {"status": "ok", "service": "Henry George AI Chatbot"}

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
