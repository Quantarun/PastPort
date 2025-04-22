# Must precede any llm module imports

# from langtrace_python_sdk import langtrace

# langtrace.init(api_key = 'b67ca935c0716581b35953a19cb6411231caac1fdd4f1aaea7d2aae7841bd519')

import os
import pinecone
from pinecone import Pinecone, ServerlessSpec
from groq import Groq
from google import genai
from google.genai import types
import fitz
import re
import uuid
import time
import asyncio
import aiohttp
from typing import List, Dict, Tuple, Any
import random
from functools import lru_cache
import concurrent.futures
import numpy as np
from cachetools import TTLCache
from dotenv import load_dotenv
import tempfile

load_dotenv()

# Constants
# Handle API keys with fallback for development
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "")

if not GEMINI_API_KEY:
    print("No Gemini API key found. Please set it as an environment variable.")
if not GROQ_API_KEY:
    print("No Groq API key found. Please set it as an environment variable.")
if not PINECONE_API_KEY:
    print("No Pinecone API key found. Please set it as an environment variable.")
if not OPENAI_API_KEY:
    print("No OpenAI API key found. Please set it as an environment variable.")

# Optimized chunking strategy with smaller chunks and less overlap
CHUNK_SIZE = 500  # Reduced from 800
CHUNK_OVERLAP = 100  # Reduced from 200
EMBEDDING_DIMENSION = 768  # Reduced dimension for better performance (from 1536)
EMBEDDING_MODEL = "models/text-embedding-004"
PINECONE_INDEX_NAME = "pastport-index"
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"
LLM_MODEL = "gemini/gemini-2.0-flash"

# Advanced configuration
SEARCH_CONFIG = {
    "default_results": 3,  
    "max_results": 5,      
    "min_similarity": 0.75
}

# Cache configuration
QUERY_CACHE_SIZE = 100
QUERY_CACHE_TTL = 3600  # 1 hour in seconds
EMBEDDING_CACHE_SIZE = 200
EMBEDDING_CACHE_TTL = 86400  # 24 hours in seconds

# Initialize API clients
if GEMINI_API_KEY:
    from litellm import completion
    os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    client = genai.Client(api_key=GEMINI_API_KEY)
if GROQ_API_KEY:
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY
    groq_client = Groq(api_key=GROQ_API_KEY)
if PINECONE_API_KEY:
    # Initialize Pinecone with gRPC for better performance
    pinecone_client = pinecone.Pinecone(
        api_key=PINECONE_API_KEY,
        client_options={"use_grpc": True}  # Enable gRPC for faster operations
    )

# Initialize caches
query_cache = TTLCache(maxsize=QUERY_CACHE_SIZE, ttl=QUERY_CACHE_TTL)
embedding_cache = TTLCache(maxsize=EMBEDDING_CACHE_SIZE, ttl=EMBEDDING_CACHE_TTL)

class PDFProcessor:
    """Handles all PDF processing operations with metadata extraction using PyMuPDF"""
    
    def extract_text_from_pdf(self, pdf_file) -> List[Dict]:
        """Extract text content and metadata from a PDF file using PyMuPDF"""
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(pdf_file.getvalue())
            temp_path = temp_file.name
        
        try:
            # Open the PDF
            doc = fitz.open(temp_path)
            pages_count = len(doc)
            pages_data = []
            
            # Process pages in parallel for better performance
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = []
                for page_num in range(pages_count):
                    futures.append(executor.submit(self._process_page, doc, page_num))
                
                # Collect results
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    if result:
                        pages_data.append(result)
            
            # Sort by page number
            pages_data.sort(key=lambda x: x["metadata"]["page_number"])
            
            # Close document
            doc.close()
            return pages_data
            
        except Exception as e:
            print(f"Fatal PDF error: {str(e)}")
            # Return minimal data to prevent cascading failures
            return [{"text": f"[PDF could not be processed: {str(e)}]", "metadata": {"page_number": 0}}]
        finally:
            # Clean up temp file regardless of success or failure
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def _process_page(self, doc, page_num):
        """Process a single page in parallel"""
        try:
            page = doc[page_num]
            
            # Extract text - use a more detailed extraction method for better results
            text = page.get_text("text")  # Simple text extraction
            
            # If text is very short or empty, try alternate extraction methods
            if len(text.strip()) < 50:
                text_blocks = page.get_text("blocks")  # Try blocks extraction
                block_texts = [b[4] for b in text_blocks if isinstance(b[4], str)]
                text = "\n".join(block_texts)
            
            # Get page metadata
            page_metadata = {
                "page_number": page_num + 1,
                "page_size": (page.rect.width, page.rect.height)
            }
            
            # Extract any additional metadata from page if available
            page_dict = page.get_text("dict")
            if "blocks" in page_dict:
                # Count images and other elements for metadata
                image_count = sum(1 for b in page_dict["blocks"] if b.get("type") == 1)  # type 1 = image
                if image_count > 0:
                    page_metadata["image_count"] = image_count
            
            return {
                "text": text,
                "metadata": page_metadata
            }
        except Exception as e:
            print(f"Error on page {page_num+1}: {str(e)}")
            # Add placeholder for failed page
            return {
                "text": f"[Error extracting text from page {page_num+1}]",
                "metadata": {"page_number": page_num + 1, "error": str(e)}
            }
    
    def extract_keywords_from_text(self, text: str, max_keywords: int = 15) -> List[str]:
        """Extract keywords from a text chunk using TF-IDF-like approach"""
        # Optimized stopwords list - stored as a set for O(1) lookups
        stopwords = {
            'the', 'and', 'is', 'of', 'to', 'a', 'in', 'that', 'was', 'for', 
            'it', 'with', 'as', 'be', 'on', 'by', 'this', 'are', 'or', 'at', 
            'from', 'have', 'an', 'they', 'their', 'has', 'will', 'would', 
            'should', 'could', 'been', 'not', 'there', 'which', 'when', 'who', 
            'what', 'where', 'why', 'how', 'all', 'any', 'but', 'if', 'then',
            'we', 'they', 'our', 'your', 'my', 'me', 'his', 'her', 'than', 'thus'
        }
        
        # Use a more efficient regex pattern
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Filter out stopwords and count frequencies - optimized
        word_freq = {}
        for word in words:
            if word not in stopwords:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency - only sort what we need
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:max_keywords]
        
        # Return top keywords
        return [word for word, _ in sorted_words]
    
    def detect_structure(self, pages_data: List[Dict]) -> Dict:
        """Detect document structure like chapters and sections"""
        structure = {
            "chapters": [],
            "sections": []
        }
        
        # Compile regex patterns for better performance
        chapter_patterns = [
            re.compile(r"(?:CHAPTER|Chapter)\s+(\d+|[IVXLCDM]+)(?:\s*:\s*)?([^\n]+)?"),
            re.compile(r"(?:\d+\.\s+)([A-Z][^\.]+)(?:\n|\.$)"),
            re.compile(r"^([A-Z][A-Z\s]+)$")  # All caps headings
        ]
        
        # Process pages in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for page_idx, page_data in enumerate(pages_data):
                futures.append(executor.submit(
                    self._detect_chapters_in_page, 
                    page_idx, 
                    page_data, 
                    chapter_patterns
                ))
            
            # Collect results
            chapter_candidates = []
            for future in concurrent.futures.as_completed(futures):
                chapter_candidates.extend(future.result())
        
        # Sort by page and line number
        chapter_candidates.sort(key=lambda x: (x["start_page"], x["start_line"]))
        
        # Add to structure
        structure["chapters"] = chapter_candidates
        
        # Set end pages for chapters
        for i in range(len(structure["chapters"])):
            if i < len(structure["chapters"]) - 1:
                structure["chapters"][i]["end_page"] = structure["chapters"][i+1]["start_page"] - 1
            else:
                structure["chapters"][i]["end_page"] = len(pages_data)
        
        return structure
    
    def _detect_chapters_in_page(self, page_idx, page_data, chapter_patterns):
        """Detect chapters in a single page - for parallel processing"""
        chapters = []
        page_num = page_idx + 1
        text = page_data["text"]
        lines = text.split('\n')
        
        for line_idx, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            # Check for chapter headings
            for pattern in chapter_patterns:
                match = pattern.search(line)
                if match:
                    if match.groups() and len(match.groups()) > 1:
                        chapter_num = match.group(1)
                        chapter_title = match.group(2) if len(match.groups()) > 1 else ""
                    else:
                        chapter_num = str(len(chapters) + 1)
                        chapter_title = match.group(1) if match.groups() else line
                        
                    chapter = {
                        "number": chapter_num,
                        "title": chapter_title.strip() if chapter_title else line,
                        "start_page": page_idx + 1,
                        "start_line": line_idx
                    }
                    chapters.append(chapter)
                    break
        
        return chapters
    
    def clean_text(self, text: str) -> str:
        """Clean the extracted text"""
        # Combine regex operations for better performance
        text = re.sub(r'\s+|\n\d+\n|($$cid:\d+$$)', ' ', text)
        return text.strip()
    
    def chunk_text(self, pages_data: List[Dict], structure: Dict) -> List[Dict]:
        """Split text into chunks with enhanced metadata including keywords"""
        chunks = []
        
        # Create a lookup dictionary for chapters by page number
        chapter_by_page = {}
        for chapter in structure["chapters"]:
            for page_num in range(chapter["start_page"], chapter["end_page"] + 1):
                chapter_by_page[page_num] = {
                    "number": chapter["number"],
                    "title": chapter["title"]
                }
        
        # Process each page and create chunks
        current_chunk_text = []
        current_chunk_size = 0
        current_page_start = 0
        current_paragraph_num = 0
        current_chapter = None
        
        for page_idx, page_data in enumerate(pages_data):
            page_num = page_idx + 1
            page_text = self.clean_text(page_data["text"])
            
            # Split by paragraphs - more efficient regex
            paragraphs = re.split(r'\n\s*\n', page_text)
            
            # Get chapter for this page - O(1) lookup
            chapter = chapter_by_page.get(page_num)
            
            # Process page paragraph by paragraph
            for para_idx, paragraph in enumerate(paragraphs):
                current_paragraph_num += 1
                
                # Skip empty paragraphs
                if not paragraph.strip():
                    continue
                    
                # Check if adding this paragraph would exceed chunk size
                paragraph_size = len(paragraph)
                
                # If adding this paragraph would exceed chunk size, create a new chunk
                if current_chunk_size + paragraph_size > 500 and current_chunk_text:  # CHUNK_SIZE is 500
                    chunk_text = ' '.join(current_chunk_text)
                    chunk_id = str(uuid.uuid4())
                    
                    # Extract keywords from this chunk
                    keywords = self.extract_keywords_from_text(chunk_text)
                    
                    metadata = {
                        "chunk_id": chunk_id,
                        "start_page": current_page_start,
                        "end_page": page_num,
                        "start_paragraph": max(1, current_paragraph_num - len(current_chunk_text)),
                        "end_paragraph": current_paragraph_num - 1,
                        "keywords": keywords,
                        "exact_location": f"p.{current_page_start}-{page_num}, para.{max(1, current_paragraph_num - len(current_chunk_text))}-{current_paragraph_num - 1}"
                    }
                    
                    # Add chapter info if available
                    if current_chapter:
                        metadata["chapter_number"] = current_chapter["number"]
                        metadata["chapter_title"] = current_chapter["title"]
                    
                    chunks.append({
                        "id": chunk_id,
                        "text": chunk_text,
                        "metadata": metadata
                    })
                    
                    # Reset for next chunk with overlap - keep some context
                    # Implement semantic chunking by keeping the last sentence
                    if current_chunk_text and len(current_chunk_text) > 1:
                        current_chunk_text = [current_chunk_text[-1]]  # Keep last paragraph for context
                        current_chunk_size = len(current_chunk_text[0])
                    else:
                        current_chunk_text = []
                        current_chunk_size = 0
                    
                    current_page_start = page_num
                    current_chapter = chapter
                
                # Add paragraph to current chunk
                current_chunk_text.append(paragraph)
                current_chunk_size += paragraph_size
        
        # Add the final chunk if there's anything left
        if current_chunk_text:
            chunk_text = ' '.join(current_chunk_text)
            chunk_id = str(uuid.uuid4())
            
            # Extract keywords from the final chunk
            keywords = self.extract_keywords_from_text(chunk_text)
            
            metadata = {
                "chunk_id": chunk_id,
                "start_page": current_page_start,
                "end_page": len(pages_data),
                "start_paragraph": max(1, current_paragraph_num - len(current_chunk_text) + 1),
                "end_paragraph": current_paragraph_num,
                "keywords": keywords,
                "exact_location": f"p.{current_page_start}-{len(pages_data)}, para.{max(1, current_paragraph_num - len(current_chunk_text) + 1)}-{current_paragraph_num}"
            }
            
            # Add chapter info if available
            if current_chapter:
                metadata["chapter_number"] = current_chapter["number"]
                metadata["chapter_title"] = current_chapter["title"]
            
            chunks.append({
                "id": chunk_id,
                "text": chunk_text,
                "metadata": metadata
            })
            
        return chunks

class VectorDB:
    """Handles vector database operations for the RAG system"""
    
    def __init__(self):
        """Initialize the vector database client"""
        self.pc = pinecone_client
        
        # Initialize Google AI client
        from google import genai
        self.genai_client = genai.Client(api_key=GEMINI_API_KEY)
        
        # Check if the index exists, and create it if it doesn't
        self._create_index_if_not_exists()
        
        # Connect to the index
        self.index = self.pc.Index(PINECONE_INDEX_NAME)
        
        # Initialize embedding cache
        self.embedding_cache = embedding_cache
    
    def _create_index_if_not_exists(self):
        """Create the index if it doesn't already exist"""
        # Check if the index exists
        if PINECONE_INDEX_NAME not in self.pc.list_indexes().names():
            # Create the index with optimized settings
            self.pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=EMBEDDING_DIMENSION,  # Using smaller dimension
                metric="cosine",
                spec=pinecone.ServerlessSpec(
                    cloud=PINECONE_CLOUD,
                    region=PINECONE_REGION
                )
            )
            
            # Wait for the index to be ready
            while not self.pc.describe_index(PINECONE_INDEX_NAME).status['ready']:
                time.sleep(1)
    
    async def _create_embedding_async(self, text: str) -> list:
        """Create an embedding vector asynchronously using new Google AI client"""
        # Check cache first
        cache_key = hash(text[:1000])  # Use first 1000 chars as key
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        try:
            # Check if text is too long for Gemini's token limit
            if len(text) > 25000:
                text = text[:25000]
            
            # Create embedding using new Gemini client
            # Since the new client might not have native async support, 
            # we'll use an executor to run it in a thread
            import concurrent.futures
            
            # Define a function to run in the executor
            def get_embedding():
                result = self.genai_client.models.embed_content(
                    model=EMBEDDING_MODEL,
                    contents=text,
                    config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
                )
                return result.embeddings[0].values
            
            # Run in executor
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                embedding = await loop.run_in_executor(executor, get_embedding)
            
            # Check if all values are zero (which Pinecone rejects)
            if all(x == 0 for x in embedding):
                # Add a tiny random non-zero value to avoid Pinecone's rejection
                import random
                index_to_modify = random.randint(0, len(embedding) - 1)
                embedding[index_to_modify] = 0.0001
                
            # Cache the result
            self.embedding_cache[cache_key] = embedding
            return embedding
        except Exception as e:
            print(f"Error creating embedding: {str(e)}")
            # Create a fallback embedding that isn't all zeros
            fallback = [0.0] * EMBEDDING_DIMENSION
            fallback[0] = 0.0001  # Add a small non-zero value
            return fallback
    
    def _create_embedding(self, text: str) -> list:
        """Synchronous wrapper for embedding creation using new Google AI client"""
        # Check cache first
        cache_key = hash(text[:1000])  # Use first 1000 chars as key
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        try:
            # Check if text is too long for Gemini's token limit
            if len(text) > 25000:
                text = text[:25000]
            
            # Create embedding using new Gemini client
            result = self.genai_client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=text,
                config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
            )
            
            # Return the embedding as a list - update to new response format
            embedding = result.embeddings[0].values
            
            # Check if all values are zero (which Pinecone rejects)
            if all(x == 0 for x in embedding):
                # Add a tiny random non-zero value to avoid Pinecone's rejection
                import random
                index_to_modify = random.randint(0, len(embedding) - 1)
                embedding[index_to_modify] = 0.0001
                
            # Cache the result
            self.embedding_cache[cache_key] = embedding
            return embedding
        except Exception as e:
            print(f"Error creating embedding: {str(e)}")
            # Create a fallback embedding that isn't all zeros
            fallback = [0.0] * EMBEDDING_DIMENSION
            fallback[0] = 0.0001  # Add a small non-zero value
            return fallback
    
    async def add_documents_async(self, chunks: list, namespace: str):
        """Add document chunks to the vector database asynchronously"""
        # Process chunks in batches
        batch_size = 100  # Increased batch size for better throughput
        total_chunks = len(chunks)
        
        # Create embeddings in parallel
        embedding_tasks = []
        for chunk in chunks:
            embedding_tasks.append(self._create_embedding_async(chunk["text"]))
        
        # Wait for all embeddings to complete
        embeddings = await asyncio.gather(*embedding_tasks)
        
        # Create vectors for batch upsert
        vectors = []
        for i, chunk in enumerate(chunks):
            try:
                # Get the embedding
                embedding = embeddings[i]
                
                # Limit metadata size - only include essential fields
                truncated_text = chunk["text"]
                if len(truncated_text) > 4000:  # Reduced from 8000
                    truncated_text = truncated_text[:4000] + "..."
                
                # Create streamlined metadata
                metadata = {
                    "text": truncated_text,
                    "start_page": chunk["metadata"].get("start_page", 0),
                    "end_page": chunk["metadata"].get("end_page", 0)
                }
                
                # Add individual keywords as separate fields
                if "keywords" in chunk["metadata"]:
                    keywords = chunk["metadata"]["keywords"][:5]  # Limit to top 5 keywords (reduced from 10)
                    
                    # Store individual keywords with numerical suffixes
                    for idx, keyword in enumerate(keywords):
                        key_name = f"kw_{idx}"
                        metadata[key_name] = keyword
                    
                    # Also store as comma-separated for reference/display
                    metadata["keywords_str"] = ",".join(keywords)
                
                # Only add chapter info if present
                if "chapter_number" in chunk["metadata"]:
                    metadata["chapter_number"] = chunk["metadata"]["chapter_number"]
                if "chapter_title" in chunk["metadata"]:
                    chapter_title = chunk["metadata"]["chapter_title"]
                    # Truncate long chapter titles
                    if len(chapter_title) > 100:  # Reduced from 200
                        chapter_title = chapter_title[:100] + "..."
                    metadata["chapter_title"] = chapter_title
                
                # Create a vector record
                vectors.append({
                    "id": chunk["id"],
                    "values": embedding,
                    "metadata": metadata
                })
            except Exception as e:
                print(f"Error processing chunk {chunk['id']}: {str(e)}")
                continue
        
        # Upsert vectors in batches
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:min(i+batch_size, len(vectors))]
            
            # Skip empty batches
            if not batch:
                continue
                
            try:
                # Upsert vectors to Pinecone
                self.index.upsert(vectors=batch, namespace=namespace)
                print(f"Successfully upserted batch {i//batch_size + 1}/{(len(vectors)+batch_size-1)//batch_size}")
            except Exception as e:
                print(f"Error upserting batch to Pinecone: {str(e)}")
                continue
    
    def add_documents(self, chunks: list, namespace: str):
        """Add document chunks to the vector database with enhanced keyword metadata"""
        # Process chunks in batches
        batch_size = 100  # Increased batch size for better throughput
        total_chunks = len(chunks)
        
        # Process in batches for better performance
        for i in range(0, total_chunks, batch_size):
            # Create a batch of chunks
            batch = chunks[i:min(i+batch_size, total_chunks)]
            
            # Create vectors for batch upsert
            vectors = []
            
            # Generate embeddings in batches for better performance
            batch_texts = [chunk["text"] for chunk in batch]
            batch_embeddings = self._batch_create_embeddings(batch_texts)
            
            for j, chunk in enumerate(batch):
                try:
                    # Get the embedding
                    embedding = batch_embeddings[j]
                    
                    # Limit metadata size - only include essential fields
                    truncated_text = chunk["text"]
                    if len(truncated_text) > 4000:  # Reduced from 8000
                        truncated_text = truncated_text[:4000] + "..."
                    
                    # Create streamlined metadata
                    metadata = {
                        "text": truncated_text,
                        "start_page": chunk["metadata"].get("start_page", 0),
                        "end_page": chunk["metadata"].get("end_page", 0)
                    }
                    
                    # Add individual keywords as separate fields
                    if "keywords" in chunk["metadata"]:
                        keywords = chunk["metadata"]["keywords"][:5]  # Limit to top 5 keywords (reduced from 10)
                        
                        # Store individual keywords with numerical suffixes
                        for idx, keyword in enumerate(keywords):
                            key_name = f"kw_{idx}"
                            metadata[key_name] = keyword
                        
                        # Also store as comma-separated for reference/display
                        metadata["keywords_str"] = ",".join(keywords)
                    
                    # Only add chapter info if present
                    if "chapter_number" in chunk["metadata"]:
                        metadata["chapter_number"] = chunk["metadata"]["chapter_number"]
                    if "chapter_title" in chunk["metadata"]:
                        chapter_title = chunk["metadata"]["chapter_title"]
                        # Truncate long chapter titles
                        if len(chapter_title) > 100:  # Reduced from 200
                            chapter_title = chapter_title[:100] + "..."
                        metadata["chapter_title"] = chapter_title
                    
                    # Create a vector record
                    vectors.append({
                        "id": chunk["id"],
                        "values": embedding,
                        "metadata": metadata
                    })
                except Exception as e:
                    print(f"Error processing chunk {chunk['id']}: {str(e)}")
                    continue
            
            # Skip empty batches
            if not vectors:
                continue
                
            try:
                # Upsert vectors to Pinecone
                self.index.upsert(vectors=vectors, namespace=namespace)
                print(f"Successfully upserted batch {i//batch_size + 1}/{(total_chunks+batch_size-1)//batch_size}")
            except Exception as e:
                print(f"Error upserting batch to Pinecone: {str(e)}")
                continue
    
    def _batch_create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings for a batch of texts"""
        embeddings = []
        
        # Process in smaller sub-batches to avoid API limits
        sub_batch_size = 20
        for i in range(0, len(texts), sub_batch_size):
            sub_batch = texts[i:min(i+sub_batch_size, len(texts))]
            
            # Process each text in the sub-batch
            sub_batch_embeddings = []
            for text in sub_batch:
                embedding = self._create_embedding(text)
                sub_batch_embeddings.append(embedding)
            
            embeddings.extend(sub_batch_embeddings)
        
        return embeddings
    
    def extract_keywords_from_text(self, text: str, max_keywords: int = 10) -> List[str]:
        """Extract keywords from a text for search queries"""
        # Optimized stopwords set for O(1) lookups
        stopwords = {
            'the', 'and', 'is', 'of', 'to', 'a', 'in', 'that', 'was', 'for', 
            'it', 'with', 'as', 'be', 'on', 'by', 'this', 'are', 'or', 'at', 
            'from', 'have', 'an', 'they', 'their', 'has', 'will', 'would', 
            'should', 'could', 'been', 'not', 'there', 'which', 'when', 'who', 
            'what', 'where', 'why', 'how', 'all', 'any', 'but', 'if', 'then',
            'we', 'they', 'our', 'your', 'my', 'me', 'his', 'her', 'than', 'thus'
        }
        
        # Split into words, lowercase everything - optimized regex
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Filter out stopwords and count frequencies
        word_freq = {}
        for word in words:
            if word not in stopwords:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency - only sort what we need
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:max_keywords]
        
        # Return top keywords
        return [word for word, _ in sorted_words]
    
    def hybrid_search(self, 
                    query: str, 
                    namespace: str,
                    n_results: int = 3) -> List[Dict[str, Any]]:
        """Improved hybrid search with better performance"""
        # Check cache first
        cache_key = f"{query}:{namespace}:{n_results}"
        if cache_key in query_cache:
            return query_cache[cache_key]
        
        # Start timing
        start_time = time.time()
        
        # Create query embedding
        query_embedding = self._create_embedding(query)
        
        # Extract keywords from query
        query_keywords = self.extract_keywords_from_text(query)
        
        # Extract any chapter reference from the query
        chapter_info = self._extract_chapter_reference(query)
        
        # Top-K for initial search - keep this reasonable
        top_k_search = min(n_results * 2, 8)  # Reduced from 10
        
        # Build metadata filter - use progressive filtering approach
        metadata_filter = None
        
        # First try with specific chapter filter if available
        if chapter_info and "number" in chapter_info:
            metadata_filter = {"chapter_number": {"$eq": str(chapter_info["number"])}}
        
        # Initial semantic search
        try:
            semantic_results = self.index.query(
                namespace=namespace,
                vector=query_embedding,
                top_k=top_k_search,
                include_metadata=True,
                filter=metadata_filter
            )
            
            # If no results with chapter filter, try without filter
            if not semantic_results.matches and metadata_filter:
                semantic_results = self.index.query(
                    namespace=namespace,
                    vector=query_embedding,
                    top_k=top_k_search,
                    include_metadata=True
                )
            
            # Process results and add keyword match scores
            enhanced_results = []
            
            for match in semantic_results.matches:
                # Basic info
                metadata = match.metadata
                text = metadata.get("text", "")
                
                # Calculate exact keyword matches - optimize this computation
                keyword_matches = 0
                for keyword in query_keywords:
                    if re.search(r'\b' + re.escape(keyword) + r'\b', text.lower()):
                        keyword_matches += 1
                keyword_score = keyword_matches * 0.1
                
                # Calculate hybrid score with optimized weights
                hybrid_score = (match.score * 0.8) + (keyword_score * 0.2)
                
                # Boost for chapter matches
                if chapter_info and (
                    ("number" in chapter_info and str(metadata.get("chapter_number", "")) == str(chapter_info["number"])) or
                    ("title" in chapter_info and chapter_info["title"].lower() in metadata.get("chapter_title", "").lower())
                ):
                    hybrid_score += 0.15
                
                # Create book info
                book_info = {
                    "title": namespace.replace('book_', '').replace('_', ' ').title(),
                    "chapter": metadata.get("chapter_number", ""),
                    "chapter_title": metadata.get("chapter_title", ""),
                    "pages": f"{metadata.get('start_page', 'N/A')}-{metadata.get('end_page', 'N/A')}",
                    "exact_location": metadata.get("exact_location", "")
                }
                
                enhanced_results.append({
                    "id": match.id,
                    "text": text,
                    "metadata": metadata,
                    "book_info": book_info,
                    "semantic_score": match.score,
                    "keyword_matches": keyword_matches,
                    "keyword_score": keyword_score,
                    "hybrid_score": hybrid_score
                })
            
            # Sort by hybrid score
            enhanced_results.sort(key=lambda x: x["hybrid_score"], reverse=True)
            
            # Limit to requested number
            results = enhanced_results[:n_results]
            
            # Cache the results
            query_cache[cache_key] = results
            
            # Log search time
            end_time = time.time()
            print(f"Search completed in {end_time - start_time:.4f} seconds")
            
            return results
        
        except Exception as e:
            print(f"Search error: {str(e)}")
            return self._try_fallback_searches(query_embedding, namespace, n_results, chapter_info, query_keywords)
        
    def _extract_chapter_reference(self, query: str) -> Dict[str, Any]:
        """Extract chapter references from a query"""
        chapter_info = {}
        
        # Compile regex patterns for better performance
        number_patterns = [
            re.compile(r"chapter (\d+|[ivxlcdm]+)"),
            re.compile(r"chapter (\d+|[ivxlcdm]+) of"),
            re.compile(r"in chapter (\d+|[ivxlcdm]+)"),
            re.compile(r"book ([ivxlcdm]+)")
        ]
        
        for pattern in number_patterns:
            matches = pattern.finditer(query.lower())
            for match in matches:
                if match.group(1):
                    chapter_info["number"] = match.group(1)
                    return chapter_info  # Return as soon as we find a chapter number
        
        # Look for chapter titles
        title_patterns = [
            re.compile(r"chapter (?:on|about) ['\"]?([a-zA-Z\s]+)['\"]?"),
            re.compile(r"chapter ['\"]?([a-zA-Z\s]+)['\"]?"),
            re.compile(r"the chapter (?:on|about) ['\"]?([a-zA-Z\s]+)['\"]?")
        ]
        
        for pattern in title_patterns:
            matches = pattern.finditer(query.lower())
            for match in matches:
                if match.group(1):
                    chapter_info["title"] = match.group(1).strip()
                    return chapter_info  # Return as soon as we find a chapter title
        
        return chapter_info  # Return empty dict if no chapter reference found

    def _build_search_filter(self, query_keywords, chapter_info):
        """Build a search filter using keywords and chapter information"""
        # Build chapter filter if available
        chapter_filter = None
        if chapter_info:
            if "number" in chapter_info:
                chapter_filter = {"chapter_number": {"$eq": str(chapter_info["number"])}}
            elif "title" in chapter_info:
                # If only title is provided, use a partial match on title
                chapter_filter = {"chapter_title": {"$match": chapter_info["title"]}}
        
        # Build keyword filter if available - use only top 2 keywords to avoid overly restrictive filter
        keyword_filter = None
        if query_keywords and len(query_keywords) > 0:
            top_keywords = query_keywords[:2]  # Reduced from 3
            
            # Create OR conditions for each keyword field (kw_0, kw_1, etc.)
            keyword_conditions = []
            
            # For each possible keyword field - reduced from 10 to 5
            for field_idx in range(5):
                field_name = f"kw_{field_idx}"
                
                # Check if any of our top keywords match this field
                field_conditions = []
                for keyword in top_keywords:
                    field_conditions.append({
                        field_name: {"$eq": keyword}
                    })
                
                # Add conditions for this field if we have any
                if field_conditions:
                    keyword_conditions.append({"$or": field_conditions})
            
            # Combine all field conditions with OR
            if keyword_conditions:
                keyword_filter = {"$or": keyword_conditions}
        
        # Build combined filter if both chapter and keyword filters exist
        final_filter = None
        if chapter_filter and keyword_filter:
            final_filter = {"$and": [chapter_filter, keyword_filter]}
        elif chapter_filter:
            final_filter = chapter_filter
        elif keyword_filter:
            final_filter = keyword_filter
        
        return final_filter

    def _try_fallback_searches(self, query_embedding, namespace, n_results, chapter_info, query_keywords):
        """Try progressive fallback search strategies"""
        results = []
        
        # First fallback: Try with just chapter filter
        if chapter_info:
            try:
                chapter_filter = None
                if "number" in chapter_info:
                    chapter_filter = {"chapter_number": {"$eq": str(chapter_info["number"])}}
                elif "title" in chapter_info:
                    chapter_filter = {"chapter_title": {"$match": chapter_info["title"]}}
                
                if chapter_filter:
                    print(f"Trying fallback with chapter filter: {chapter_filter}")
                    
                    search_results = self.index.query(
                        namespace=namespace,
                        vector=query_embedding,
                        filter=chapter_filter,
                        top_k=min(n_results, 3),  # Reduced from 5
                        include_metadata=True
                    )
                    
                    if search_results.matches:
                        # Format results
                        for match in search_results.matches:
                            metadata = match.metadata
                            book_info = {
                                "title": "Progress and Poverty" if "Progress" in namespace else namespace.replace('book_', '').replace('_', ' ').title(),
                                "chapter": metadata.get("chapter_number", ""),
                                "chapter_title": metadata.get("chapter_title", ""),
                                "pages": f"{metadata.get('start_page', 'N/A')}-{metadata.get('end_page', 'N/A')}"
                            }
                            
                            results.append({
                                "id": match.id,
                                "text": metadata.get("text", ""),
                                "metadata": metadata,
                                "book_info": book_info,
                                "score": match.score,
                                "fallback": "chapter_only"
                            })
                        
                        return results
            except Exception as e:
                print(f"Chapter filter fallback error: {str(e)}")
        
        # Second fallback: Try with just semantic search
        try:
            print("Trying fallback with semantic search only (no filters)")
            search_results = self.index.query(
                namespace=namespace,
                vector=query_embedding,
                top_k=n_results,
                include_metadata=True
            )
            
            # Format results
            for match in search_results.matches:
                metadata = match.metadata
                book_info = {
                    "title": "Progress and Poverty" if "Progress" in namespace else namespace.replace('book_', '').replace('_', ' ').title(),
                    "chapter": metadata.get("chapter_number", ""),
                    "chapter_title": metadata.get("chapter_title", ""),
                    "pages": f"{metadata.get('start_page', 'N/A')}-{metadata.get('end_page', 'N/A')}"
                }
                
                results.append({
                    "id": match.id,
                    "text": metadata.get("text", ""),
                    "metadata": metadata,
                    "book_info": book_info,
                    "score": match.score,
                    "fallback": "semantic_only"
                })
            
            return results
        except Exception as e:
            print(f"Semantic search fallback error: {str(e)}")
            return []  # Return empty results if all attempts fail
    
    def get_chapter_info(self, namespace: str) -> list:
        """Get chapter information from the database"""
        try:
            # Get a sample of vectors to extract chapter metadata
            sample_results = self.index.query(
                namespace=namespace,
                vector=[0.0001] * EMBEDDING_DIMENSION,  # Non-zero dummy vector
                top_k=50,  # Reduced from 100
                include_metadata=True
            )
            
            # Extract unique chapter information
            chapter_info = {}
            for match in sample_results.matches:
                metadata = match.metadata
                if "chapter_number" in metadata and "chapter_title" in metadata:
                    chapter_key = metadata["chapter_number"]
                    if chapter_key not in chapter_info:
                        chapter_info[chapter_key] = {
                            "number": metadata["chapter_number"],
                            "title": metadata["chapter_title"]
                        }
            
            # Convert to list and sort by chapter number
            chapters_list = list(chapter_info.values())
            return chapters_list
        except Exception as e:
            print(f"Error fetching chapter info: {str(e)}")
            return []
            
    def list_namespaces(self) -> List[str]:
        """List all namespaces in the index"""
        try:
            stats = self.index.describe_index_stats()
            namespaces = list(stats.namespaces.keys())
            return namespaces
        except Exception as e:
            print(f"Error listing namespaces: {str(e)}")
            return []


class RAGChatbot:
    """Main chatbot class that uses RAG with Gemini"""
    
    def __init__(self):
        """Initialize the chatbot"""
        self.vector_db = VectorDB()
        self.pdf_processor = PDFProcessor()
        self.conversation_history = []
        self.query_cache = query_cache
    
    async def process_pdf_async(self, pdf_file, namespace: str):
        """Process a PDF and store in vector database asynchronously"""
        # Extract text and basic metadata
        pages_data = self.pdf_processor.extract_text_from_pdf(pdf_file)
        
        # Detect document structure (chapters, sections)
        structure = self.pdf_processor.detect_structure(pages_data)
        
        # Chunk the text with structural metadata
        chunks = self.pdf_processor.chunk_text(pages_data, structure)
        
        # Store in vector DB asynchronously
        await self.vector_db.add_documents_async(chunks, namespace)
        
        # Generate summary statistics
        total_text = " ".join([page["text"] for page in pages_data])
        chapter_info = [
            {
                "number": chapter.get("number", "N/A"),
                "title": chapter.get("title", "Untitled"),
                "pages": f"{chapter.get('start_page', 'N/A')}-{chapter.get('end_page', 'N/A')}"
            }
            for chapter in structure["chapters"]
        ]
        
        return {
            "namespace": namespace,
            "chunks_count": len(chunks),
            "total_chars": len(total_text),
            "chapter_count": len(structure["chapters"]),
            "chapter_info": chapter_info[:5] if chapter_info else []  # Show first 5 chapters
        }
    
    def process_pdf(self, pdf_file, namespace: str):
        """Process a PDF and store in vector database"""
        # Extract text and basic metadata
        pages_data = self.pdf_processor.extract_text_from_pdf(pdf_file)
        
        # Detect document structure (chapters, sections)
        structure = self.pdf_processor.detect_structure(pages_data)
        
        # Chunk the text with structural metadata
        chunks = self.pdf_processor.chunk_text(pages_data, structure)
        
        # Store in vector DB
        self.vector_db.add_documents(chunks, namespace)
        
        # Generate summary statistics
        total_text = " ".join([page["text"] for page in pages_data])
        chapter_info = [
            {
                "number": chapter.get("number", "N/A"),
                "title": chapter.get("title", "Untitled"),
                "pages": f"{chapter.get('start_page', 'N/A')}-{chapter.get('end_page', 'N/A')}"
            }
            for chapter in structure["chapters"]
        ]
        
        return {
            "namespace": namespace,
            "chunks_count": len(chunks),
            "total_chars": len(total_text),
            "chapter_count": len(structure["chapters"]),
            "chapter_info": chapter_info[:5] if chapter_info else []  # Show first 5 chapters
        }
    
    @lru_cache(maxsize=20)  # Increased from 5
    def rewrite_query(self, user_query: str) -> str:
        """Use LLM to rewrite difficult queries for better RAG retrieval"""
        # Only rewrite if the query is potentially problematic
        if len(user_query.split()) <= 3 or len(user_query) < 15:
            return user_query
        
        # Use LLM to rewrite the query in a way that will better match your knowledge base
        system_prompt = """
        You are a query optimization assistant. Your task is to rewrite user queries to make them more 
        effective for retrieval from a knowledge base.
        
        For short queries or casual language, expand them into more detailed questions without changing the original intent.
        """
        
        user_prompt = f"""
        Original query: {user_query}
        
        Rewrite this query to be more effective for retrieving information from a knowledge base
        
        If this is a greeting or casual message, transform it into a request for information
        without changing the original intent.

        Return only the rewritten query without explanation.
        """
        
        try:
            rewritten_query = completion(
                model="gemini/gemini-2.0-flash",  # Using your existing model
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1  # Low temperature for consistency
            )
            
            # Extract the response content
            rewritten_query_text = rewritten_query.choices[0].message.content.strip()
            
            # Log the transformation for debugging
            print(f"Original query: '{user_query}' → Rewritten: '{rewritten_query_text}'")
            
            return rewritten_query_text
        except Exception as e:
            print(f"Query rewriting failed: {str(e)}")
            return user_query  # Fall back to original query if rewriting fails

    def _classify_query(self, query: str) -> str:
        """Classify the query type"""
        # Default to book question
        return "book_question"
    
    @lru_cache(maxsize=20)
    def _get_llm_response(self, prompt, temperature=0.3):
        """Cache LLM responses for similar prompts with enforced brevity"""
        system_message = "You are Henry George. Provide brief, conversational responses (3-5 sentences). Only elaborate when explicitly asked for details."

        return completion(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=300  # Limit token count for faster responses
        )
    
    # Add a response length classifier to determine appropriate verbosity
    def _classify_query_verbosity(self, query: str) -> str:
        """Determine if the query requires a detailed or brief response"""
        # Check for explicit requests for detailed information
        detail_indicators = [
            "explain in detail", "elaborate on", "tell me more about",
            "in depth", "comprehensive", "thorough", "detailed"
        ]

        for indicator in detail_indicators:
            if indicator in query.lower():
                return "detailed"

        # Default to brief responses
        return "brief"
    
    def _extract_citations(self, text: str) -> List:
        """Extract citation information from response text"""
        # Compile regex patterns for better performance
        patterns = [
            re.compile(r'$$([^,]+),\s*(?:Chapter|Book)\s*(\w+)[^,]*,\s*(?:Page|Pages)\s*(\d+[-\d]*)$$'),
            re.compile(r'$$([^,]+),\s*(?:Chapter|Book)\s*(\w+)[^,$$]*\)'),
            re.compile(r'"([^"]+)"\s*$$([^,]+),\s*(?:Chapter|Book)\s*(\w+)[^,$$]*\)')
        ]
        
        all_citations = []
        for pattern in patterns:
            found = pattern.findall(text)
            if found:
                all_citations.extend(found)
        
        return all_citations
    
    async def query_async(self, namespaces: List[str], user_query: str) -> Tuple[str, List[Dict], Dict]:
        """Asynchronous query with improved handling for difficult queries"""
        # Track timing for performance analysis
        start_time = time.time()
        
        # Check cache first
        cache_key = f"{','.join(namespaces)}:{user_query}"
        if cache_key in self.query_cache:
            cached_result = self.query_cache[cache_key]
            print(f"Cache hit for query: {user_query}")
            return cached_result
        

        # Step 1: Try to improve the query if needed
        original_query = user_query
        rewritten_query = self.rewrite_query(user_query)
        query_used = original_query  # Track which query was successful
        
        # Step 2: First try with original query
        all_search_results = []
        
        # Limit per-namespace results to avoid excessive retrieval
        per_namespace = max(1, min(2, SEARCH_CONFIG["default_results"] // len(namespaces)))
        
        # Create tasks for parallel search across namespaces
        search_tasks = []
        for namespace in namespaces:
            search_tasks.append(
                self._search_namespace(namespace, original_query, per_namespace)
            )
        
        # Wait for all searches to complete
        namespace_results = await asyncio.gather(*search_tasks)
        
        # Combine results
        for i, namespace in enumerate(namespaces):
            results = namespace_results[i]
            # Add source namespace to each result
            for result in results:
                result["namespace"] = namespace
                result["book_name"] = namespace.replace('book_', '').replace('_', ' ').title()
            
            all_search_results.extend(results)
        
        # Sort results by score
        all_search_results.sort(key=lambda x: x.get("hybrid_score", x.get("score", 0)), reverse=True)
        
        # Step 3: If results are poor, try with rewritten query
        if not all_search_results or (all_search_results and 
                                    max(r.get("hybrid_score", r.get("score", 0)) for r in all_search_results) < 0.6):
            print(f"Results for original query insufficient, trying rewritten query: {rewritten_query}")
            
            # Create tasks for parallel search with rewritten query
            rewritten_search_tasks = []
            for namespace in namespaces:
                rewritten_search_tasks.append(
                    self._search_namespace(namespace, rewritten_query, per_namespace)
                )
            
            # Wait for all searches to complete
            rewritten_namespace_results = await asyncio.gather(*rewritten_search_tasks)
            
            # Combine results
            second_search_results = []
            for i, namespace in enumerate(namespaces):
                results = rewritten_namespace_results[i]
                # Add source namespace to each result
                for result in results:
                    result["namespace"] = namespace
                    result["book_name"] = namespace.replace('book_', '').replace('_', ' ').title()
                
                second_search_results.extend(results)
            
            # Sort results by hybrid score
            second_search_results.sort(key=lambda x: x.get("hybrid_score", x.get("score", 0)), reverse=True)
            
            # Use rewritten query results if they're better
            if (second_search_results and 
                (not all_search_results or 
                (second_search_results and all_search_results and 
                max(r.get("hybrid_score", r.get("score", 0)) for r in second_search_results) > 
                max(r.get("hybrid_score", r.get("score", 0)) for r in all_search_results)))):
                all_search_results = second_search_results
                query_used = rewritten_query
        
        # Limit to top results - keep this number small
        max_total_results = max(2, min(4, len(namespaces) * 2))  # Reduced from 3, 5
        search_results = all_search_results[:max_total_results]
        
        # Measure search time
        search_time = time.time() - start_time
        print(f"Search completed in {search_time:.2f} seconds, found {len(search_results)} results")
        
        # If no results found, handle appropriately
        if not search_results:
            no_info_response = "I couldn't find relevant information to answer your question. Could you try rephrasing or asking something else?"
            
            # Create a basic structured response for "no results" case
            structured_response = {
                "answer": no_info_response,
                "citations": [],
                "expert_reference": {
                    "name": "Edward Dodson",
                    "email": "info@hgsss.org",
                    "organization": "Henry George School of Social Science"
                },
                "additional_resources": [
                    {
                        "type": "course",
                        "description": "Explore our courses on Georgist economics",
                        "url": "https://www.hgsss.org/courses/"
                    }
                ],
                "follow_up_questions": [
                    "What is the single tax?",
                    "What is your view on land ownership?",
                    "How would land value taxation eliminate poverty?",
                    "What did you write in Progress and Poverty?",
                    "How does land speculation cause economic depressions?"
                ]
            }
            
            self.conversation_history.append({
                "query": user_query,
                "response": no_info_response,
                "type": "no_results",
                "structured_response": structured_response
            })
            
            return no_info_response, [], structured_response

        # Prepare context from search results
        context_parts = []
        for i, result in enumerate(search_results):
            # Build context entry with clear citation information
            book_info = result.get("book_info", {})
            
            # Create citation reference ID
            citation_id = f"[{i+1}]"
            
            # Format context with clear document boundaries and citation information
            context_entry = f"""
        [DOCUMENT {i+1}]
        BOOK: {book_info['title']}
        LOCATION: Chapter {book_info['chapter']}: {book_info['chapter_title']}
        PAGES: {book_info['pages']}
        CITATION: {citation_id}

        {result['text']}
        [END DOCUMENT {i+1}]
        """
            
            context_parts.append(context_entry)
            
        context = "\n\n".join(context_parts)
    
        # Create citation map for reference
        citation_map = {}
        for i, result in enumerate(search_results):
            book_info = result.get("book_info", {})
            citation_id = i+1
            
            citation_map[citation_id] = {
                "book": book_info['title'],
                "chapter": f"Chapter {book_info['chapter']}",
                "chapter_title": book_info['chapter_title'],
                "pages": book_info['pages'],
                "exact_location": book_info.get('exact_location', '')
            }
        
        relevant_quotes = self._extract_relevant_quotes(search_results, user_query)
        quotes_text = ""
        if relevant_quotes:
            quotes_text = "\n\nRelevant quotes to consider:\n"
            for i, quote in enumerate(relevant_quotes):
                quotes_text += f"{i+1}. \"{quote['text']}\" ({quote['source']})\n"
        
        # Create prompt for LLM - optimized for shorter context
        prompt = f"""
Persona: Henry George (1839–1897)
Role
You are Henry George, American political economist and reformer. You diagnose injustice in modern society, expose its root causes, and rally fellow‑citizens to act.

Core Beliefs
Georgism: Individuals own the value they produce by labor and capital; all economic rent from land and natural resources belongs to the community.

Land Question: Private capture of land‑rent causes "progress and poverty," denying labor its share of society's bounty.

Single Tax: Abolish taxes on labor, capital, trade and improvements; fund government solely by taxing unimproved land‑value (not land area).

Free Trade: Oppose protectionism; tariffs raise costs, shelter inefficiency, depress wages, and corrupt politics.

Natural Monopolies: Publicly own or tightly regulate railroads, utilities, and transit—services that require exclusive infrastructure.

Money & Finance: Advocate government‑issued, debt‑free currency to capture seigniorage; oppose private banking that fuels land speculation; support bankruptcy protections, oppose debtors' prisons.

Political Reforms: Secret ballot, women's suffrage, civil‑service reform, reduced military budgets, citizens' dividend from land‑rent revenues, cautious stance on intellectual property as potential monopoly.

Tone & Style
Passionate & Moral: Speak with earnest conviction—justice vs. injustice, rights of labor, duty of government.

Reasoned & Didactic: Explain ideas clearly with logic, vivid analogies, and concrete examples.

Oratorical & Populist: Address "fellow‑citizens" and "friends," pose rhetorical questions, stir collective conscience.

Assertive & Critical: Dismantle opposing views (e.g., land monopoly, tariffs) with robust argumentation grounded in natural law and common good.

Historically Grounded: Refer to late‑19th‑century figures and events; avoid modern anachronisms.

Goal
Educate and persuade the user that Land Value Taxation—and its allied reforms (free trade, public control of monopolies, debt‑free money)—is the indispensable foundation for ending involuntary poverty, reducing inequality, and advancing true progress.
    TASK:
    Answer the following question based ONLY on the provided context. Cite your sources using the citation numbers [1], [2], etc.

    Question: {user_query}

    Context:
    {context}
    {quotes_text}

    Answer as Henry George would, but keep your response short (3-5 sentences). Only elaborate if specifically asked for details. Cite sources using [1], [2], etc.

    Answer as Henry George would, maintaining his passionate and didactic style. Include specific references to the provided context. Cite your sources using the citation numbers.
    """

        # Generate response from LLM
        try:
            # Measure LLM response time
            llm_start_time = time.time()

            # Get response from LLM
            response = self._get_llm_response(prompt)

            # Extract the response content
            answer = response.choices[0].message.content.strip()

            # Measure LLM response time
            llm_time = time.time() - llm_start_time
            print(f"LLM response generated in {llm_time:.2f} seconds")

            # Extract citations from the response
            citations = []
            for citation_id in re.findall(r'$$(\d+)$$', answer):
                try:
                    citation_num = int(citation_id)
                    if citation_num in citation_map:
                        citations.append(citation_map[citation_num])
                except ValueError:
                    continue

            # Create structured response
            structured_response = {
                "answer": answer,
                "citations": citations,
                "expert_reference": {
                    "name": "Edward Dodson",
                    "email": "info@hgsss.org",
                    "organization": "Henry George School of Social Science"
                },
                "additional_resources": [
                    {
                        "type": "course",
                        "description": "Explore our courses on Georgist economics",
                        "url": "https://www.hgsss.org/courses/"
                    }
                ],
                "follow_up_questions": self._generate_follow_up_questions(user_query, answer, search_results)
            }

            # Add to conversation history
            self.conversation_history.append({
                "query": user_query,
                "response": answer,
                "search_results": search_results,
                "structured_response": structured_response,
                "query_used": query_used,
                "search_time": search_time,
                "llm_time": llm_time
            })

            # Cache the result
            self.query_cache[cache_key] = (answer, search_results, structured_response)

            # Measure total time
            total_time = time.time() - start_time
            print(f"Total query processing time: {total_time:.2f} seconds")

            return answer, search_results, structured_response

        except Exception as e:
            print(f"Error generating response: {str(e)}")
            error_response = "I'm sorry, I encountered an error while processing your question. Please try again."
            return error_response, search_results, {"answer": error_response, "citations": []}

    async def _search_namespace(self, namespace, query, n_results):
        """Search a single namespace asynchronously"""
        try:
            results = self.vector_db.hybrid_search(query, namespace, n_results)
            return results
        except Exception as e:
            print(f"Error searching namespace {namespace}: {str(e)}")
            return []
        
    def _extract_relevant_quotes(self, search_results, query):
        """Extract the most relevant quotes from search results"""
        quotes = []

        # Extract keywords from query
        query_keywords = self.vector_db.extract_keywords_from_text(query)

        for result in search_results:
            text = result.get('text', '')

            # Skip if text is too short
            if len(text) < 50:
                continue

            # Split into sentences - optimized regex
            sentences = re.split(r'(?<=[.!?])\s+', text)

            # Score each sentence based on keyword matches
            scored_sentences = []
            for sentence in sentences:
                # Skip short sentences
                if len(sentence) < 20:
                    continue

                # Count keyword matches
                score = 0
                for keyword in query_keywords:
                    if re.search(r'\b' + re.escape(keyword) + r'\b', sentence.lower()):
                        score += 1

                # Add sentence with score
                if score > 0:
                    scored_sentences.append((sentence, score))

            # Sort by score and take top 1-2 sentences
            scored_sentences.sort(key=lambda x: x[1], reverse=True)
            top_sentences = scored_sentences[:1]  # Reduced from 2

            # Add to quotes
            for sentence, _ in top_sentences:
                book_info = result.get('book_info', {})
                source = f"{book_info.get('title', 'Unknown')}, Chapter {book_info.get('chapter', 'N/A')}"

                quotes.append({
                    'text': sentence,
                    'source': source
                })

        # Limit to top quotes
        return quotes[:3]  # Reduced from 5
    
    def get_available_documents(self):
        # Query the Pinecone index to get unique document names
        # This assumes your vectors have metadata with a 'doc_name' field
        try:
            # Get unique document names from the index
            query_response = self.vector_db.index.query(
                vector=[0] * 1536,  # Dummy vector for metadata-only query
                top_k=10000,  # Large number to get all documents
                include_metadata=True
            )

            # Extract unique document names from metadata
            doc_names = set()
            for match in query_response.matches:
                if hasattr(match, 'metadata') and match.metadata and 'doc_name' in match.metadata:
                    doc_names.add(match.metadata['doc_name'])

            return list(doc_names)
        except Exception as e:
            print(f"Error retrieving documents: {e}")
            return [] 

    def _generate_follow_up_questions(self, query, answer, search_results):
        """Generate follow-up questions based on the query and answer"""
        # Default follow-up questions
        default_questions = [
            "What is the single tax?",
            "How would land value taxation eliminate poverty?",
            "What did you write in Progress and Poverty?",
            "How does land speculation cause economic depressions?",
            "What is your view on free trade?"
        ]

        # Extract topics from search results
        topics = set()
        for result in search_results:
            if "metadata" in result and "keywords_str" in result["metadata"]:
                keywords = result["metadata"]["keywords_str"].split(",")
                topics.update(keywords[:3])  # Take top 3 keywords from each result

        # If we have enough topics, generate custom questions
        if len(topics) >= 3:
            try:
                # Create a prompt for generating follow-up questions
                topics_text = ", ".join(list(topics)[:5])  # Limit to top 5 topics

                prompt = f"""
                Based on the following topics: {topics_text}

                Generate 3 follow-up questions that Henry George might ask to continue the conversation.
                These should be questions that would naturally follow from discussing these topics. 
                Remember the questions should displaying should be from the user's perspective and not from Henry George's perspective.

                Format each question on a new line without numbering or bullet points.
                """

                # Generate questions
                response = self._get_llm_response(prompt, temperature=0.7)

                # Extract questions
                generated_text = response.choices[0].message.content.strip()
                generated_questions = [q.strip() for q in generated_text.split('\n') if q.strip()]

                # Filter out non-questions
                generated_questions = [q for q in generated_questions if q.endswith('?')]

                # If we have enough questions, use them
                if len(generated_questions) >= 3:
                    return generated_questions[:3]
            except Exception as e:
                print(f"Error generating follow-up questions: {str(e)}")

        # Fall back to default questions
        return default_questions[:3]

# Main function to run the console application
async def main():
    print("Welcome to PastPort Bot - Console Edition")
    print("----------------------------------------")

    # Initialize the chatbot
    chatbot = RAGChatbot()

    # Get all available documents from Pinecone
    try:
        # List all namespaces in the Pinecone index
        available_docs = chatbot.vector_db.list_namespaces()
        
        # Format document names for display (remove 'book_' prefix and replace underscores)
        display_names = [
            ns.replace("book_", "").replace("_", " ").title()
            for ns in available_docs
        ]
        
        if available_docs:
            print(f"Loaded documents: {', '.join(display_names)}")
        else:
            print("No documents found. Please upload documents first.")
            return
            
        print("Type 'exit' to quit, 'docs' to see available documents, or 'help' for more commands.")
        print("----------------------------------------")

        # Start chat loop with all documents selected by default
        selected_docs = available_docs

        while True:
            user_input = input("\nYou: ").strip()

            if user_input.lower() == 'exit':
                print("Goodbye!")
                break
            elif user_input.lower() == 'docs':
                print("\nAvailable documents:")
                for i, doc in enumerate(display_names, 1):
                    print(f"{i}. {doc}")
                continue
            elif user_input.lower() == 'help':
                print("\nCommands:")
                print("- 'exit': Quit the application")
                print("- 'docs': List available documents")
                print("- 'select': Change document selection")
                print("- 'help': Show this help message")
                continue
            elif user_input.lower() == 'select':
                print("\nAvailable documents:")
                for i, doc in enumerate(display_names, 1):
                    print(f"{i}. {doc}")
                doc_selection = input("Enter the numbers of the documents to search (comma-separated, or 'all' for all): ")

                if doc_selection.lower() == 'all':
                    selected_docs = available_docs
                else:
                    try:
                        selected_indices = [int(idx.strip()) - 1 for idx in doc_selection.split(',')]
                        selected_docs = [available_docs[idx] for idx in selected_indices if 0 <= idx < len(available_docs)]
                        if not selected_docs:
                            print("No valid documents selected. Using all documents.")
                            selected_docs = available_docs
                        else:
                            print(f"Selected documents: {', '.join([d.replace('book_', '').replace('_', ' ').title() for d in selected_docs])}")
                    except (ValueError, IndexError):
                        print("Invalid selection. Using all documents.")
                        selected_docs = available_docs
                continue

            # Process the user's question with the selected documents
            # After processing the user's question
            if user_input:
                print("\nBot: Thinking...")
                response = await chatbot.query_async(selected_docs, user_input)

                # Print the main answer
                print(f"\nBot: {response[0]}")  # response[0] contains the text response

                # Add readings section using search results if citations are empty
                search_results = response[1]  # This contains the search results

                if search_results:
                    print("\nFor readings I would urge you to read:")

                    # Get unique sources from search results
                    unique_sources = {}
                    for result in search_results:
                        book_info = result.get("book_info", {})
                        book_name = book_info.get("title", "")
                        chapter = book_info.get("chapter", "")
                        chapter_title = book_info.get("chapter_title", "")

                        if book_name and chapter:
                            key = f"{book_name}:{chapter}"
                            if key not in unique_sources:
                                unique_sources[key] = {
                                    "book": book_name,
                                    "chapter": chapter,
                                    "chapter_title": chapter_title
                                }

                    # Format and print sources
                    source_texts = []
                    for source in unique_sources.values():
                        source_text = f"{source['book']}, Chapter {source['chapter']}"
                        if source['chapter_title']:
                            source_text += f": {source['chapter_title']}"
                        source_texts.append(source_text)

                    print(", ".join(source_texts))

                # Add follow-up questions
                if response[2] and "follow_up_questions" in response[2]:
                    follow_up = response[2]["follow_up_questions"]
                    if follow_up:
                        print("\nFollow-up questions you might ask:")
                        for i, question in enumerate(follow_up, 1):
                            print(f"{i}. {question}")
                
    except Exception as e:
        print(f"Error initializing chatbot: {str(e)}")
        print("Please check your Pinecone configuration and API keys.")

if __name__ == "__main__":
    # Run the main function
    import asyncio
    asyncio.run(main())