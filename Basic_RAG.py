import os
import time
import logging
from dotenv import load_dotenv
from pypdf import PdfReader
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from google import genai
from google.genai import types

load_dotenv()
API_KEY = os.environ.get("GEMINI_API_KEY")

# Safe Settings for Free Tier
BATCH_SIZE = 20        # Number of chunks to send at once
SLEEP_PER_BATCH = 2    # Seconds to wait between batches (avoids 429 errors)
DB_PATH = "./my_vector_store" # Where to save the database on disk

# Setup Logging (Cleaner than print statements)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs.txt'),
        logging.StreamHandler()
    ]
)

if not API_KEY:
    msg = "API Key not found! Please check your .env file."
    logging.error(msg)
    raise ValueError(msg)

client = genai.Client(api_key=API_KEY)

# BETTER CHUNKING 
def chunk_text(text, max_length=1000, overlap=100):
    """
    Splits text into chunks with overlap.
    Corrected logic to ensure no data is lost.
    """
    if not text:
        return []
        
    chunks = []
    # We step forward by (max_length - overlap) to create the sliding window
    step = max_length - overlap
    
    for i in range(0, len(text), step):
        chunk = text[i:i + max_length]
        # Only add valid chunks
        if len(chunk) > 50: 
            chunks.append(chunk.strip())
            
    return chunks

# EMBEDDING FUNCTION 
class GeminiEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        try:
            response = client.models.embed_content(
                model="models/gemini-embedding-001", 
                contents=input
            )
            return [e.values for e in response.embeddings]
        except Exception as e:
            logging.error(f" Embedding API Error: {e}")
            # Return empty embeddings to avoid crashing the whole DB
            return [[] for _ in input] 

# DATABASE MANAGEMENT
def get_or_create_db(collection_name):
    # Use PersistentClient so data is saved to disk
    db = chromadb.PersistentClient(path=DB_PATH)

    collection = db.get_or_create_collection(
        name=collection_name,
        embedding_function=GeminiEmbeddingFunction()
    )
    return collection

def ingest_data(collection, chunks):
    # Check if data already exists to save quota!
    if collection.count() > 0:
        logging.info(" Data already exists in DB. Skipping ingestion to save quota.")
        return

    logging.info(f" Starting ingestion of {len(chunks)} chunks...")
    
    # Batch processing loop
    total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        ids = [f"id_{i+j}" for j in range(len(batch))]
        
        try:
            collection.add(documents=batch, ids=ids)
            logging.info(f"   Indexed batch {i//BATCH_SIZE + 1}/{total_batches}")
            
            # CRITICAL: Sleep to respect rate limits
            time.sleep(SLEEP_PER_BATCH)
            
        except Exception as e:
            logging.error(f" Failed to ingest batch {i}: {e}")

    logging.info(" Ingestion Complete!")

# RETRIEVAL & GENERATION
def query_and_answer(collection, query):
    if not query.strip():
        return
        
    logging.info(f" Searching for: '{query}'")
    
    # 1. Retrieve
    results = collection.query(query_texts=[query], n_results=3)
    
    if not results['documents'] or not results['documents'][0]:
        msg = "No relevant information found in the document."
        print(msg)
        logging.info(msg)
        return

    # Combine top 3 chunks into one context block (Used for better context and rich answers)
    context_text = "\n\n---\n\n".join(results['documents'][0])
    
    # Save raw retrieved context for manual checking
    with open("raw_responses.txt", "a+", encoding="utf-8") as f:
        f.write(f"Query: {query}\n")
        f.write(f"Retrieved Context:\n{context_text}\n")
        f.write("=" * 80 + "\n\n")
    
    # 2. Generate
    prompt = f"""
    You are a helpful teaching assistant. Answer the student's question based ONLY on the provided context.
    If the answer is not in the context, say "I cannot find the answer in the document."
    
    Context:
    {context_text}
    
    Question: {query}
    """
    
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite", 
            contents=prompt
        )
        
        answer = response.text
        output = "\n" + "="*40 + f"\n GEMINI ANSWER:\n{answer}\n" + "="*40 + "\n"
        print(output)
        logging.info(f"GEMINI ANSWER: {answer}")
        
        # Save to file
        with open("chat_history.txt", "a+", encoding="utf-8") as f:
            f.write(f"Q: {query}\n")
            f.write(f"A: {answer}\n")
            f.write("-" * 40 + "\n")
            
    except Exception as e:
        logging.error(f" Generation Error: {e}")

def main():
    pdf_path = "MACHINE LEARNING(R17A0534).pdf"
    
    # Edge Case: File not found
    if not os.path.exists(pdf_path):
        msg = f"Error: File '{pdf_path}' not found."
        print(msg)
        logging.error(msg)
        return

    # 1. Read PDF
    try:
        reader = PdfReader(pdf_path)
        full_text = ""
        for page in reader.pages:
            full_text += page.extract_text() or ""
        
        logging.info(f"Read {len(full_text)} characters from PDF.")
    except Exception as e:
        msg = f"Error reading PDF: {e}"
        print(msg)
        logging.error(msg)
        return

    # 2. Chunk
    chunks = chunk_text(full_text, max_length=1000, overlap=200)

    # 3. Setup & Ingest 
    collection_name = "ml_textbook_v1" # Hardcode or ask user, but keep it consistent
    collection = get_or_create_db(collection_name)
    ingest_data(collection, chunks)

    # 4. Chat Loop
    while True:
        q = input("Enter query (or 'exit'): ")
        logging.info(f"User query: {q}")
        if q.lower() in ["exit", "quit"]:
            logging.info("User exited the chat.")
            break
        query_and_answer(collection, q)

if __name__ == "__main__":
    main()