# from google import genai
# from dotenv import load_dotenv
# import os
# from pypdf import PdfReader
# import os
# import chromadb
# from chromadb import Documents, EmbeddingFunction, Embeddings
# from google import genai

# load_dotenv()

# client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# def chunk (text , max_length):
#     f=[]
#     for i in range(0, len(text), max_length):
#         if i > 0:
#             f.append(text[i-50:i].strip())  # Add 50 characters of overlap for context
#         else:
#             f.append(text[i:i+max_length].strip())
#     return f


# # This class teaches ChromaDB how to use Gemini
# class GeminiEmbeddingFunction(EmbeddingFunction):
#     def __init__(self):
#         pass
    
#     def __call__(self, input: Documents) -> Embeddings:
#         # We send all the text chunks to Google at once
#         response = client.models.embed_content(
#             model="models/gemini-embedding-001",
#             contents=input
#         )
#         # We extract just the vectors from the response
#         return [e.values for e in response.embeddings]

# #  SETUP DATABASE 
# def setup_db(name, chunks):
#     # Create an in-memory database (resets when script stops)
#     db = chromadb.Client()

#     # Create a collection using our custom Gemini function
#     collection = db.create_collection(
#         name=name,
#         embedding_function=GeminiEmbeddingFunction()
#     )

#     # 3. Add data in a collection
#     print("Adding documents...")
#     collection.add(
#         documents=chunks,
#         ids=[str(i) for i in range(len(chunks))]
#     )
#     return db, collection

# def query_db(collection, query):

#     print(f"\nQuerying for: '{query}'")

#     results = collection.query(
#         query_texts=[query],
#         n_results=3  # We only want the top 3 matches
#     )
#     return results
#     # # Output the result
#     # print("\n--- RESULT ---")
#     # print(f"Found Document: {results['documents'][0][0]}")
#     # print(f"Distance/Score: {results['distances'][0][0]}") 

# def generate_answer(query, results):
#     response = client.models.generate_content(
#         model="gemini-flash-latest",

#         #Passing the top 3 retrieved documents as context as 1 document is not sufficient
#         contents=f"Based on the following document, answer the question:\n\nDocument: {results['documents'][0][0] , results['documents'][0][1] , results['documents'][0][2]}\n\nQuestion: {query}\n\nAnswer:"
#     )
#     # print("\n--- GENERATED ANSWER ---")
#     # print(response.text)
#     with open("output.txt", "a+", encoding="utf-8") as f:
#         f.write("Question:\n")
#         f.write(query)
#         f.write("\n\nAnswer:\n")
#         f.write(response.text)
#         f.write("\n\n" + "="*80 + "\n\n")
    
#     print("\n--- GENERATED ANSWER ---")
#     print(response.text)

# def extract_text_from_pdf(file_path):
#     book = PdfReader(file_path)
#     text = ""
#     for p in book.pages:
#         text += p.extract_text()
#     return text

# def main():

#     text=extract_text_from_pdf("MACHINE LEARNING(R17A0534).pdf")
#     chunks = chunk(text, 3000)

#     name_coll = input("Enter the name of the collection: ")
#     db, collection = setup_db(name_coll, chunks)

#     q = input("Enter your query: ")
#     while q.lower() != "exit":
#         results=query_db(collection, q)
#         # print("Raw Retrieval Results:", results['documents'][0][0])
#         generate_answer(q, results)
#         q = input("\nEnter your next query , Enter 'exit' to quit: ")

# if __name__ == "__main__":
#     main()













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
            model="gemini-flash-latest", 
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