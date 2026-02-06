from google import genai
from dotenv import load_dotenv
import os
from pypdf import PdfReader
import os
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from google import genai

load_dotenv()

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

def chunk (text , max_length):
    f=""
    for i in range(0, len(text), max_length):
        if i > 0:
            f += text[i-100:i].strip()  # Add 100 characters of overlap for context
        else:
            f += text[i:i+max_length].strip()
    return f

# --- 1. THE GLUE (Custom Embedding Function) ---
# This class teaches ChromaDB how to use Gemini
class GeminiEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        # We send all the text chunks to Google at once
        response = client.models.embed_content(
            model="text-embedding-004",
            contents=input
        )
        # We extract just the math (vectors) from the response
        return [e.values for e in response.embeddings]

#  SETUP DATABASE 
def setup_db(name, chunks):
    # Create an in-memory database (resets when script stops)
    db = chromadb.Client()

    # Create a collection using our custom Gemini function
    collection = db.create_collection(
        name=name,
        embedding_function=GeminiEmbeddingFunction()
    )

    # 3. ADD DATA (Ingestion)
    print("Adding documents...")
    collection.add(
        documents=chunks,
        ids=[str(i) for i in range(len(chunks))]
    )
    return db, collection

def query_db(collection, query):
    #  QUERY (Retrieval)

    print(f"\nQuerying for: '{query}'")

    results = collection.query(
        query_texts=[query],
        n_results=1  # We only want the top 4 matches
    )
    return results
    # # Output the result
    # print("\n--- RESULT ---")
    # print(f"Found Document: {results['documents'][0][0]}")
    # print(f"Distance/Score: {results['distances'][0][0]}") 

def generate_answer(query, results):
    response = client.models.generate_content(
        model="gemini-flash-latest",
        contents=f"Based on the following document, answer the question:\n\nDocument: {results['documents'][0][0]}\n\nQuestion: {query}\n\nAnswer:"
    )
    # print("\n--- GENERATED ANSWER ---")
    # print(response.text)
    with open("output.txt", "w", encoding="utf-8") as f:
        f.write("Based on the following document, answer the question:\n\n")
        f.write(response.text)
    
    print("\n--- GENERATED ANSWER ---")
    print(response.text)

def extract_text_from_pdf(file_path):
    book = PdfReader(file_path)
    text = ""
    for p in book.pages:
        text += p.extract_text()
    return text

def main():

    text=extract_text_from_pdf("MACHINE LEARNING(R17A0534).pdf")
    chunks = chunk(text, 3500)

    name_coll = input("Enter the name of the collection: ")
    db, collection = setup_db(name_coll, chunks)

    q = input("Enter your query: ")
    while q.lower() != "exit":
        results=query_db(collection, q)
        # print("Raw Retrieval Results:", results['documents'][0][0])
        generate_answer(q, results)
        q = input("\nEnter your next query , Enter 'exit' to quit: ")

if __name__ == "__main__":
    main()