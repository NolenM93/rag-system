"""
RAG_app.py - Retrieval-Augmented Generation system with re-ranking
"""

import logging
import warnings
from dotenv import load_dotenv
import os
import openai
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
import faiss


# ============================================================================
# Step 3.1: Suppress Noisy Logs
# ============================================================================
logging.getLogger('langchain.text_splitter').setLevel(logging.ERROR)
try:
    import transformers
    transformers.logging.set_verbosity_error()
except:
    pass
warnings.filterwarnings('ignore')


# ============================================================================
# Step 3.2: ChatGPT API Credentials
# ============================================================================
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')


# ============================================================================
# Step 3.3: Parameters
# ============================================================================
chunk_size = 250
chunk_overlap = 25
model_name = "sentence-transformers/all-distilroberta-v1"
top_k = 20

# Re-ranking parameters
cross_encoder_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
top_m = 8


# ============================================================================
# Step 3.4: Read the Pre-scraped Document
# ============================================================================
with open('Selected_Document.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print(f"Loaded document with {len(text)} characters")


# ============================================================================
# Step 3.5: Split into Appropriately-Sized Chunks
# ============================================================================
text_splitter = RecursiveCharacterTextSplitter(
    separators=['\n\n', '\n', ' ', ''],
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap
)

chunks = text_splitter.split_text(text)
print(f"Split document into {len(chunks)} chunks")


# ============================================================================
# Step 3.6: Embed & Build FAISS Index
# ============================================================================
print(f"Loading embedding model: {model_name}")
embedder = SentenceTransformer(model_name)

print("Generating embeddings for all chunks...")
embeddings = embedder.encode(chunks, show_progress_bar=False)

# Convert to NumPy float32 array
embeddings_array = np.array(embeddings, dtype=np.float32)

# Initialize FAISS index
dimension = embeddings_array.shape[1]
faiss_index = faiss.IndexFlatL2(dimension)

# Add embeddings to the index
faiss_index.add(embeddings_array)
print(f"FAISS index created with {faiss_index.ntotal} vectors of dimension {dimension}")


# ============================================================================
# Step 3.7: Retrieval Function
# ============================================================================
def retrieve_chunks(question, k=top_k):
    """
    Retrieve the top k most relevant chunks for a given question.
    
    Args:
        question (str): The user's question
        k (int): Number of chunks to retrieve
        
    Returns:
        list[str]: The k most relevant text chunks
    """
    # Encode the question
    q_vec = embedder.encode([question], show_progress_bar=False)
    
    # Convert to NumPy float32 array
    q_arr = np.array(q_vec, dtype=np.float32)
    
    # Search FAISS index for top k nearest neighbors
    distances, I = faiss_index.search(q_arr, k)
    
    # Return the corresponding chunks
    retrieved = [chunks[i] for i in I[0]]
    return retrieved


# ============================================================================
# Step 3.8: Implement a Cross-Encoder Re-Ranker
# ============================================================================
print(f"Loading cross-encoder model: {cross_encoder_name}")
reranker = CrossEncoder(cross_encoder_name)


def dedupe_preserve_order(items):
    """
    Remove duplicates from a list while preserving first occurrence order.
    Normalizes whitespace to avoid near-duplicate slices.
    
    Args:
        items (list[str]): List of text items
        
    Returns:
        list[str]: Deduplicated list
    """
    seen = set()
    result = []
    
    for item in items:
        # Normalize whitespace for comparison
        normalized = ' '.join(item.split())
        if normalized not in seen:
            seen.add(normalized)
            result.append(item)
    
    return result


def rerank_chunks(question, candidate_chunks, m=top_m):
    """
    Re-rank candidate chunks using a cross-encoder and return the top m.
    
    Args:
        question (str): The user's question
        candidate_chunks (list[str]): List of candidate chunks to re-rank
        m (int): Number of top chunks to return after re-ranking
        
    Returns:
        list[str]: The m most relevant chunks after re-ranking
    """
    # Create (question, chunk) pairs
    pairs = [(question, chunk) for chunk in candidate_chunks]
    
    # Score with cross-encoder (higher score = more relevant)
    scores = reranker.predict(pairs)
    
    # Sort by score descending and select top m
    scored_chunks = list(zip(scores, candidate_chunks))
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    
    # Extract top m chunks
    top_chunks = [chunk for score, chunk in scored_chunks[:m]]
    
    # Light deduplication
    top_chunks = dedupe_preserve_order(top_chunks)
    
    return top_chunks


# ============================================================================
# Step 3.9: Q&A with ChatGPT
# ============================================================================
def answer_question(question):
    """
    Answer a question using retrieval, re-ranking, and ChatGPT.
    
    Args:
        question (str): The user's question
        
    Returns:
        str: The generated answer
    """
    # Step 1: Retrieve top_k candidate chunks
    candidates = retrieve_chunks(question, k=top_k)
    
    # Step 2: Re-rank to get top_m most relevant chunks
    relevant_chunks = rerank_chunks(question, candidates, m=top_m)
    
    # Step 3: Join chunks into context
    context = '\n\n'.join(relevant_chunks)
    
    # Step 4: Define prompts
    system_prompt = (
        "You are a knowledgeable assistant that answers questions based on the provided context. "
        "If the answer is not in the context, say you don't know."
    )
    
    user_prompt = f"""Context:
{context}

Question: {question}

Answer:"""
    
    # Step 5: Call OpenAI ChatGPT API
    try:
        from openai import OpenAI
        client = OpenAI(api_key=openai.api_key)
        
        resp = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=500
        )
        
        answer = resp.choices[0].message.content.strip()
        return answer
        
    except Exception as e:
        return f"Error generating answer: {e}"


# ============================================================================
# Step 3.10: Interactive Loop
# ============================================================================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("RAG System Ready!")
    print("="*70)
    print("Enter 'exit' or 'quit' to end.\n")
    
    while True:
        question = input("Your question: ")
        
        if question.lower() in ("exit", "quit"):
            print("Goodbye!")
            break
            
        if not question.strip():
            continue
            
        print("\nAnswer:", answer_question(question))
        print()
