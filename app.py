import streamlit as st
from sentence_transformers import CrossEncoder, SentenceTransformer
import numpy as np
import pickle
import faiss

# Load BM25 index
with open("bm25_index.pkl", "rb") as f:
    bm25 = pickle.load(f)

# Load FAISS index
faiss_index = faiss.read_index("Cisco_FinanicalData_index.faiss")

# Load embeddings
balance_sheet_embeddings = np.load("Cisco_FinanicalData_embeddings.npy")

# Load Cross-Encoder for re-ranking
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

FINANCE_KEYWORDS = ['assets', 'total', 'liabilities', 'net', 'current', 'deferred', 'equity', 'payable', 'october', 'january', 'april', 'july', 'cash', 'accounts', 'financing', 'receivables', 'debt', 'income', 'taxes', 'revenue', 'historical', 'financials', 'balance', 'sheets', 'millions', 'equivalents', 'investments', 'receivable', 'inventories', 'property', 'equipment', 'goodwill', 'purchased', 'intangible', 'tax', 'accrued', 'compensation', 'cisco', 'systems', 'inc']

# Function to validate user query
def is_valid_finance_query(query):
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in FINANCE_KEYWORDS)

# Load financial data
with open("Cisco_FinanicalData_chunks.txt", "r") as f:
    balance_sheet_chunks = f.readlines()

def hybrid_retrieve(query, top_k=5, bm25_weight=0.4, faiss_weight=0.6):
    """Retrieve relevant results using BM25 and FAISS, then merge rankings."""

    # Encode query for FAISS
    query_embedding = embedding_model.encode([query])
    
    # Retrieve from FAISS
    D, I = faiss_index.search(query_embedding, top_k)
    faiss_results = [(balance_sheet_chunks[i], 1 - D[0][idx]) for idx, i in enumerate(I[0])]
    
    # Retrieve from BM25
    query_tokens = query.lower().split()
    bm25_scores = bm25.get_scores(query_tokens)
    bm25_top_idxs = np.argsort(bm25_scores)[-top_k:]
    bm25_results = [(balance_sheet_chunks[i], bm25_scores[i]) for i in bm25_top_idxs]
    
    # Merge FAISS & BM25 results with weighted scoring
    merged_results = {}
    for text, score in faiss_results:
        merged_results[text] = merged_results.get(text, 0) + score * faiss_weight
    for text, score in bm25_results:
        merged_results[text] = merged_results.get(text, 0) + score * bm25_weight
    
    # Sort final results by weighted score
    ranked_results = sorted(merged_results.items(), key=lambda x: x[1], reverse=True)
    
    return ranked_results[:top_k]

def rerank_results(query, retrieved_texts):
    """Re-rank retrieved results using a Cross-Encoder model."""
    query_pairs = [[query, text] for text in retrieved_texts]  # Create query-text pairs
    scores = cross_encoder.predict(query_pairs)  # Compute similarity scores
    ranked_results = sorted(zip(retrieved_texts, scores), key=lambda x: x[1], reverse=True)
    return ranked_results

# Streamlit UI
st.set_page_config(page_title="Financial RAG Chatbot", layout="wide")
st.title("💬 Financial RAG Chatbot")
st.write("Ask any financial question about Cisco's Finanical Statement like from balance sheets.")

# User query input
query = st.text_input("Enter your financial query:")

if query:
    if not is_valid_finance_query(query):
        st.error("Invalid Query: Please ask finance-related questions only.")
    else:
    # Retrieve initial results
        retrieved_data = hybrid_retrieve(query)
        retrieved_texts = [res[0] for res in retrieved_data]
        
        # Re-rank results
        reranked_data = rerank_results(query, retrieved_texts)
        
        # Normalize scores to 0-100% scale
        min_score = min([score for _, score in reranked_data])
        max_score = max([score for _, score in reranked_data])

        def normalize_score(score):
            return int(((score - min_score) / (max_score - min_score)) * 100) if max_score != min_score else 50

        # Display results
        st.subheader("📊 Retrieved Financial Data:")
        for idx, (text, score) in enumerate(reranked_data):
            normalized_score = normalize_score(score)
            st.markdown(f"**{idx+1}.** {text} \n**Score:** {normalized_score}%")
            st.progress(normalized_score / 100)
