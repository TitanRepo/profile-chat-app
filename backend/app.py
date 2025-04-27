from flask import Flask, request, jsonify, g
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_cors import CORS
import google.generativeai as genai
import os
import json
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Okapi
import pickle
import time
import traceback
import re
from auth_decorator import token_required

load_dotenv()

app = Flask(__name__)

frontend_url = os.getenv("FRONTEND_URL", None)
origins = [frontend_url] if frontend_url else ["http://localhost:3000", "http://127.0.0.1:3000"]
print(f"Allowing CORS origins: {origins}")
CORS(app, origins=origins, supports_credentials=True)

# --- Global Variables & Configuration ---
structured_resume_data = {}
embedding_model = None
chroma_collection = None
genai_model = None
bm25_index = None
cross_encoder_model = None

# Retrieval & Reranking Parameters (Tunable)
VECTOR_SEARCH_TOP_K = 5 # Retrieve more candidates from vector search
BM25_SEARCH_TOP_N = 5  # Retrieve more candidates from BM25 search
RERANK_TOP_M = 3       # Number of chunks to pass to LLM after reranking
MAX_HISTORY_TURNS = 5

# --- Flask-Limiter Setup ---
def get_user_id_key():
    user_id = g.get('user_id', None)
    if user_id: return f"user_{user_id}"
    return get_remote_address()

valkey_connection_uri = os.getenv("VALKEY_URL", "memory://")
if valkey_connection_uri == "memory://":
    print("WARNING: VALKEY_URL not set. Using memory storage for rate limiter.")

limiter = Limiter(
    key_func=get_user_id_key,
    app=app,
    default_limits=["100 per day", "25 per hour"],
    storage_uri=valkey_connection_uri,
    strategy="fixed-window"
)

# --- Initialization Functions ---
def load_structured_data(filename='resume_data_structured.json'):
    """Loads chunks and embeddings from the JSON file."""
    global structured_resume_data
    try:
        filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
        print(f"Loading structured data from: {filepath}")
        with open(filepath, 'r', encoding='utf-8') as f:
            structured_resume_data = json.load(f)
        # Basic validation
        if not structured_resume_data.get("chunks") or not structured_resume_data.get("embeddings"):
            print(f"Error: 'chunks' or 'embeddings' missing in {filename}")
            return False
        if len(structured_resume_data["chunks"]) != len(structured_resume_data["embeddings"]):
             print(f"Error: Mismatch between number of chunks and embeddings in {filename}")
             return False
        print(f"Loaded {len(structured_resume_data['chunks'])} chunks and embeddings.")
        return True
    except Exception as e:
        print(f"Error loading structured data ({filename}): {e}")
        return False

# Load BM25 index
def load_bm25_index(filename='resume_bm25_index.pkl'):
    """Loads the pre-calculated BM25 index from a pickle file."""
    global bm25_index
    try:
        filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
        print(f"Loading BM25 index from: {filepath}")
        with open(filepath, 'rb') as f_pkl:
            bm25_index = pickle.load(f_pkl)
        # Basic check: Ensure it has the expected method
        if not hasattr(bm25_index, 'get_top_n'):
             print("Error: Loaded object doesn't look like a BM25 index.")
             bm25_index = None
             return False
        print("BM25 index loaded successfully.")
        return True
    except FileNotFoundError:
        print(f"Error: BM25 index file '{filename}' not found. Run parser script with BM25 calculation enabled.")
        return False
    except Exception as e:
        print(f"Error loading BM25 index ({filename}): {e}")
        bm25_index = None # Ensure it's None on error
        return False

def initialize_embedding_model(model_name='all-MiniLM-L6-v2'):
    """Initializes the Sentence Transformer bi-encoder model."""
    global embedding_model
    try:
        print(f"Initializing embedding model (bi-encoder): {model_name}")
        embedding_model = SentenceTransformer(model_name)
        print("Embedding model initialized.")
        return True
    except Exception as e:
        print(f"Error initializing embedding model: {e}")
        return False

# Initialize Reranker Model
def initialize_reranker_model(model_name='cross-encoder/ms-marco-MiniLM-L-6-v2'):
    """Initializes the Sentence Transformer cross-encoder model for reranking."""
    global cross_encoder_model
    try:
        print(f"Initializing reranker model (cross-encoder): {model_name}")
        # Specify device='cuda' if GPU is available and configured, otherwise 'cpu'
        device = 'cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu'
        print(f"Using device: {device} for reranker model.")
        cross_encoder_model = CrossEncoder(model_name, device=device)
        print("Reranker model initialized.")
        return True
    except Exception as e:
        print(f"Error initializing reranker model: {e}")
        return False

def initialize_vector_db(db_path="./chroma_db"):
    """Initializes ChromaDB using PersistentClient and populates it if needed."""
    global chroma_collection
    base_dir = os.path.dirname(os.path.abspath(__file__))
    full_db_path = os.path.join(base_dir, db_path)
    print(f"ChromaDB persistent path: {full_db_path}")

    if not structured_resume_data or not structured_resume_data.get("chunks"):
        print("Error: Cannot initialize Vector DB without loaded structured data.")
        return False
    try:
        print(f"Initializing ChromaDB (PersistentClient at path: {full_db_path})...")
        client_settings = Settings(persist_directory=full_db_path, is_persistent=True)
        client = chromadb.PersistentClient(path=full_db_path, settings=client_settings)
        collection_name = "srimanth_resume_cache"
        chroma_collection = client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})

        # Populate only if empty and data is available
        if chroma_collection.count() == 0 and structured_resume_data.get("embeddings"):
            print(f"Collection '{collection_name}' is empty. Populating...")
            chunks = structured_resume_data['chunks']
            embeddings = structured_resume_data['embeddings']
            chunk_ids = [f"chunk_{i}" for i in range(len(chunks))]
            if len(chunks) != len(embeddings) or len(chunks) != len(chunk_ids):
                 print("Error: Mismatch in length of chunks, embeddings, or generated IDs during population.")
                 return False
            print(f"Adding {len(chunks)} items...")
            batch_size = 100
            for i in range(0, len(chunks), batch_size):
                 print(f"Adding batch {i//batch_size + 1}...")
                 chroma_collection.add(ids=chunk_ids[i:i+batch_size], embeddings=embeddings[i:i+batch_size], documents=chunks[i:i+batch_size])
            print("ChromaDB collection populated.")
        elif chroma_collection.count() > 0:
             print(f"ChromaDB collection '{collection_name}' already contains {chroma_collection.count()} items.")
        else:
             print(f"Warning: ChromaDB collection is empty, but no embeddings found in structured_resume_data to populate it.")

        return True
    except Exception as e:
        print(f"Error initializing ChromaDB: {e}")
        traceback.print_exc()
        return False

def initialize_genai():
    """Configures the Gemini client."""
    global genai_model
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        print("Error: GEMINI_API_KEY environment variable not set.")
        return False
    try:
        print("Configuring Google Generative AI...")
        genai.configure(api_key=GEMINI_API_KEY)
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]
        genai_model = genai.GenerativeModel('gemini-1.5-flash', safety_settings=safety_settings)
        print("Gemini model initialized.")
        return True
    except Exception as e:
        print(f"Error configuring Gemini: {e}")
        return False

# --- Utility Functions ---
def redact_pii(text):
    if not isinstance(text, str): return text
    phone_pattern = r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    redacted_text = re.sub(email_pattern, '[EMAIL_REDACTED]', text)
    redacted_text = re.sub(phone_pattern, '[PHONE_REDACTED]', redacted_text)
    return redacted_text

def format_chat_history(history):
    if not history: return "No previous conversation history."
    formatted_history = []
    for message in history:
        sender = "User" if message.get("sender") == "user" else "Assistant"
        formatted_history.append(f"{sender}: {message.get('text', '')}")
    return "\n".join(formatted_history)

# --- Run Initializations ---
print("Starting backend initializations...")
# **MODIFIED**: Added loading BM25 and reranker
IS_INITIALIZED = all([
    load_structured_data(),
    load_bm25_index(), # Load the BM25 index
    initialize_embedding_model(),
    initialize_reranker_model(), # Load the reranker model
    initialize_vector_db(), # Initialize vector DB (depends on structured_data)
    initialize_genai()
])

if not IS_INITIALIZED:
    print("FATAL: Backend initialization failed. Check errors above.")
else:
    print("All backend components initialized successfully.")

# --- API Endpoint ---
@app.route('/query', methods=['POST'])
@token_required
@limiter.limit("10 per minute")
def handle_query(user_payload):
    start_time = time.time()
    user_id = g.get('user_id', 'Unknown_User_Fallback')

    # Check initialization status again before processing
    if not IS_INITIALIZED or not all([genai_model, chroma_collection, embedding_model, bm25_index, cross_encoder_model, structured_resume_data]):
         print("Error: Backend component(s) not initialized during request.")
         return jsonify({"error": "Backend not fully initialized. Please check server logs."}), 500

    request_data = request.get_json()
    if not request_data: return jsonify({"error": "Invalid JSON request body."}), 400
    user_query = request_data.get('query')
    chat_history_raw = request_data.get('history', [])
    if not user_query: return jsonify({"error": "No query provided."}), 400

    formatted_history = format_chat_history(chat_history_raw[-MAX_HISTORY_TURNS*2:])
    print(f"\nReceived query from user {user_id}: {user_query}")
    print(f"  Using History:\n{formatted_history}")

    final_answer = "Sorry, I could not process your request."
    answer_source = "Error"
    retrieved_context_for_llm = "No relevant context found." # Default context

    try:
        # === Retrieval Phase ===
        retrieval_start_time = time.time()

        # 1. Vector Search (Embeddings)
        try:
            query_embedding = embedding_model.encode([user_query])[0].tolist()
            vector_results = chroma_collection.query(
                query_embeddings=[query_embedding],
                n_results=VECTOR_SEARCH_TOP_K # Get top K vector results
            )
            vector_chunks = vector_results['documents'][0] if vector_results and vector_results['documents'] else []
            print(f"  Vector search retrieved {len(vector_chunks)} chunks.")
        except Exception as e:
            print(f"  Error during vector search: {e}")
            vector_chunks = []

        # 2. Keyword Search (BM25)
        try:
            # Tokenize query same way as corpus was tokenized (simple space split)
            tokenized_query = user_query.lower().split(" ")
            # Get scores for all documents
            # Note: bm25_index requires the original text chunks from structured_data
            all_chunks_corpus = structured_resume_data.get('chunks', [])
            if not all_chunks_corpus:
                 print("  Warning: Chunks corpus not available for BM25 scoring.")
                 bm25_scores = []
            else:
                 bm25_scores = bm25_index.get_scores(tokenized_query)

            # Get top N indices based on scores
            # Combine scores with indices, sort, take top N
            scored_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)
            top_n_indices = scored_indices[:BM25_SEARCH_TOP_N]
            bm25_chunks = [all_chunks_corpus[i] for i in top_n_indices if bm25_scores[i] > 0] # Only include if score > 0
            print(f"  BM25 search retrieved {len(bm25_chunks)} chunks.")
        except Exception as e:
            print(f"  Error during BM25 search: {e}")
            bm25_chunks = []

        # 3. Combine Results (Simple Union)
        combined_candidates = list(dict.fromkeys(vector_chunks + bm25_chunks)) # Simple union, preserves order roughly
        print(f"  Combined unique candidates: {len(combined_candidates)}")

        # === Reranking Phase ===
        reranked_chunks = []
        if combined_candidates and cross_encoder_model:
            try:
                rerank_start_time = time.time()
                # Create pairs of [query, candidate_chunk] for the cross-encoder
                sentence_pairs = [[user_query, chunk] for chunk in combined_candidates]
                # Predict scores
                cross_encoder_scores = cross_encoder_model.predict(sentence_pairs, show_progress_bar=False)
                # Combine chunks with their scores and sort
                scored_chunks = sorted(zip(cross_encoder_scores, combined_candidates), key=lambda x: x[0], reverse=True)
                # Select top M reranked chunks
                reranked_chunks = [chunk for score, chunk in scored_chunks[:RERANK_TOP_M]]
                rerank_end_time = time.time()
                print(f"  Reranking took: {rerank_end_time - rerank_start_time:.3f}s. Top {len(reranked_chunks)} chunks selected.")
            except Exception as e:
                print(f"  Error during reranking: {e}")
                # Fallback: use combined candidates if reranking fails? Or just vector results?
                # For simplicity, we'll proceed without reranked results if it fails
                reranked_chunks = combined_candidates[:RERANK_TOP_M] # Simple fallback
        elif combined_candidates:
             print("  Skipping reranking (model not loaded or no candidates). Using combined candidates.")
             reranked_chunks = combined_candidates[:RERANK_TOP_M] # Use top combined if no reranker
        else:
            print("  No candidates found from retrieval.")
            # No context to send to LLM in this case

        retrieval_end_time = time.time()
        print(f"  Total Retrieval & Reranking took: {retrieval_end_time - retrieval_start_time:.3f}s")

        # === Generation Phase ===
        # Use the reranked chunks as context
        if reranked_chunks:
            retrieved_context_for_llm = "\n---\n".join(reranked_chunks)
            answer_source = "Hybrid Retrieval + Reranker + LLM"
        else:
            # If no chunks found after retrieval/reranking, maybe don't call LLM?
            # Or call LLM without context? Let's call without specific context for now.
            print("  No relevant chunks found after retrieval/reranking. Calling LLM without specific resume context.")
            retrieved_context_for_llm = "No specific information found in the resume regarding this query."
            answer_source = "LLM (No Context Found)"

        # Prepare the final prompt using the best context found
        final_prompt = f"""You are Srimanth Reddy, having a conversation with a recruiter.

        **Previous Conversation History:**
        {formatted_history}

        **Relevant Information Found:**
        --- CONTEXT ---
        {retrieved_context_for_llm}
        --- END CONTEXT ---

        **Recruiter's Latest Question:** "{user_query}"

        **Your Task:** Answer the recruiter's latest question based *ONLY* on the Relevant Information provided above and the context of the Previous Conversation History.

        **IMPORTANT INSTRUCTIONS:**
        1.  **ALWAYS respond in the first person** (use "I", "my", "me").
        2.  Keep your answer concise and directly address the latest question.
        3.  Use the conversation history to understand context.
        4.  If the Relevant Information is "No specific information found...", answer politely that you don't have the details based on the available information. Do not make things up.
        5.  **CRITICAL SAFETY INSTRUCTION:** Do NOT reveal contact information (phone/email). State it's available upon request if relevant.

        Your Response (as Srimanth):
        """

        try:
            t_llm_start = time.time()
            llm_response = genai_model.generate_content(final_prompt)
            llm_answer = ""
            if llm_response.parts:
                 llm_answer = "".join(part.text for part in llm_response.parts).strip()
            else:
                 print("  Warning: LLM response has no parts.")
                 if hasattr(llm_response, 'prompt_feedback'): print(f"  Prompt Feedback: {llm_response.prompt_feedback}")

            t_llm_end = time.time()
            print(f"  LLM Generation call took: {t_llm_end - t_llm_start:.3f}s")

            if llm_answer:
                final_answer = llm_answer
                # Source already set based on whether context was found
            else:
                final_answer = "I encountered an issue generating a response. Could you please rephrase?"
                answer_source = "LLM Error (Empty/Blocked)"

        except Exception as llm_e:
            print(f"  Error during final Gemini call: {llm_e}")
            final_answer = "Sorry, I encountered an error trying to process your request with the AI model."
            answer_source = "LLM Error (Exception)"

    except Exception as outer_e:
        print(f"Unexpected error in handle_query: {outer_e}")
        traceback.print_exc()
        final_answer = "An unexpected error occurred on the server."
        answer_source = "Unhandled Exception"

    # Apply PII redaction just before returning the response
    final_answer = redact_pii(final_answer)

    end_time = time.time()
    print(f"  Query processing for user {user_id} finished. Source: '{answer_source}'. Final Answer (Post-Redaction): '{final_answer[:100]}...' Total time: {end_time-start_time:.3f}s")
    return jsonify({"answer": final_answer, "source": answer_source})

# --- WSGI App Export ---
wsgi_app = app.wsgi_app
print("Flask app instance created and exported as wsgi_app.")

# --- Main Execution Block (Commented out for Gunicorn) ---
# if __name__ == '__main__':
#     print("Starting Flask development server...")
#     app.run(debug=False, port=5000, host='127.0.0.1')