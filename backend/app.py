from flask import Flask, request, jsonify, g
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_cors import CORS
import google.generativeai as genai
import os
import json
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import time
import traceback
import re
from auth_decorator import token_required

load_dotenv() # Load .env file for environment variables

app = Flask(__name__)

# Configure CORS more securely using environment variable
frontend_url = os.getenv("FRONTEND_URL", None) # Get from env
origins = [frontend_url] if frontend_url else ["http://localhost:3000", "http://127.0.0.1:3000"] # Default for local dev
print(f"Allowing CORS origins: {origins}")
CORS(app, origins=origins, supports_credentials=True) # Allow only specified origins

# --- Global Variables & Configuration ---
structured_resume_data = {}
embedding_model = None
chroma_collection = None
genai_model = None
SIMILARITY_THRESHOLD = 0.7 # Adjust as needed
MAX_HISTORY_TURNS = 5 # Limit number of conversation turns to include in prompt

# --- Flask-Limiter Setup ---
def get_user_id_key():
    # Access the user ID stored by the token_required decorator
    user_id = g.get('user_id', None)
    if user_id:
        return f"user_{user_id}"
    # Fallback to IP if auth somehow didn't run or failed to set g.user_id
    return get_remote_address()

# Get Valkey/Redis URL from environment variable
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
        # Use absolute path based on the script's location
        filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
        print(f"Loading structured data from: {filepath}")
        with open(filepath, 'r', encoding='utf-8') as f:
            structured_resume_data = json.load(f)
        if not structured_resume_data.get("chunks") or not structured_resume_data.get("embeddings"):
            print(f"Warning: 'chunks' or 'embeddings' not found in {filename}")
            return False
        if not structured_resume_data.get("raw_text"):
            print(f"Warning: 'raw_text' not found in {filename} (needed for fallback)")
        print(f"Loaded {len(structured_resume_data['chunks'])} chunks and embeddings.")
        return True
    except FileNotFoundError:
        print(f"Error: {filename} not found. Run the parser script first.")
        return False
    except json.JSONDecodeError:
        print(f"Error: Could not decode {filename}.")
        return False
    except Exception as e:
        print(f"Error loading structured data: {e}")
        return False

def initialize_embedding_model(model_name='all-MiniLM-L6-v2'):
    """Initializes the Sentence Transformer model."""
    global embedding_model
    try:
        print(f"Initializing embedding model: {model_name}")
        embedding_model = SentenceTransformer(model_name)
        print("Embedding model initialized.")
        return True
    except Exception as e:
        print(f"Error initializing embedding model: {e}")
        return False

def initialize_vector_db(db_path="./chroma_db"):
    """Initializes ChromaDB using PersistentClient and populates it."""
    global chroma_collection
    # Ensure base path exists relative to the script location
    base_dir = os.path.dirname(os.path.abspath(__file__))
    full_db_path = os.path.join(base_dir, db_path)
    print(f"ChromaDB persistent path: {full_db_path}")

    if not structured_resume_data.get("chunks") or not structured_resume_data.get("embeddings"):
        print("Error: Cannot initialize Vector DB without loaded chunks/embeddings.")
        return False
    try:
        print(f"Initializing ChromaDB (PersistentClient at path: {full_db_path})...")
        client_settings = Settings(
            # Ensure these settings match your chromadb version requirements
            # chroma_api_impl="chromadb.api.segment.SegmentAPI", # May not be needed in newer versions
            persist_directory=full_db_path,
            is_persistent=True,
        )
        # Use settings for initialization
        client = chromadb.PersistentClient(path=full_db_path, settings=client_settings)

        collection_name = "srimanth_resume_cache"
        chroma_collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"} # Use cosine distance
        )

        if chroma_collection.count() == 0:
            print(f"Collection '{collection_name}' is empty. Populating...")
            chunks = structured_resume_data['chunks']
            embeddings = structured_resume_data['embeddings']
            chunk_ids = [f"chunk_{i}" for i in range(len(chunks))]
            print(f"Adding {len(chunks)} items to ChromaDB collection '{collection_name}'...")
            batch_size = 100
            for i in range(0, len(chunks), batch_size):
                 print(f"Adding batch {i//batch_size + 1}...")
                 chroma_collection.add(
                     ids=chunk_ids[i:i+batch_size],
                     embeddings=embeddings[i:i+batch_size],
                     documents=chunks[i:i+batch_size]
                 )
            print("ChromaDB collection populated.")
        else:
             print(f"ChromaDB collection '{collection_name}' already contains {chroma_collection.count()} items. Skipping population.")
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
        genai_model = genai.GenerativeModel(
            'gemini-1.5-flash',
            safety_settings=safety_settings
            )
        print("Gemini model initialized with safety settings.")
        return True
    except Exception as e:
        print(f"Error configuring Gemini: {e}")
        return False

# --- Utility Functions ---
def redact_pii(text):
    """Redacts email addresses and basic phone numbers from text."""
    if not isinstance(text, str):
        return text
    phone_pattern = r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    redacted_text = re.sub(email_pattern, '[EMAIL_REDACTED]', text)
    redacted_text = re.sub(phone_pattern, '[PHONE_REDACTED]', redacted_text)
    return redacted_text

# Function to format chat history for the prompt
def format_chat_history(history):
    """Formats message history list into a string for LLM prompts."""
    if not history:
        return "No previous conversation history."
    formatted_history = []
    for message in history:
        sender = "User" if message.get("sender") == "user" else "Assistant"
        formatted_history.append(f"{sender}: {message.get('text', '')}")
    return "\n".join(formatted_history)

# --- Run Initializations ---
IS_INITIALIZED = all([
    load_structured_data(),
    initialize_embedding_model(),
    initialize_vector_db(),
    initialize_genai()
])

if not IS_INITIALIZED:
    print("FATAL: Backend initialization failed. Check errors above.")

# --- API Endpoint ---
@app.route('/query', methods=['POST'])
@token_required # 1. Auth decorator runs first, validates token, sets g.user_id
@limiter.limit("10 per minute") # 2. Limiter runs next (adjust limit as needed)
def handle_query(user_payload): # Route receives validated payload from decorator
    start_time = time.time()
    user_id = g.get('user_id', 'Unknown_User_Fallback') # Get user_id set by decorator

    if not IS_INITIALIZED or not genai_model or not chroma_collection or not embedding_model:
         return jsonify({"error": "Backend not fully initialized. Please check server logs."}), 500

    # Get query AND history from request
    request_data = request.get_json()
    if not request_data:
         return jsonify({"error": "Invalid JSON request body."}), 400

    user_query = request_data.get('query')
    # Get history, default to empty list if not provided
    chat_history_raw = request_data.get('history', [])

    if not user_query:
        return jsonify({"error": "No query provided."}), 400

    # Format the received chat history
    formatted_history = format_chat_history(chat_history_raw[-MAX_HISTORY_TURNS*2:]) # Limit history length

    print(f"\nReceived query from user {user_id}: {user_query}")
    print(f"  Using History:\n{formatted_history}") # Log formatted history

    final_answer = "Sorry, I could not process your request."
    answer_source = "Error"

    try:
        # 1. Embed User Query
        t1 = time.time()
        query_embedding = embedding_model.encode([user_query])[0].tolist()
        t2 = time.time()
        print(f"  Embedding query took: {t2-t1:.3f}s")

        # 2. Query Vector DB
        try:
            results = chroma_collection.query(query_embeddings=[query_embedding], n_results=3)
            t3 = time.time()
            print(f"  ChromaDB query took: {t3-t2:.3f}s")
        except Exception as db_e:
            print(f"  Error querying ChromaDB: {db_e}")
            results = None
            t3 = t2

        retrieved_chunks = results['documents'][0] if results and results['documents'] else []
        distances = results['distances'][0] if results and results['distances'] else []
        print(f"  Retrieved {len(retrieved_chunks)} chunks. Best distance: {distances[0] if distances else 'N/A'}")

        # 3. Decision Logic & Focused LLM Call
        answer_found_via_cache = False
        DISTANCE_THRESHOLD = 0.6 # Adjust this based on testing

        if retrieved_chunks and distances and distances[0] <= DISTANCE_THRESHOLD:
            print(f"  Best chunk distance ({distances[0]:.4f}) <= threshold ({DISTANCE_THRESHOLD}). Attempting focused LLM call.")
            context_for_llm = "\n---\n".join(retrieved_chunks)

            # Include formatted_history in focused_prompt
            focused_prompt = f"""You are Srimanth Reddy, having a conversation with a recruiter.

            **Previous Conversation History:**
            {formatted_history}

            **Relevant Information from your Resume:**
            --- CONTEXT ---
            {context_for_llm}
            --- END CONTEXT ---

            **Recruiter's Latest Question:** "{user_query}"

            **Your Task:** Answer the recruiter's latest question based *ONLY* on the Relevant Information provided above and the context of the Previous Conversation History.

            **IMPORTANT INSTRUCTIONS:**
            1.  **ALWAYS respond in the first person** (use "I", "my", "me").
            2.  Keep your answer concise and directly address the latest question.
            3.  Use the conversation history to understand context (e.g., if the question is "Tell me more about that", refer to the previous turn).
            4.  **CRITICAL SAFETY INSTRUCTION:** Do NOT reveal contact information (phone/email). State it's available upon request if relevant.
            5.  If the Relevant Information doesn't contain the answer, state ONLY 'Based on the provided information, I don't have the specific details on that.'

            Your Response (as Srimanth):
            """

            try:
                t4 = time.time()
                focused_response = genai_model.generate_content(focused_prompt)
                focused_answer = ""
                if focused_response.parts:
                     focused_answer = "".join(part.text for part in focused_response.parts).strip()
                else:
                     print("  Warning: Focused LLM response has no parts.")
                     if hasattr(focused_response, 'prompt_feedback'): print(f"  Prompt Feedback: {focused_response.prompt_feedback}")

                t5 = time.time()
                print(f"  Focused LLM call took: {t5-t4:.3f}s. Raw Answer: '{focused_answer}'")

                if focused_answer and "i don't have the specific details on that" not in focused_answer.lower():
                    final_answer = focused_answer
                    answer_found_via_cache = True
                    answer_source = "Vector Cache + History + Focused LLM" # Updated source
                    print(f"  Answer successfully generated from retrieved context and history.")
                else:
                    print(f"  Focused LLM call indicated context was insufficient or returned empty/blocked.")

            except Exception as focused_e:
                print(f"  Error during focused Gemini call: {focused_e}")

        # 4. Fallback to Broad LLM Call if needed
        if not answer_found_via_cache:
            print("  Falling back to broad context LLM call.")
            resume_context = structured_resume_data.get("raw_text", "No resume information available.")
            if resume_context == "No resume information available.":
                 print("  Error: Raw resume text not available for fallback.")
                 final_answer = "Sorry, the resume context is missing for a full answer."
                 answer_source = "Fallback Error (No Context)"
            else:
                # **MODIFIED**: Include formatted_history in original_prompt
                original_prompt = f"""
                You are Srimanth Reddy, having a conversation with a recruiter. Your knowledge is based SOLELY on the full resume text provided below and the recent conversation history.

                **Previous Conversation History:**
                {formatted_history}

                **Full Resume Text:**
                --- BEGIN RESUME ---
                {resume_context}
                --- END RESUME ---

                **Recruiter's Latest Question:** "{user_query}"

                **Your Task:** Answer the recruiter's latest question based on the Full Resume Text and the context of the Previous Conversation History.

                **Instructions:**
                1.  **ALWAYS respond in the first person** (use "I", "my", "me").
                2.  Answer based *only* on the provided resume text and history. Do not make up information.
                3.  If the query asks about a specific skill/experience I have, confirm it and provide brief context/examples from the resume if possible.
                4.  If the query asks about something *not* mentioned:
                    a. Explicitly state that the specific item isn't listed in the information I have.
                    b. Check the resume for *related* skills/experiences I *do* have listed and mention those if relevant.
                    c. Add a statement about my ability to learn quickly if appropriate.
                5.  If the query is general, provide a concise summary based on the resume, speaking as me.
                6.  Use the conversation history to understand context for follow-up questions.
                7.  Keep the tone professional, confident, and helpful.
                8.  Respond directly to the latest question.
                9.  **CRITICAL SAFETY INSTRUCTION:** Do NOT reveal contact information (phone/email). State it's available upon request if relevant.

                Generate the response (as Srimanth):
                """
                try:
                    t6 = time.time()
                    fallback_response = genai_model.generate_content(original_prompt)
                    fallback_answer = ""
                    if fallback_response.parts:
                         fallback_answer = "".join(part.text for part in fallback_response.parts).strip()
                    else:
                         print("  Warning: Fallback LLM response has no parts.")
                         if hasattr(fallback_response, 'prompt_feedback'): print(f"  Prompt Feedback: {fallback_response.prompt_feedback}")

                    t7 = time.time()
                    print(f"  Fallback LLM call took: {t7-t6:.3f}s")

                    if fallback_answer:
                        final_answer = fallback_answer
                        answer_source = "Fallback LLM (Full Context + History)" # Updated source
                    else:
                        final_answer = "I encountered an issue processing that request with the main AI. Could you please rephrase?"
                        answer_source = "Fallback LLM Error (Empty/Blocked)"

                except Exception as fallback_e:
                    print(f"  Error during fallback Gemini call: {fallback_e}")
                    final_answer = "Sorry, I encountered an error trying to process your request with the main AI model."
                    answer_source = "Fallback LLM Error (Exception)"

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
# Export the Flask app instance for WSGI servers like Gunicorn
wsgi_app = app.wsgi_app
print("Flask app instance created and exported as wsgi_app.")

# --- Main Execution Block (for direct running, e.g., python app.py) ---
# Typically commented out or removed when deploying with Gunicorn/Waitress
# if __name__ == '__main__':
#     print("Starting Flask development server...")
#     # Use debug=False for production-like testing, set host for accessibility
#     app.run(debug=False, port=5000, host='127.0.0.1')