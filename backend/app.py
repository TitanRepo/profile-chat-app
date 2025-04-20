# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
import os
import json
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import time # For basic timing
import traceback # For detailed error logging
import re # **NEW**: Import regex for PII filtering

load_dotenv()

app = Flask(__name__)
CORS(app)

# --- Global Variables & Configuration ---
structured_resume_data = {}
embedding_model = None
chroma_collection = None
genai_model = None
SIMILARITY_THRESHOLD = 0.7 # Adjust as needed (lower L2/Cosine distance means more similar)

# --- Initialization Functions ---

def load_structured_data(filename='resume_data_structured.json'):
    """Loads chunks and embeddings from the JSON file."""
    global structured_resume_data
    try:
        filepath = os.path.join(os.path.dirname(__file__), filename)
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
    if not structured_resume_data.get("chunks") or not structured_resume_data.get("embeddings"):
        print("Error: Cannot initialize Vector DB without loaded chunks/embeddings.")
        return False
    try:
        print(f"Initializing ChromaDB (PersistentClient at path: {db_path})...")
        client_settings = Settings(
            chroma_api_impl="chromadb.api.segment.SegmentAPI",
            persist_directory=db_path,
            is_persistent=True,
        )
        client = chromadb.PersistentClient(path=db_path, settings=client_settings)
        collection_name = "srimanth_resume_cache"
        chroma_collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
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
        # Consider safety settings for the model
        # Example: Block harmful content (adjust thresholds as needed)
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]
        genai_model = genai.GenerativeModel(
            'gemini-1.5-flash',
            safety_settings=safety_settings # Apply safety settings
            )
        print("Gemini model initialized with safety settings.")
        return True
    except Exception as e:
        print(f"Error configuring Gemini: {e}")
        return False

# **NEW**: PII Redaction Function
def redact_pii(text):
    """Redacts email addresses and basic phone numbers from text."""
    if not isinstance(text, str): # Ensure input is a string
        return text

    # Basic North American style phone - adjust regex for other international formats if needed
    # Handles formats like (123) 456-7890, 123-456-7890, 123.456.7890, 123 456 7890, 1234567890
    # Added optional country code (+1, +91 etc.)
    phone_pattern = r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'
    # Standard email pattern
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'

    redacted_text = re.sub(email_pattern, '[EMAIL_REDACTED]', text)
    redacted_text = re.sub(phone_pattern, '[PHONE_REDACTED]', redacted_text)
    return redacted_text


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
def handle_query():
    start_time = time.time()
    if not IS_INITIALIZED or not genai_model or not chroma_collection or not embedding_model:
         return jsonify({"error": "Backend not fully initialized. Please check server logs."}), 500

    user_query = request.json.get('query')
    if not user_query:
        return jsonify({"error": "No query provided."}), 400

    print(f"\nReceived query: {user_query}")
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
        DISTANCE_THRESHOLD = 0.6

        if retrieved_chunks and distances and distances[0] <= DISTANCE_THRESHOLD:
            print(f"  Best chunk distance ({distances[0]:.4f}) <= threshold ({DISTANCE_THRESHOLD}). Attempting focused LLM call.")
            context_for_llm = "\n---\n".join(retrieved_chunks)

            # **MODIFIED**: Added PII redaction instruction to focused prompt
            focused_prompt = f"""You are an AI assistant representing Srimanth.
            Based *ONLY* on the following context extracted from Srimanth's resume:
            --- CONTEXT ---
            {context_for_llm}
            --- END CONTEXT ---

            Answer the user's question concisely: "{user_query}"

            **CRITICAL SAFETY INSTRUCTION:** Do NOT reveal any contact information like phone numbers or email addresses, even if they appear in the context. If you encounter such information that is relevant to the answer, state that contact information is available upon request but do not include the actual phone number or email address in your response.

            If the context does not contain the answer to the user's question, state ONLY 'Information not found in context.' and nothing else. Do not make up information.
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

                if focused_answer and 'information not found in context' not in focused_answer.lower():
                    final_answer = focused_answer
                    answer_found_via_cache = True
                    answer_source = "Vector Cache + Focused LLM"
                    print(f"  Answer successfully generated from retrieved context.")
                else:
                    print(f"  Focused LLM call indicated context was insufficient or returned empty/blocked.")

            except Exception as focused_e:
                print(f"  Error during focused Gemini call: {focused_e}")
                # Continue to fallback

        # 4. Fallback to Broad LLM Call if needed
        if not answer_found_via_cache:
            print("  Falling back to broad context LLM call.")
            resume_context = structured_resume_data.get("raw_text", "No resume information available.")
            if resume_context == "No resume information available.":
                 print("  Error: Raw resume text not available for fallback.")
                 final_answer = "Sorry, the resume context is missing for a full answer."
                 answer_source = "Fallback Error (No Context)"
            else:
                # **MODIFIED**: Added PII redaction instruction to original prompt
                original_prompt = f"""
                You are a helpful AI assistant representing Srimanth Reddy, speaking to a recruiter.
                Your knowledge is based SOLELY on the resume text provided below.
                The recruiter's query is: "{user_query}"

                Analyze the following resume text:
                --- BEGIN RESUME ---
                {resume_context}
                --- END RESUME ---

                Instructions:
                1. Answer the recruiter's query based *only* on the provided resume text. Do not make up information.
                2. If the query asks about a specific skill or experience mentioned, confirm it and provide brief context/examples found in the text if possible.
                3. If the query asks about something *not* mentioned:
                    a. Explicitly state that the specific item isn't listed.
                    b. Check for *related* skills/experiences and mention them if relevant.
                    c. Add a statement about Srimanth's ability to learn quickly if appropriate.
                4. If the query is general, provide a concise summary based on the resume.
                5. Keep the tone professional, confident, and helpful.
                6. Respond directly to the query. Do not start with "Based on the resume...".
                7. **CRITICAL SAFETY INSTRUCTION:** Do NOT reveal any contact information like phone numbers or email addresses, even if they appear in the resume text. If you encounter such information that is relevant to the answer, state that contact information is available upon request but do not include the actual phone number or email address in your response.

                Generate the response:
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
                        answer_source = "Fallback LLM (Full Context)"
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

    # **NEW**: Apply PII redaction just before returning the response
    final_answer = redact_pii(final_answer)

    end_time = time.time()
    print(f"  Query processing finished. Source: '{answer_source}'. Final Answer (Post-Redaction): '{final_answer[:100]}...' Total time: {end_time-start_time:.3f}s") # Log snippet of final answer
    return jsonify({"answer": final_answer, "source": answer_source})

# Add this line to explicitly export the app for WSGI servers like Gunicorn/Waitress
wsgi_app = app.wsgi_app

if __name__ == '__main__':
    print("Starting Flask app...")
    app.run(debug=False, port=5000, host='127.0.0.1')
