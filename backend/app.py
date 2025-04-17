# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS # To allow requests from your frontend
import google.generativeai as genai
import os
import json
from dotenv import load_dotenv

load_dotenv() # Load environment variables from .env file

app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# Add this line to explicitly export the app
wsgi_app = app.wsgi_app

# --- Load Resume Data ---
resume_data = {}
try:
    with open('resume_data.json', 'r') as f:
        resume_data = json.load(f)
    if not resume_data.get("raw_text"):
         print("Warning: 'raw_text' not found in resume_data.json")
except FileNotFoundError:
    print("Error: resume_data.json not found. Run the parser script first.")
    # You might want to exit or handle this more gracefully
except json.JSONDecodeError:
    print("Error: Could not decode resume_data.json.")
    # Handle error

# --- Configure Gemini ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("Error: GEMINI_API_KEY environment variable not set.")
    # Handle error (exit or disable LLM features)
else:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-2.0-flash') # Or choose another appropriate model

# --- API Endpoint ---
@app.route('/query', methods=['POST'])
def handle_query():
    if not resume_data or not GEMINI_API_KEY:
         return jsonify({"error": "Backend not configured correctly (resume data or API key missing)."}), 500

    user_query = request.json.get('query')
    if not user_query:
        return jsonify({"error": "No query provided."}), 400

    resume_context = resume_data.get("raw_text", "No resume information available.")

    # --- Prepare the Prompt for Gemini ---
    prompt = f"""
    You are a helpful AI assistant representing Srimanth Reddy, speaking to a recruiter.
    Your knowledge is based SOLELY on the resume text provided below.
    The recruiter's query is: "{user_query}"

    Analyze the following resume text:
    --- BEGIN RESUME ---
    {resume_context}
    --- END RESUME ---

    Instructions:
    1. Answer the recruiter's query based *only* on the provided resume text. Do not make up information.
    2. If the query asks about a specific skill or experience mentioned in the resume, confirm it and provide brief context or examples found in the text if possible.
    3. If the query asks about a skill or experience *not* mentioned in the resume:
        a. Explicitly state that the specific skill isn't listed.
        b. Check the resume for *similar* or *related* skills/technologies/experiences and mention them. For example, if they ask for 'AWS EKS' and the resume lists 'Docker' and 'Kubernetes', mention those.
        c. *Crucially*, add a statement emphasizing Srimanth's proven ability to learn new technologies quickly and strong analytical skills, making them confident they can master the requested skill.
    4. If the query is general (e.g., "Tell me about your experience"), provide a concise summary based on the resume.
    5. Keep the tone professional, confident, and helpful.
    6. Respond directly to the query. Do not start with "Based on the resume..." unless it feels natural in the flow.

    Generate the response:
    """

    try:
        response = model.generate_content(prompt)
        answer = response.text
        # Basic check for safety ratings if needed (response.prompt_feedback)
        if not answer: # Handle cases where the model might return an empty response
             answer = "I encountered an issue processing that request. Could you please rephrase?"

    except Exception as e:
        print(f"Error generating content with Gemini: {e}")
        answer = "Sorry, I encountered an error trying to process your request with the AI model."

    return jsonify({"answer": answer})

if __name__ == '__main__':
    # Remember to set the FLASK_APP environment variable: export FLASK_APP=app.py
    # Then run: flask run
    app.run(debug=True) # Use debug=False in production