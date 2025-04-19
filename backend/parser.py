import PyPDF2
import re
import json
import os
import sys
from sentence_transformers import SentenceTransformer
# Using LangChain's splitter is convenient, install if needed: pip install langchain
# If you don't want LangChain, you'll need to implement your own chunking logic.
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError:
    print("Warning: LangChain not found. Chunking will be very basic.")
    # Basic fallback chunking (split by double newline)
    def basic_chunker(text):
        return [chunk for chunk in text.split('\n\n') if chunk.strip()]
    # Assign the fallback
    RecursiveCharacterTextSplitter = None


def parse_resume_pdf(pdf_path):
    """Parses a PDF resume to extract text content."""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            num_pages = len(reader.pages)
            print(f"Reading {num_pages} pages from {pdf_path}...")
            for i, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n" # Add newline between pages
                    else:
                        print(f"Warning: No text extracted from page {i+1}")
                except Exception as page_e:
                    print(f"Error extracting text from page {i+1}: {page_e}")
            print(f"Successfully extracted text (length: {len(text)}) from {pdf_path}")
        return text
    except FileNotFoundError:
        print(f"Error: File not found at {pdf_path}")
        return None
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
        return None

def chunk_text(text, chunk_size=700, chunk_overlap=70):
    """Chunks the text using RecursiveCharacterTextSplitter or basic fallback."""
    if RecursiveCharacterTextSplitter:
        print(f"Using LangChain RecursiveCharacterTextSplitter (chunk_size={chunk_size}, chunk_overlap={chunk_overlap})")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        return text_splitter.split_text(text)
    else:
        print("Using basic chunking (split by double newline).")
        return basic_chunker(text)

def generate_embeddings(text_chunks, model_name='all-MiniLM-L6-v2'):
    """Generates embeddings for text chunks using SentenceTransformer."""
    print(f"Loading embedding model: {model_name}")
    try:
        model = SentenceTransformer(model_name)
        print("Generating embeddings...")
        embeddings = model.encode(text_chunks, show_progress_bar=True)
        print(f"Generated {len(embeddings)} embeddings.")
        # Convert numpy arrays to lists for JSON serialization
        return [emb.tolist() for emb in embeddings]
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return None

# --- Main execution logic ---
if __name__ == "__main__":
    # Default path or accept command line argument
    # *** IMPORTANT: Update this path to your actual resume file ***
    default_resume_path = 'C:/Users/Srimanth/Downloads/SRP-SPR.pdf' # CHANGE THIS
    resume_file = sys.argv[1] if len(sys.argv) > 1 else default_resume_path

    # Check if file exists
    if not os.path.exists(resume_file):
        print(f"Error: Resume file not found at {resume_file}")
        print("Please provide the correct path to your PDF resume as a command line argument or update the default_resume_path variable in the script.")
        sys.exit(1)

    print(f"Processing resume file: {resume_file}")
    raw_text = parse_resume_pdf(resume_file)

    if raw_text:
        print(f"Successfully extracted {len(raw_text)} characters of text.")

        # 1. Chunk the text
        chunks = chunk_text(raw_text)
        if not chunks:
            print("Error: Failed to chunk text.")
            sys.exit(1)
        print(f"Split text into {len(chunks)} chunks.")

        # 2. Generate Embeddings
        embeddings = generate_embeddings(chunks)
        if not embeddings:
            print("Error: Failed to generate embeddings.")
            sys.exit(1)

        # 3. Prepare data for JSON
        # We save raw_text for the fallback scenario
        resume_data_structured = {
            "raw_text": raw_text,
            "chunks": chunks,
            "embeddings": embeddings
        }

        # 4. Save structured data
        output_filename = 'resume_data_structured.json'
        output_path = os.path.join(os.path.dirname(__file__), output_filename)
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(resume_data_structured, f, indent=4)
            print(f"Structured resume data saved to: {output_path}")
            print("---")
            print(f"IMPORTANT: Make sure the Flask app ('app.py') loads '{output_filename}'")
            print("---")
        except Exception as e:
            print(f"Error saving structured resume data: {e}")
    else:
        print("Failed to extract text from the resume.")
        sys.exit(1)