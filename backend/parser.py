import PyPDF2
import re
import json
import os
import sys
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import pickle # To save the BM25 object

# Using LangChain's splitter is convenient
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError:
    print("Warning: LangChain not found. Chunking will be very basic.")
    def basic_chunker(text):
        return [chunk for chunk in text.split('\n\n') if chunk.strip()]
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
                        # Basic cleaning - remove excessive newlines/spaces
                        cleaned_page = re.sub(r'\s{2,}', ' ', page_text.replace('\n', ' '))
                        text += cleaned_page + "\n" # Use newline as separator again
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
            is_separator_regex=False, # Treat newlines as separators if needed
            separators=["\n\n", "\n", ". ", " ", ""] # Common separators
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
        return [emb.tolist() for emb in embeddings]
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return None

# Function to create BM25 index
def create_bm25_index(text_chunks):
    """Creates a BM25 index from text chunks."""
    print("Tokenizing chunks for BM25...")
    # Basic tokenization: lowercase and split by space. Consider more robust tokenization (e.g., nltk, spacy)
    tokenized_corpus = [chunk.lower().split(" ") for chunk in text_chunks]
    print("Creating BM25 index...")
    try:
        bm25 = BM25Okapi(tokenized_corpus)
        print("BM25 index created.")
        return bm25
    except Exception as e:
        print(f"Error creating BM25 index: {e}")
        return None

# --- Main execution logic ---
if __name__ == "__main__":
    default_resume_path = 'C:/Users/Srimanth/Downloads/Profile Chat Resume.pdf' # CHANGE THIS
    resume_file = sys.argv[1] if len(sys.argv) > 1 else default_resume_path

    if not os.path.exists(resume_file):
        print(f"Error: Resume file not found at {resume_file}")
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

        # 3. Create BM25 Index
        bm25_index = create_bm25_index(chunks)
        if not bm25_index:
            print("Error: Failed to create BM25 index.")
            # Decide if you want to exit or continue without BM25
            sys.exit(1)

        # 4. Prepare data for JSON (excluding BM25 object)
        resume_data_structured = {
            "raw_text": raw_text,
            "chunks": chunks,
            "embeddings": embeddings
            # BM25 object will be saved separately using pickle
        }

        # 5. Save structured data (JSON)
        output_filename_json = 'resume_data_structured.json'
        output_path_json = os.path.join(os.path.dirname(__file__), output_filename_json)
        try:
            with open(output_path_json, 'w', encoding='utf-8') as f:
                json.dump(resume_data_structured, f, indent=4)
            print(f"Structured resume data saved to: {output_path_json}")
        except Exception as e:
            print(f"Error saving structured resume data (JSON): {e}")
            sys.exit(1) # Exit if JSON saving fails

        # 6. Save BM25 index (Pickle)
        output_filename_bm25 = 'resume_bm25_index.pkl'
        output_path_bm25 = os.path.join(os.path.dirname(__file__), output_filename_bm25)
        try:
            with open(output_path_bm25, 'wb') as f_pkl:
                pickle.dump(bm25_index, f_pkl)
            print(f"BM25 index saved to: {output_path_bm25}")
            print("---")
            print(f"IMPORTANT: Ensure Flask app ('app.py') loads BOTH '{output_filename_json}' AND '{output_filename_bm25}'")
            print("---")
        except Exception as e:
            print(f"Error saving BM25 index (Pickle): {e}")
            # Decide if you want to exit or allow running without BM25 cache

    else:
        print("Failed to extract text from the resume.")
        sys.exit(1)