# parser.py
import PyPDF2
import re
import json

def parse_resume_pdf(pdf_path):
    """Parses a PDF resume to extract text content."""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + "\n" # Add newline between pages
        print(f"Successfully extracted text from {pdf_path}")
        return text
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
        return None

def extract_skills_from_text(resume_text):
    """
    A very basic example of extracting skills.
    You'll likely need a more sophisticated approach, possibly using
    regex, keyword lists, or even prompting an LLM for extraction.
    """
    # Example: Look for a line starting with "Skills:" or similar
    skills_section = re.search(r"Skills:?\s*(.*?)\n\n", resume_text, re.IGNORECASE | re.DOTALL)
    skills = []
    if skills_section:
        # Basic split - needs refinement based on your resume format
        skills = [s.strip() for s in skills_section.group(1).split(',')]
    # You would add more logic here for other sections (Experience, Projects, etc.)
    return skills

# --- Main execution logic (example) ---
if __name__ == "__main__":
    import os
    import sys
    
    # Default path or accept command line argument
    resume_file = sys.argv[1] if len(sys.argv) > 1 else 'C:/Users/Srimanth/Downloads/SRP-SPR.pdf'
    
    # Check if file exists
    if not os.path.exists(resume_file):
        print(f"Error: Resume file not found at {resume_file}")
        print("Please provide the correct path to your PDF resume")
        sys.exit(1)

    print(f"Processing resume file: {resume_file}")
    raw_text = parse_resume_pdf(resume_file)

    if raw_text:
        print(f"Successfully extracted {len(raw_text)} characters of text")
        resume_data = {"raw_text": raw_text}

        # Save to the same directory as the script
        output_path = os.path.join(os.path.dirname(__file__), 'resume_data.json')
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(resume_data, f, indent=4)
            print(f"Resume data saved to: {output_path}")
        except Exception as e:
            print(f"Error saving resume data: {e}")
    else:
        print("Failed to extract text from the resume")