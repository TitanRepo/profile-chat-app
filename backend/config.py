# config.py
import os
import time
from dotenv import load_dotenv
import requests
from jwt import PyJWKClient

# Load environment variables from .env file
load_dotenv()

# --- Cognito Configuration ---
COGNITO_REGION = os.getenv('COGNITO_REGION')  # Changed from AWS_REGION to match .env
COGNITO_USERPOOL_ID = os.getenv('COGNITO_USERPOOL_ID')
COGNITO_APP_CLIENT_ID = os.getenv('COGNITO_APP_CLIENT_ID')

# Validate required environment variables
if not all([COGNITO_REGION, COGNITO_USERPOOL_ID, COGNITO_APP_CLIENT_ID]):
    print("ERROR: Missing required AWS Cognito configuration.")
    print("Please ensure these environment variables are set in your .env file:")
    print("- COGNITO_REGION")
    print("- COGNITO_USERPOOL_ID")
    print("- COGNITO_APP_CLIENT_ID")
    raise ValueError("Missing required AWS Cognito configuration")

# Construct Cognito URLs
COGNITO_ISSUER = f'https://cognito-idp.{COGNITO_REGION}.amazonaws.com/{COGNITO_USERPOOL_ID}'
JWKS_URL = f'{COGNITO_ISSUER}/.well-known/jwks.json'

# --- Initialize JWKS Client ---
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

def initialize_jwks_client():
    """Initialize the JWKS client with retry logic"""
    for attempt in range(MAX_RETRIES):
        try:
            print(f"Attempt {attempt + 1}/{MAX_RETRIES}: Connecting to JWKS endpoint")
            
            # Test connection first
            response = requests.get(
                JWKS_URL, 
                timeout=5,
                headers={'User-Agent': 'Python/JWT-Validator'}
            )
            response.raise_for_status()
            
            # If connection successful, create JWKS client with only supported options
            jwks_client = PyJWKClient(JWKS_URL)
            print(f"Successfully connected to JWKS endpoint: {JWKS_URL}")
            return jwks_client

        except requests.exceptions.RequestException as e:
            print(f"Connection error (attempt {attempt + 1}): {str(e)}")
            if attempt < MAX_RETRIES - 1:
                print(f"Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
            else:
                print("\nFATAL: Could not establish connection to JWKS endpoint")
                print("Please check:")
                print(f"1. Your internet connection")
                print(f"2. AWS Region ({COGNITO_REGION}) and User Pool ID ({COGNITO_USERPOOL_ID}) are correct")
                print(f"3. AWS Cognito service is available in your region")
                print(f"4. The JWKS URL is accessible: {JWKS_URL}")
                return None

# Initialize JWKS client
jwks_client = initialize_jwks_client()

# --- Other Configurations ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

