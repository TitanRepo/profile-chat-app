from functools import wraps
from flask import request, jsonify, g
import jwt # PyJWT library
from jwt import PyJWKClient, ExpiredSignatureError, InvalidTokenError
import traceback # Import traceback for logging

# Import necessary configuration from config.py
try:
    from config import jwks_client, COGNITO_APP_CLIENT_ID, COGNITO_ISSUER
except ImportError:
    print("Error importing configuration from config.py. Make sure config.py exists and is accessible.")
    # Set defaults to prevent crash, but auth will fail
    jwks_client = None
    COGNITO_APP_CLIENT_ID = None
    COGNITO_ISSUER = None


def token_required(f):
    """
    Decorator to validate Cognito JWT tokens.
    Expects the token in the 'Authorization: Bearer <token>' header.
    Stores the validated user ID ('sub') in flask.g.user_id.
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        # Clear any previous user ID from context at the start of the request
        g.user_id = None
        token = None
        auth_header = request.headers.get('Authorization') # Use .get for safety

        if auth_header:
            parts = auth_header.split()
            if len(parts) == 2 and parts[0].lower() == 'bearer':
                token = parts[1]
            else:
                print("Invalid Authorization header format.")
                return jsonify({'message': 'Invalid Authorization header format.'}), 401

        if not token:
            print("Auth token is missing!")
            return jsonify({'message': 'Token is missing!'}), 401

        if not jwks_client or not COGNITO_APP_CLIENT_ID or not COGNITO_ISSUER:
             print("Authentication configuration is missing or incomplete (JWKS/Client ID/Issuer).")
             return jsonify({'message': 'Server configuration error (Auth Setup)'}), 500

        try:
            # Get the signing key from JWKS cache/fetcher
            signing_key = jwks_client.get_signing_key_from_jwt(token)

            # Decode and validate the token
            payload = jwt.decode(
                token,
                signing_key.key,
                algorithms=["RS256"], # Cognito uses RS256
                audience=COGNITO_APP_CLIENT_ID, # Verify audience matches your App Client ID
                issuer=COGNITO_ISSUER # Verify issuer matches your User Pool
            )

            # Store the validated user ID ('sub') in Flask's 'g'
            g.user_id = payload.get('sub')
            if not g.user_id:
                 # This case should ideally not happen if Cognito issues valid tokens
                 print("Validated token is missing 'sub' claim.")
                 return jsonify({'message': 'Invalid token payload (missing sub)'}), 401
            
            # Token is valid
            print(f"Token validated successfully for user sub: {payload.get('sub')}")

            # Only pass user_payload if the decorated function accepts it
            from inspect import signature
            sig = signature(f)
            if 'user_payload' in sig.parameters:
                kwargs['user_payload'] = payload
            return f(*args, **kwargs)

        except ExpiredSignatureError:
            print("Auth token has expired!")
            return jsonify({'message': 'Token has expired!'}), 401
        except jwt.exceptions.InvalidAudienceError:
             print("Auth token has invalid audience!")
             return jsonify({'message': 'Token has invalid audience!'}), 401
        except jwt.exceptions.InvalidIssuerError:
             print("Auth token has invalid issuer!")
             return jsonify({'message': 'Token has invalid issuer!'}), 401
        except InvalidTokenError as e:
            # Catches other JWT errors (e.g., invalid signature, malformed)
            print(f"Auth token is invalid: {e}")
            return jsonify({'message': f'Token is invalid: {e}'}), 401
        except Exception as e:
             # Catch potential errors during key fetching or other unexpected issues
             print(f"Error during token validation: {e}")
             traceback.print_exc() # Log detailed error for debugging
             return jsonify({'message': 'Token validation error'}), 500

    return decorated

