"""
Firebase Authentication Module

This module provides Firebase token verification and authentication decorators
for protecting Flask routes. It uses Firebase Admin SDK for server-side token
verification.

Usage:
    from auth import require_auth, require_payment

    @app.route("/protected")
    @require_auth
    def protected_route():
        user = g.current_user  # Contains firebase_uid, email, etc.
        return jsonify({"user": user})

    @app.route("/paid-only")
    @require_auth
    @require_payment
    def paid_route():
        # Only accessible to users who have paid
        return jsonify({"message": "Premium content"})
"""

import os
import functools
from flask import request, jsonify, g
from typing import Optional, Dict, Any

# Firebase Admin SDK
import firebase_admin
from firebase_admin import credentials, auth as firebase_auth

from db import get_user_by_firebase_uid, create_or_update_user, check_user_payment_status

# Initialize Firebase Admin SDK
_firebase_app = None


def _init_firebase():
    """Initialize Firebase Admin SDK. Call once at app startup."""
    global _firebase_app
    if _firebase_app is not None:
        return _firebase_app

    # Option 1: Use service account JSON file
    service_account_path = os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH")
    if service_account_path and os.path.exists(service_account_path):
        cred = credentials.Certificate(service_account_path)
        _firebase_app = firebase_admin.initialize_app(cred)
        return _firebase_app

    # Option 2: Use environment variable with JSON content
    service_account_json = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")
    if service_account_json:
        import json
        cred_dict = json.loads(service_account_json)
        cred = credentials.Certificate(cred_dict)
        _firebase_app = firebase_admin.initialize_app(cred)
        return _firebase_app

    # Option 3: Use Application Default Credentials (for GCP environments)
    # Or just project ID for development
    project_id = os.getenv("FIREBASE_PROJECT_ID")
    if project_id:
        _firebase_app = firebase_admin.initialize_app(options={"projectId": project_id})
        return _firebase_app

    raise ValueError(
        "Firebase credentials not configured. Set one of: "
        "FIREBASE_SERVICE_ACCOUNT_PATH, FIREBASE_SERVICE_ACCOUNT_JSON, or FIREBASE_PROJECT_ID"
    )


def verify_firebase_token(id_token: str) -> Optional[Dict[str, Any]]:
    """
    Verify a Firebase ID token and return decoded token data.

    Args:
        id_token: The Firebase ID token from the client

    Returns:
        Decoded token containing uid, email, name, etc. or None if invalid
    """
    try:
        _init_firebase()
        decoded_token = firebase_auth.verify_id_token(id_token)
        return {
            "uid": decoded_token.get("uid"),
            "email": decoded_token.get("email"),
            "name": decoded_token.get("name"),
            "picture": decoded_token.get("picture"),
            "email_verified": decoded_token.get("email_verified", False),
            "auth_time": decoded_token.get("auth_time"),
        }
    except firebase_admin.exceptions.FirebaseError as e:
        print(f"Firebase token verification failed: {e}")
        return None
    except Exception as e:
        print(f"Token verification error: {e}")
        return None


def get_token_from_request() -> Optional[str]:
    """Extract Bearer token from Authorization header."""
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        return auth_header[7:]  # Remove "Bearer " prefix
    return None


def require_auth(f):
    """
    Decorator that requires a valid Firebase ID token.

    Sets g.current_user with user info from database (or creates new user).
    Returns 401 if token is missing or invalid.
    """
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        token = get_token_from_request()
        if not token:
            return jsonify({
                "error": "Authorization required",
                "message": "Missing Bearer token in Authorization header"
            }), 401

        decoded = verify_firebase_token(token)
        if not decoded:
            return jsonify({
                "error": "Invalid token",
                "message": "Firebase ID token is invalid or expired"
            }), 401

        # Get or create user in database
        firebase_uid = decoded["uid"]
        user = get_user_by_firebase_uid(firebase_uid)

        if not user:
            # First-time user - create in database
            create_or_update_user(
                firebase_uid=firebase_uid,
                email=decoded.get("email"),
                display_name=decoded.get("name")
            )
            user = get_user_by_firebase_uid(firebase_uid)

        # Store user in Flask's g context for use in route handlers
        g.current_user = user
        g.firebase_token = decoded

        return f(*args, **kwargs)
    return decorated_function


def require_payment(f):
    """
    Decorator that requires the user to have report credits available.

    Must be used AFTER @require_auth decorator.
    Returns 402 Payment Required if user has no credits.
    """
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        # Ensure require_auth was called first
        if not hasattr(g, 'current_user') or g.current_user is None:
            return jsonify({
                "error": "Authentication required",
                "message": "This endpoint requires authentication"
            }), 401

        user = g.current_user
        remaining_credits = user.get("report_credits", 0)
        
        if remaining_credits <= 0:
            return jsonify({
                "error": "Payment required",
                "message": "You have no report credits remaining. Please purchase more credits to continue.",
                "payment_status": "no_credits",
                "remaining_credits": 0
            }), 402

        return f(*args, **kwargs)
    return decorated_function


def consume_credit(f):
    """
    Decorator that atomically consumes one report credit after successful execution.
    
    Must be used AFTER @require_auth and @require_payment decorators.
    
    This decorator:
    1. Attempts to atomically consume one credit before calling the wrapped function
    2. If credit consumption fails (race condition), returns 402
    3. On success, includes remaining_credits in the response
    
    Note: Credit is consumed BEFORE report generation to prevent generating
    reports without valid credits in race conditions.
    """
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        from db import consume_user_credit, get_user_credits
        
        # Ensure require_auth was called first
        if not hasattr(g, 'current_user') or g.current_user is None:
            return jsonify({
                "error": "Authentication required",
                "message": "This endpoint requires authentication"
            }), 401
        
        user = g.current_user
        firebase_uid = user.get("firebase_uid")
        
        # Atomically consume one credit
        if not consume_user_credit(firebase_uid):
            return jsonify({
                "error": "Payment required",
                "message": "Unable to consume credit. You may have no credits remaining.",
                "payment_status": "no_credits",
                "remaining_credits": 0
            }), 402
        
        # Store remaining credits for use in response (after consumption)
        g.remaining_credits = get_user_credits(firebase_uid)
        
        return f(*args, **kwargs)
    return decorated_function



def optional_auth(f):
    """
    Decorator that optionally processes authentication if present.

    Unlike require_auth, this doesn't return 401 for missing/invalid tokens.
    Sets g.current_user to None if no valid auth is present.
    """
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        g.current_user = None
        g.firebase_token = None

        token = get_token_from_request()
        if token:
            decoded = verify_firebase_token(token)
            if decoded:
                firebase_uid = decoded["uid"]
                user = get_user_by_firebase_uid(firebase_uid)
                if user:
                    g.current_user = user
                    g.firebase_token = decoded

        return f(*args, **kwargs)
    return decorated_function
