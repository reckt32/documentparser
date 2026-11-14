"""Firebase authentication endpoints."""

from flask import Blueprint, jsonify, request

from backend.auth import require_auth, verify_firebase_token
from backend.db import create_or_update_user, get_user_by_firebase_uid
from backend.payment import get_report_price_paise

bp = Blueprint("auth", __name__)


@bp.route("/auth/verify", methods=["POST"])
def auth_verify():
    """
    Verify Firebase ID token and create/update user in database.
    Called after client-side Firebase login.
    
    Request body:
        - id_token: Firebase ID token
    
    Returns:
        - User info and payment status
    """
    data = request.get_json(force=True) or {}
    id_token = data.get("id_token")
    
    if not id_token:
        return jsonify({"error": "id_token required"}), 400
    
    decoded = verify_firebase_token(id_token)
    if not decoded:
        return jsonify({"error": "Invalid or expired token"}), 401
    
    # Create or update user in database
    firebase_uid = decoded["uid"]
    user_id = create_or_update_user(
        firebase_uid=firebase_uid,
        email=decoded.get("email"),
        display_name=decoded.get("name")
    )
    
    # Get full user info
    user = get_user_by_firebase_uid(firebase_uid)
    
    return jsonify({
        "success": True,
        "user": {
            "firebase_uid": user["firebase_uid"],
            "email": user["email"],
            "display_name": user["display_name"],
            "has_paid": user["has_paid"],
            "payment_date": user["payment_date"],
            "report_credits": user.get("report_credits", 0),
        }
    }), 200


@bp.route("/auth/status", methods=["GET"])
@require_auth
def auth_status():
    """
    Get current user's authentication and payment status.
    Requires valid Firebase ID token in Authorization header.
    """
    from flask import g
    user = g.current_user
    
    return jsonify({
        "authenticated": True,
        "user": {
            "firebase_uid": user["firebase_uid"],
            "email": user["email"],
            "display_name": user["display_name"],
            "has_paid": user["has_paid"],
            "report_credits": user.get("report_credits", 0),
            "payment_date": user["payment_date"],
        },
        "report_price_paise": get_report_price_paise(),
        "credits_per_payment": 3,
    }), 200
