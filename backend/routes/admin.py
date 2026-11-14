"""Admin panel endpoints."""

from flask import Blueprint, jsonify, request

from backend.auth import require_admin, require_auth
from backend.db import (
    delete_user,
    get_payment_history,
    get_user_by_firebase_uid,
    get_user_count,
    list_all_users,
    set_user_credits,
)

bp = Blueprint("admin", __name__)


@bp.route("/api/admin/stats", methods=["GET"])
@require_auth
@require_admin
def admin_stats():
    """Get dashboard statistics: total users, paid users, users with credits."""
    counts = get_user_count()
    return jsonify({"success": True, "stats": counts}), 200


@bp.route("/api/admin/users", methods=["GET"])
@require_auth
@require_admin
def admin_list_users():
    """
    List all users with pagination and optional search.
    Query params: page (default 1), per_page (default 25), search (optional)
    """
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 25, type=int)
    search = request.args.get("search", "").strip() or None

    # Clamp per_page to prevent abuse
    per_page = min(max(per_page, 1), 100)

    users, total = list_all_users(page=page, per_page=per_page, search=search)
    return jsonify({
        "success": True,
        "users": users,
        "pagination": {
            "page": page,
            "per_page": per_page,
            "total": total,
            "pages": (total + per_page - 1) // per_page if per_page else 1,
        },
    }), 200


@bp.route("/api/admin/users/<firebase_uid>", methods=["GET"])
@require_auth
@require_admin
def admin_get_user(firebase_uid):
    """Get single user details including payment history."""
    user = get_user_by_firebase_uid(firebase_uid)
    if not user:
        return jsonify({"error": "User not found"}), 404

    payments = get_payment_history(firebase_uid)
    return jsonify({
        "success": True,
        "user": user,
        "payments": payments,
    }), 200


@bp.route("/api/admin/users/<firebase_uid>", methods=["DELETE"])
@require_auth
@require_admin
def admin_delete_user(firebase_uid):
    """Delete a user and their payment history."""
    deleted = delete_user(firebase_uid)
    if not deleted:
        return jsonify({"error": "User not found"}), 404
    return jsonify({"success": True, "message": "User deleted"}), 200


@bp.route("/api/admin/users/<firebase_uid>/credits", methods=["POST"])
@require_auth
@require_admin
def admin_set_credits(firebase_uid):
    """
    Set or add credits for a user.
    Request body: { "credits": int, "mode": "set" | "add" }
    """
    data = request.get_json(force=True) or {}
    credits = data.get("credits")
    mode = data.get("mode", "set")

    if credits is None or not isinstance(credits, int) or credits < 0:
        return jsonify({"error": "Valid non-negative integer 'credits' required"}), 400

    user = get_user_by_firebase_uid(firebase_uid)
    if not user:
        return jsonify({"error": "User not found"}), 404

    if mode == "add":
        from backend.db import add_user_credits
        success = add_user_credits(firebase_uid, credits)
    else:
        success = set_user_credits(firebase_uid, credits)

    if not success:
        return jsonify({"error": "Failed to update credits"}), 500

    updated_user = get_user_by_firebase_uid(firebase_uid)
    return jsonify({
        "success": True,
        "message": f"Credits {'added' if mode == 'add' else 'set'} successfully",
        "report_credits": updated_user.get("report_credits", 0),
    }), 200
