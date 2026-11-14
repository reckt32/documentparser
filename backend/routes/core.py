"""Index, health check and legacy login endpoints."""

from flask import Blueprint, jsonify, request, send_from_directory

bp = Blueprint("core", __name__)


@bp.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')


@bp.route('/healthz', methods=['GET'])
def healthz():
    return jsonify({"status": "ok"}), 200


@bp.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    if data.get('username') == 'user' and data.get('password') == 'password':
        return jsonify({"message": "Login successful", "token": "fake-jwt-token"}), 200
    return jsonify({"message": "Invalid credentials"}), 401
