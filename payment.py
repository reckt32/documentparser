"""
Razorpay Payment Module

This module handles Razorpay integration for one-time payment processing.
It provides order creation, signature verification, and webhook handling.

Usage:
    from payment import create_razorpay_order, verify_razorpay_signature

    # Create order
    order = create_razorpay_order(user_id="firebase_uid_123")

    # Verify payment after client-side success
    is_valid = verify_razorpay_signature(order_id, payment_id, signature)
"""

import os
import hmac
import hashlib
from typing import Optional, Dict, Any

import razorpay

from db import mark_user_as_paid, get_user_by_firebase_uid

# Razorpay client (lazy initialization)
_razorpay_client = None


def _get_razorpay_client() -> razorpay.Client:
    """Get or initialize Razorpay client."""
    global _razorpay_client
    if _razorpay_client is not None:
        return _razorpay_client

    key_id = os.getenv("RAZORPAY_KEY_ID")
    key_secret = os.getenv("RAZORPAY_KEY_SECRET")

    if not key_id or not key_secret:
        raise ValueError(
            "Razorpay credentials not configured. "
            "Set RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET environment variables."
        )

    _razorpay_client = razorpay.Client(auth=(key_id, key_secret))
    return _razorpay_client


def get_report_price_paise() -> int:
    """Get the report price in paise from environment or default (₹499 = 49900 paise)."""
    return int(os.getenv("REPORT_PRICE_PAISE", "49900"))


def create_razorpay_order(
    firebase_uid: str,
    amount_paise: Optional[int] = None,
    currency: str = "INR",
    receipt: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a Razorpay order for payment.

    Args:
        firebase_uid: User's Firebase UID (stored in order notes)
        amount_paise: Amount in paise (defaults to REPORT_PRICE_PAISE)
        currency: Currency code (default INR)
        receipt: Optional receipt/reference ID

    Returns:
        Order details including id, amount, currency, key_id for client
    """
    client = _get_razorpay_client()

    if amount_paise is None:
        amount_paise = get_report_price_paise()

    if receipt is None:
        import uuid
        receipt = f"rcpt_{uuid.uuid4().hex[:12]}"

    order_data = {
        "amount": amount_paise,
        "currency": currency,
        "receipt": receipt,
        "notes": {
            "firebase_uid": firebase_uid,
            "product": "financial_report_access"
        }
    }

    order = client.order.create(data=order_data)

    return {
        "order_id": order["id"],
        "amount": order["amount"],
        "currency": order["currency"],
        "receipt": order["receipt"],
        "key_id": os.getenv("RAZORPAY_KEY_ID"),
        "status": order["status"]
    }


def verify_razorpay_signature(
    order_id: str,
    payment_id: str,
    signature: str
) -> bool:
    """
    Verify Razorpay payment signature using HMAC-SHA256.

    Args:
        order_id: Razorpay order ID
        payment_id: Razorpay payment ID
        signature: Razorpay signature from payment response

    Returns:
        True if signature is valid, False otherwise
    """
    key_secret = os.getenv("RAZORPAY_KEY_SECRET")
    if not key_secret:
        print("Error: RAZORPAY_KEY_SECRET not configured")
        return False

    # Generate signature using order_id|payment_id
    message = f"{order_id}|{payment_id}"
    expected_signature = hmac.new(
        key_secret.encode(),
        message.encode(),
        hashlib.sha256
    ).hexdigest()

    return hmac.compare_digest(expected_signature, signature)


def verify_webhook_signature(payload: bytes, signature: str) -> bool:
    """
    Verify Razorpay webhook signature.

    Args:
        payload: Raw request body bytes
        signature: X-Razorpay-Signature header value

    Returns:
        True if signature is valid, False otherwise
    """
    webhook_secret = os.getenv("RAZORPAY_WEBHOOK_SECRET")
    if not webhook_secret:
        print("Error: RAZORPAY_WEBHOOK_SECRET not configured")
        return False

    expected_signature = hmac.new(
        webhook_secret.encode(),
        payload,
        hashlib.sha256
    ).hexdigest()

    return hmac.compare_digest(expected_signature, signature)


def process_payment_captured(payment_data: Dict[str, Any]) -> bool:
    """
    Process a successful payment.captured webhook event.

    Args:
        payment_data: Payment entity from webhook payload

    Returns:
        True if user was successfully marked as paid
    """
    try:
        payment_id = payment_data.get("id")
        order_id = payment_data.get("order_id")
        amount = payment_data.get("amount")
        notes = payment_data.get("notes", {})

        firebase_uid = notes.get("firebase_uid")
        if not firebase_uid:
            print(f"Warning: payment {payment_id} has no firebase_uid in notes")
            return False

        # Mark user as paid
        success = mark_user_as_paid(
            firebase_uid=firebase_uid,
            payment_id=payment_id,
            order_id=order_id,
            amount_paise=amount
        )

        if success:
            print(f"User {firebase_uid} marked as paid (payment: {payment_id})")
        else:
            print(f"Failed to mark user {firebase_uid} as paid - user may not exist")

        return success

    except Exception as e:
        print(f"Error processing payment: {e}")
        return False


def get_payment_status(firebase_uid: str) -> Dict[str, Any]:
    """
    Get payment status for a user.

    Returns:
        Dict with has_paid status and payment details if available
    """
    user = get_user_by_firebase_uid(firebase_uid)
    if not user:
        return {
            "has_paid": False,
            "user_exists": False
        }

    return {
        "has_paid": user.get("has_paid", False),
        "user_exists": True,
        "payment_id": user.get("payment_id"),
        "payment_date": user.get("payment_date"),
        "amount_paise": user.get("payment_amount_paise")
    }
