"""Razorpay payment endpoints."""

import logging

from flask import Blueprint, jsonify, request

from backend.auth import require_auth
from backend.payment import (
    create_razorpay_order,
    get_report_price_paise,
    process_payment_captured,
    verify_razorpay_signature,
    verify_webhook_signature,
)

logger = logging.getLogger(__name__)
bp = Blueprint("payment", __name__)


@bp.route("/payment/create-order", methods=["POST"])
@require_auth
def payment_create_order():
    """
    Create a Razorpay order for payment.
    Requires valid Firebase ID token in Authorization header.
    
    Users can purchase multiple times to stack credits.
    Each payment grants 3 report credits.
    
    Returns:
        - Razorpay order details for client-side payment
    """
    from flask import g
    user = g.current_user
    
    # Note: We no longer block "already paid" users - they can buy more credits
    
    try:
        order = create_razorpay_order(firebase_uid=user["firebase_uid"])
        return jsonify({
            "success": True,
            "order": order,
            "user_email": user.get("email"),
            "current_credits": user.get("report_credits", 0),
            "credits_per_payment": 3,
        }), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        logger.error(f"Error creating Razorpay order: {e}")
        return jsonify({"error": "Failed to create payment order"}), 500


@bp.route("/payment/verify", methods=["POST"])
@require_auth
def payment_verify():
    """
    Verify payment after successful Razorpay checkout.
    Requires valid Firebase ID token in Authorization header.
    
    Request body:
        - order_id: Razorpay order ID
        - payment_id: Razorpay payment ID
        - signature: Razorpay signature
    
    Returns:
        - Payment verification status and updated credit count
    """
    from flask import g
    from backend.db import mark_user_as_paid, get_user_credits
    
    user = g.current_user
    data = request.get_json(force=True) or {}
    
    order_id = data.get("order_id")
    payment_id = data.get("payment_id")
    signature = data.get("signature")
    
    if not all([order_id, payment_id, signature]):
        return jsonify({"error": "Missing payment verification data"}), 400
    
    # Verify signature
    if not verify_razorpay_signature(order_id, payment_id, signature):
        return jsonify({"error": "Invalid payment signature"}), 400
    
    # Mark user as paid and add credits (default 3 credits per payment)
    success = mark_user_as_paid(
        firebase_uid=user["firebase_uid"],
        payment_id=payment_id,
        order_id=order_id,
        amount_paise=get_report_price_paise(),
        credits_to_add=3
    )
    
    if success:
        # Get updated credit count
        remaining_credits = get_user_credits(user["firebase_uid"])
        return jsonify({
            "success": True,
            "message": "Payment verified successfully. 3 report credits added.",
            "has_paid": True,
            "remaining_credits": remaining_credits,
            "credits_added": 3
        }), 200
    else:
        return jsonify({
            "error": "Failed to update payment status",
            "message": "Please contact support"
        }), 500


@bp.route("/payment/webhook", methods=["POST"])
def payment_webhook():
    """
    Razorpay webhook handler for payment events.
    Verifies webhook signature and processes payment.captured events.
    
    IMPORTANT: Returns 4xx/5xx on errors so Razorpay will retry the webhook.
    """
    # Get raw payload for signature verification
    payload = request.get_data()
    signature = request.headers.get("X-Razorpay-Signature", "")
    
    if not signature:
        logger.warning("Webhook received without signature - rejecting")
        return jsonify({"error": "Missing signature"}), 400  # 400 so Razorpay knows it failed
    
    # Verify webhook signature
    if not verify_webhook_signature(payload, signature):
        logger.warning("Webhook signature verification failed")
        return jsonify({"error": "Invalid signature"}), 400
    
    # Parse payload
    try:
        data = request.get_json(force=True)
    except Exception as e:
        logger.error(f"Webhook JSON parse error: {e}")
        return jsonify({"error": "Invalid JSON"}), 400
    
    event = data.get("event")
    logger.info(f"Processing Razorpay webhook event: {event}")
    
    if event == "payment.captured":
        payment_entity = data.get("payload", {}).get("payment", {}).get("entity", {})
        if payment_entity:
            success = process_payment_captured(payment_entity)
            if not success:
                # Return 500 so Razorpay retries the webhook
                logger.error(f"Failed to process payment.captured for payment {payment_entity.get('id')}")
                return jsonify({"error": "Processing failed", "retry": True}), 500
        else:
            logger.warning("payment.captured event received but no payment entity found")
        return jsonify({"status": "processed"}), 200
    
    # Acknowledge other events
    logger.info(f"Acknowledged Razorpay webhook event: {event}")
    return jsonify({"status": "acknowledged"}), 200


@bp.route("/payment/status", methods=["GET"])
@require_auth
def payment_status():
    """
    Get current user's payment status and credit information.
    """
    from flask import g
    user = g.current_user
    remaining_credits = user.get("report_credits", 0)
    
    return jsonify({
        "has_paid": user.get("has_paid", False),
        "payment_id": user.get("payment_id"),
        "payment_date": user.get("payment_date"),
        "report_price_paise": get_report_price_paise(),
        "remaining_credits": remaining_credits,
        "can_generate_report": remaining_credits > 0,
        "credits_per_payment": 3,
    }), 200


@bp.route("/payment/reconcile", methods=["POST"])
@require_auth
def payment_reconcile():
    """
    Reconcile payment status by checking Razorpay directly.
    
    Use this when a user claims they paid but their status shows unpaid.
    This can happen if:
    - /payment/verify call failed/timed out
    - Webhook didn't fire or failed
    - Database wasn't updated for some reason
    
    This endpoint queries Razorpay for orders with the user's firebase_uid
    and marks them as paid if a completed payment is found.
    """
    from flask import g
    from backend.db import mark_user_as_paid
    from backend.payment import _get_razorpay_client
    
    user = g.current_user
    firebase_uid = user["firebase_uid"]
    
    # If already marked as paid in our system, no need to reconcile
    if user.get("has_paid"):
        return jsonify({
            "status": "already_paid",
            "message": "User already marked as paid in our system",
            "has_paid": True
        }), 200
    
    try:
        client = _get_razorpay_client()
        
        # Fetch recent orders from Razorpay (last 100)
        # We look for orders with this user's firebase_uid in notes
        orders = client.order.all({"count": 100})
        
        for order in orders.get("items", []):
            notes = order.get("notes", {})
            
            # Check if this order belongs to the current user
            if notes.get("firebase_uid") != firebase_uid:
                continue
            
            # Check if the order is paid
            if order.get("status") == "paid":
                order_id = order.get("id")
                
                # Get the payments for this order to find the payment_id
                try:
                    payments = client.order.payments(order_id)
                    payment_items = payments.get("items", [])
                    
                    if payment_items:
                        # Use the first captured/authorized payment
                        payment = payment_items[0]
                        payment_id = payment.get("id")
                        amount = order.get("amount")
                        
                        # Mark user as paid
                        success = mark_user_as_paid(
                            firebase_uid=firebase_uid,
                            payment_id=payment_id,
                            order_id=order_id,
                            amount_paise=amount
                        )
                        
                        if success:
                            logger.info(
                                f"Reconciliation successful: User {firebase_uid} marked as paid. "
                                f"Order: {order_id}, Payment: {payment_id}"
                            )
                            return jsonify({
                                "status": "reconciled",
                                "message": "Found paid order in Razorpay, user marked as paid",
                                "has_paid": True,
                                "order_id": order_id,
                                "payment_id": payment_id
                            }), 200
                        else:
                            logger.error(f"Reconciliation failed: Could not update user {firebase_uid}")
                            
                except Exception as e:
                    logger.error(f"Error fetching payments for order {order_id}: {e}")
                    continue
        
        # No paid orders found for this user
        logger.info(f"Reconciliation complete: No paid orders found for user {firebase_uid}")
        return jsonify({
            "status": "no_payment_found",
            "message": "No completed payment found in Razorpay for this user",
            "has_paid": False
        }), 200
        
    except Exception as e:
        logger.exception(f"Error during payment reconciliation for user {firebase_uid}: {e}")
        return jsonify({
            "error": "Reconciliation failed",
            "message": "Could not check payment status with Razorpay"
        }), 500
