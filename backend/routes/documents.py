"""Document upload, download and debug endpoints."""

import io
import json
import logging
import os

from flask import Blueprint, jsonify, request, send_file, send_from_directory, url_for

from backend.config import OUTPUT_DIR, SAVE_TX_JSON, STORE_REPORTS
from backend.db import (
    get_document_by_sha,
    link_questionnaire_upload,
    list_metrics,
    list_sections,
    update_questionnaire_upload_metadata,
)
from backend.extraction.bank import (
    extract_bank_statement_hybrid,
    extract_bank_statement_transactions,
)
from backend.extraction.extractors import index_and_extract
from backend.extraction.insights import (
    aggregate_doc_insights_for_questionnaire,
    build_prefill_from_insights,
)
from backend.extraction.insurance import extract_insurance_hybrid
from backend.extraction.itr import extract_itr_hybrid
from backend.extraction.metrics import _persist_metrics_for_doc
from backend.extraction.mutual_fund import extract_mutual_fund_cas_hybrid
from backend.extraction.pdf_text import _validate_pdf_file, extract_structured_text_with_tables
from backend.reports.extraction_summary import generate_pdf_summary

logger = logging.getLogger(__name__)
bp = Blueprint("documents", __name__)

# Simple in-memory store for ephemeral (one-time) downloads
TEMP_REPORTS = {}


def _register_temp_download(data: bytes, filename: str, mimetype: str = "application/pdf") -> str:
    token = os.urandom(16).hex()
    TEMP_REPORTS[token] = (filename, data, mimetype)
    return token


@bp.route('/upload', methods=['POST'])
def upload_document():
    print("Upload request received.")
    files, types = [], []
    for key in request.files:
        if key.startswith('file'):
            files.append(request.files[key])
            types.append(request.form.get(f"type{key[4:]}", "Unknown"))
    questionnaire_id_raw = request.form.get("questionnaireId") or request.form.get("questionnaire_id")
    questionnaire_id = None
    try:
        if questionnaire_id_raw:
            questionnaire_id = int(questionnaire_id_raw)
    except Exception:
        questionnaire_id = None

    if not files:
        return jsonify({"error": "No files uploaded"}), 400

    extracted_data = {}
    debug_shas = []
    upload_link_ids = {}  # Track upload link IDs for metadata updates
    for idx, (file, doc_type) in enumerate(zip(files, types)):
        try:
            logger.info(f"Processing file {idx + 1}: {file.filename}")
            # Read file bytes once so we can reuse for multiple parsers
            file_bytes = file.read()
            is_valid, validation_error = _validate_pdf_file(file_bytes, file.filename or "upload.pdf")
            if not is_valid:
                extracted_data[f"Document {idx+1} ({doc_type})"] = {"error": validation_error}
                continue
            file_stream_for_text = io.BytesIO(file_bytes)
            text = extract_structured_text_with_tables(file_stream_for_text)
            logger.debug(f"Extracted text for file {idx + 1} successfully.")

            # Index and extract deterministic summaries (snapshot/statement) with provenance
            sha, doc_id, summaries = index_and_extract(file_bytes, filename=file.filename or "upload.pdf")
            debug_shas.append(sha)
            # Link to questionnaire if provided
            if questionnaire_id:
                try:
                    link_id = link_questionnaire_upload(
                        questionnaire_id=questionnaire_id,
                        document_id=doc_id,
                        sha256=sha,
                        doc_type=doc_type,
                        filename=file.filename,
                        metadata={"size_bytes": len(file_bytes)}
                    )
                    # Store link_id for updating CAS metadata later
                    upload_link_ids[idx] = link_id
                except Exception as e:
                    logger.error(f"Failed linking upload to questionnaire {questionnaire_id}: {e}")
            
            if not text.strip():
                extracted_data[f"Document {idx+1} ({doc_type})"] = {"error": "No text could be extracted from this PDF."}
                continue
            
            # Map document types to functions
            doc_map = {
                "Bank statement": ("Bank Statement", extract_bank_statement_hybrid),
                "ITR": ("ITR", extract_itr_hybrid),
                "Insurance document": ("Insurance Document", extract_insurance_hybrid),
                "Mutual fund CAS (Consolidated Account Statement)": ("Mutual Fund CAS", extract_mutual_fund_cas_hybrid)
            }
            
            if doc_type in doc_map:
                name, func = doc_map[doc_type]
                if doc_type == "Bank statement":
                    # Build structured transactions and optionally persist as JSON
                    tx_payload = extract_bank_statement_transactions(io.BytesIO(file_bytes))
                    json_url = None
                    if SAVE_TX_JSON:
                        json_name = f"bank_transactions_{os.urandom(6).hex()}.json"
                        json_path = os.path.join(OUTPUT_DIR, json_name)
                        try:
                            with open(json_path, 'w', encoding='utf-8') as jf:
                                json.dump(tx_payload, jf, ensure_ascii=False, indent=2)
                            json_url = url_for("documents.download_file", filename=json_name, _external=True)
                        except Exception as je:
                            print(f"Failed to save transactions JSON: {je}")
                            json_url = None

                    bank_data = func(text, transactions_payload=tx_payload, save_json_path=json_url)
                    # Merge DB-backed summaries if present
                    if isinstance(summaries, dict):
                        if summaries.get("account_summary"):
                            # Per-key merge: indexed values win for the keys they carry,
                            # but LLM-only fields (account_holder_name,
                            # average_monthly_balance) must survive.
                            merged_acct = dict(bank_data.get("account_summary") or {})
                            merged_acct.update({
                                k: v for k, v in summaries["account_summary"].items() if v is not None
                            })
                            bank_data["account_summary"] = merged_acct
                        if summaries.get("investment_snapshot"):
                            bank_data["investment_snapshot"] = summaries["investment_snapshot"]
                        if summaries.get("portfolio_summary"):
                            bank_data["portfolio_summary"] = summaries["portfolio_summary"]
                        if summaries.get("provenance"):
                            bank_data["provenance"] = summaries["provenance"]
                    # Do not include raw transactions in PDF; attach summary only
                    extracted_data[f"{name} {idx+1}"] = bank_data
                    try:
                        _persist_metrics_for_doc(doc_id, bank_data)
                    except Exception as e:
                        print(f"Persist metrics (bank) failed: {e}")
                    
                    # Save bank statement metadata including recurring debits (for EMI prefill)
                    if idx in upload_link_ids:
                        try:
                            bank_metadata = {
                                "size_bytes": len(file_bytes),
                                "account_holder_name": (bank_data.get("account_summary") or {}).get("account_holder_name"),
                                "bank_data": {
                                    "account_summary": bank_data.get("account_summary", {}),
                                    "recurring_debits": bank_data.get("recurring_debits", []),
                                    "recurring_credits": bank_data.get("recurring_credits", []),
                                }
                            }
                            update_questionnaire_upload_metadata(upload_link_ids[idx], bank_metadata)
                            recurring_debits_count = len(bank_data.get("recurring_debits", []))
                            logger.info(f"Bank statement metadata saved: {recurring_debits_count} recurring debits")
                        except Exception as e:
                            logger.error(f"Error updating Bank statement metadata: {e}")
                else:
                    other_data = func(text)
                    # Merge DB-backed summaries if present (useful for CAS/Portfolio PDFs)
                    if isinstance(summaries, dict):
                        if summaries.get("account_summary"):
                            other_data["account_summary"] = summaries["account_summary"]
                        if summaries.get("investment_snapshot"):
                            other_data["investment_snapshot"] = summaries["investment_snapshot"]
                        if summaries.get("portfolio_summary"):
                            other_data["portfolio_summary"] = summaries["portfolio_summary"]
                        if summaries.get("provenance"):
                            other_data["provenance"] = summaries["provenance"]

                    # Update document metadata if linked to questionnaire
                    logger.debug(f"Checking upload link: doc_type={doc_type}, idx={idx}")
                    if idx in upload_link_ids:
                        try:
                            metadata_update = {"size_bytes": len(file_bytes)}
                            
                            # CAS metadata
                            if doc_type == "Mutual fund CAS (Consolidated Account Statement)":
                                metadata_update["investor_name"] = other_data.get("investor_name")
                                metadata_update["cas_data"] = {
                                    "sip_details": other_data.get("sip_details", []),
                                    "transaction_summary": other_data.get("transaction_summary", {}),
                                    "asset_allocation": other_data.get("asset_allocation", {}),
                                    "investment_snapshot": other_data.get("investment_snapshot", {}),
                                    "holdings": other_data.get("holdings", []),
                                }
                            
                            # ITR metadata - store personal info
                            elif doc_type == "ITR":
                                metadata_update["assessee_name"] = other_data.get("assessee_name")
                                metadata_update["date_of_birth"] = other_data.get("date_of_birth")
                                metadata_update["pan"] = other_data.get("pan")
                            
                            # Insurance metadata - store policy holder and insurance_type
                            elif doc_type == "Insurance document":
                                metadata_update["policy_holder"] = other_data.get("policy_holder")
                                metadata_update["insurance_type"] = other_data.get("insurance_type")
                                metadata_update["insurer_name"] = other_data.get("insurer_name")
                                metadata_update["policy_number"] = other_data.get("policy_number")
                                metadata_update["sum_assured_or_insured"] = other_data.get("sum_assured_or_insured")
                                metadata_update["date_of_birth"] = other_data.get("date_of_birth")
                                logger.debug(f"Insurance metadata: type={metadata_update.get('insurance_type')}, sum={metadata_update.get('sum_assured_or_insured')}")
                            
                            update_questionnaire_upload_metadata(upload_link_ids[idx], metadata_update)
                            logger.info(f"Metadata updated: doc_type={doc_type}, upload_id={upload_link_ids[idx]}")
                        except Exception as e:
                            logger.error(f"Error updating {doc_type} metadata: {e}")

                    extracted_data[f"{name} {idx+1}"] = other_data
                    try:
                        _persist_metrics_for_doc(doc_id, other_data)
                    except Exception as e:
                        logger.error(f"Persist metrics ({doc_type}) failed: {e}")
            else:
                # Generic fallback: still return deterministic DB-backed summaries even if doc type is unknown
                generic = {}
                if isinstance(summaries, dict):
                    if summaries.get("account_summary"):
                        generic["account_summary"] = summaries["account_summary"]
                    if summaries.get("investment_snapshot"):
                        generic["investment_snapshot"] = summaries["investment_snapshot"]
                    if summaries.get("portfolio_summary"):
                        generic["portfolio_summary"] = summaries["portfolio_summary"]
                    if summaries.get("provenance"):
                        generic["provenance"] = summaries["provenance"]
                if not generic:
                    generic = {"error": "Unknown document type"}
                extracted_data[f"Document {idx+1} ({doc_type})"] = generic
                
        except Exception as e:
            extracted_data[f"Document {idx+1}"] = {"error": f"Failed to process file: {str(e)}"}

    # If questionnaire_id is present, DO NOT generate the plan here.
    # Only link uploads and return insights/prefill; plan should be generated after questionnaire submission.
    if questionnaire_id:
        try:
            doc_insights = aggregate_doc_insights_for_questionnaire(questionnaire_id)
        except Exception as e:
            doc_insights = {"error": str(e)}
        # Build lightweight prefill suggestions to allow frontend to pre-populate fields
        try:
            prefill = build_prefill_from_insights(questionnaire_id)
        except Exception as e:
            prefill = {"error": str(e)}
        return jsonify({
            "summary_pdf_url": None,  # no summary when using questionnaire flow
            "debug_shas": debug_shas,
            "questionnaire_id": questionnaire_id,
            "docInsights": doc_insights,
            "prefill": prefill
        }), 200

    # Fallback to legacy summary report if no questionnaire_id provided
    pdf_filename = f"PortfolioSummary_{os.urandom(8).hex()}.pdf"
    pdf_path = os.path.join(OUTPUT_DIR, pdf_filename)
    generate_pdf_summary(extracted_data, pdf_path)

    if STORE_REPORTS == "memory":
        with open(pdf_path, "rb") as f:
            data = f.read()
        token = _register_temp_download(data, pdf_filename)
        try:
            os.remove(pdf_path)
        except Exception:
            pass
        download_url = url_for("documents.download_temp", token=token, _external=True)
    else:
        download_url = url_for("documents.download_file", filename=pdf_filename, _external=True)
    return jsonify({
        "summary_pdf_url": download_url,
        "debug_shas": debug_shas
    }), 200


@bp.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=True)


@bp.route('/download-temp/<token>')
def download_temp(token):
    item = TEMP_REPORTS.pop(token, None)
    if not item:
        return jsonify({"error": "not found"}), 404
    filename, data, mimetype = item
    return send_file(io.BytesIO(data), mimetype=mimetype, as_attachment=True, download_name=filename)


# Debug endpoints to inspect indexing and extracted metrics
@bp.route('/debug/docs/<sha>', methods=['GET'])
def debug_list_sections(sha):
    row = get_document_by_sha(sha)
    if not row:
        return jsonify({"error": "not found"}), 404
    doc_id = row["id"]
    secs = [dict(r) for r in list_sections(doc_id)]
    return jsonify({
        "sha": sha,
        "filename": row["filename"],
        "page_count": row["page_count"],
        "sections": secs
    }), 200


@bp.route('/debug/docs/<sha>/metrics', methods=['GET'])
def debug_list_metrics(sha):
    row = get_document_by_sha(sha)
    if not row:
        return jsonify({"error": "not found"}), 404
    doc_id = row["id"]
    mets = [dict(r) for r in list_metrics(doc_id)]
    return jsonify({
        "sha": sha,
        "metrics": mets
    }), 200
