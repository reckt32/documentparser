"""PDF validation, text extraction and text clean-up helpers."""

import logging
import re

import pdfplumber

from backend.extraction.metadata import _redact_pii

logger = logging.getLogger(__name__)

# --- Security: PDF Validation ---
MAX_PDF_SIZE_BYTES = 10 * 1024 * 1024  # 10MB limit


def _validate_pdf_file(file_bytes: bytes, filename: str = "upload.pdf") -> tuple:
    """
    Validate a PDF file for security concerns.
    
    Args:
        file_bytes: Raw file content
        filename: Original filename for logging
    
    Returns:
        (is_valid, error_message) tuple
    """
    # Check file size
    if len(file_bytes) > MAX_PDF_SIZE_BYTES:
        return False, f"File too large ({len(file_bytes) / 1024 / 1024:.1f}MB). Maximum size is 10MB."
    
    if len(file_bytes) < 4:
        return False, "File is too small to be a valid PDF."
    
    # Check PDF magic bytes (%PDF)
    if not file_bytes[:4] == b'%PDF':
        return False, "Invalid PDF format. File does not start with PDF header."
    
    # Basic check for PDF end marker (%%EOF)
    # Note: Some PDFs have trailing content after %%EOF, so we check the last 1024 bytes
    if b'%%EOF' not in file_bytes[-1024:]:
        logger.warning(f"PDF {_redact_pii(filename)} may be incomplete (no %%EOF marker found)")
        # Don't fail validation, just log warning - some PDFs are valid without this
    
    return True, None


def _sanitize_text_for_llm(text: str, max_length: int = 20000) -> str:
    """
    Sanitize text before sending to LLM to prevent prompt injection.
    
    Args:
        text: Raw extracted text
        max_length: Maximum text length to send to LLM
    
    Returns:
        Sanitized text string
    """
    if not text:
        return ""
    
    # Truncate to max length
    sanitized = text[:max_length]
    
    # Remove potential prompt injection patterns (basic heuristic)
    injection_patterns = [
        r'(?i)ignore\s+(?:all\s+)?(?:previous|above)\s+instructions?',
        r'(?i)you\s+are\s+now\s+(?:a|in)\s+(?:new|different)',
        r'(?i)system\s*:\s*you\s+(?:are|will)',
        r'(?i)<\s*/?system\s*>',
    ]
    
    for pattern in injection_patterns:
        sanitized = re.sub(pattern, '[FILTERED]', sanitized)

    return sanitized


def _select_text_for_llm(text: str, max_length: int = 20000, head_ratio: float = 0.6) -> str:
    """
    Pick the document text the LLM actually needs, within a character budget.

    Financial documents put their critical numbers at the END (bank statement
    summary, CAS grand total, ITR tax computation), so instead of a blind
    text[:max_length] we keep the head AND the tail and drop the middle.
    Also runs the prompt-injection filter.
    """
    if not text:
        return ""
    sanitized = _sanitize_text_for_llm(text, max_length=len(text))
    if len(sanitized) <= max_length:
        return sanitized
    marker = "\n\n[... middle of document omitted ...]\n\n"
    head = int((max_length - len(marker)) * head_ratio)
    tail = max_length - len(marker) - head
    return sanitized[:head] + marker + sanitized[-tail:]


def extract_structured_text_with_tables(file_stream):
    """
    Extracts structured text and tables from a PDF file stream using pdfplumber.
    Tables are converted to a markdown-like format.
    """
    logger.debug("Starting structured text extraction...")
    text_content = []
    try:
        # Use the file stream directly with pdfplumber
        with pdfplumber.open(file_stream) as pdf:
            logger.debug(f"PDF has {len(pdf.pages)} pages.")
            for page_num, page in enumerate(pdf.pages):
                text_content.append(f"\n--- Page {page_num + 1} ---\n")
                
                # Extract text first, which is the default behavior of page.extract_text()
                page_text = page.extract_text()
                if page_text:
                    text_content.append(page_text)

                # Extract tables and format them as markdown
                tables = page.extract_tables()
                if tables:
                    text_content.append("\n\n--- Extracted Tables ---\n")
                    for i, table in enumerate(tables):
                        if not table or len(table) == 0: continue
                        
                        text_content.append(f"\n**Table {i+1} on Page {page_num + 1}:**\n")
                        
                        # Filter out None values from header and rows to prevent errors
                        header = [str(h).replace('\n', ' ') if h is not None else '' for h in table[0]]
                        table_md = f"| {' | '.join(header)} |\n"
                        table_md += f"| {' | '.join(['---'] * len(header))} |\n"
                        for row in table[1:]:
                            cleaned_row = [str(cell).replace('\n', ' ') if cell is not None else '' for cell in row]
                            table_md += f"| {' | '.join(cleaned_row)} |\n"
                        text_content.append(table_md)
                        text_content.append("\n")
    
    
        logger.debug("Finished structured text extraction.")
        return "".join(text_content)
    except Exception as e:
        logger.error(f"Error in extract_structured_text_with_tables: {e}")
        raise
