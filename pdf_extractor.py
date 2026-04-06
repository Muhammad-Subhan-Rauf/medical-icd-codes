"""
PDF text extraction with page-level source tracking.
Supports multi-file upload from Streamlit.
Falls back to Gemini Vision OCR for scanned/image PDFs.
"""

import os
import io
import base64
import logging
from dataclasses import dataclass

import fitz  # PyMuPDF

logger = logging.getLogger("medic.pdf_extractor")


@dataclass
class PageChunk:
    text: str
    source_file: str
    page_number: int  # 1-based


def _ocr_page_with_gemini(page, page_num: int, filename: str) -> str | None:
    """Render a PDF page to image and use Gemini Vision to extract text."""
    try:
        from google import genai

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return None

        client = genai.Client(api_key=api_key)

        # Render page to PNG image at 200 DPI
        pix = page.get_pixmap(dpi=200)
        img_bytes = pix.tobytes("png")
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                {
                    "parts": [
                        {"text": "Extract ALL text from this document image exactly as written. Preserve the original structure, headings, and formatting. Return ONLY the extracted text, nothing else."},
                        {"inline_data": {"mime_type": "image/png", "data": img_b64}},
                    ]
                }
            ],
            config={"temperature": 0.0},
        )
        text = response.text.strip()
        if text:
            logger.info("OCR extracted %d chars from %s p%d via Gemini Vision", len(text), filename, page_num)
            return text
        return None
    except Exception as e:
        logger.error("Gemini Vision OCR failed for %s p%d: %s", filename, page_num, e)
        return None


def extract_text_from_pdf(file_bytes: bytes, filename: str) -> list[PageChunk]:
    """Extract text from a PDF, returning one PageChunk per page.
    Falls back to Gemini Vision OCR for scanned/image pages."""
    chunks = []
    scanned_pages = []
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            if text.strip():
                chunks.append(PageChunk(
                    text=text.strip(),
                    source_file=filename,
                    page_number=page_num + 1,
                ))
            else:
                scanned_pages.append((page, page_num + 1))

        # OCR scanned pages via Gemini Vision
        if scanned_pages:
            logger.info(
                "%d of %d pages in %s are scanned/image. Running Gemini Vision OCR...",
                len(scanned_pages), len(doc), filename
            )
            for page, page_num in scanned_pages:
                ocr_text = _ocr_page_with_gemini(page, page_num, filename)
                if ocr_text:
                    chunks.append(PageChunk(
                        text=ocr_text,
                        source_file=filename,
                        page_number=page_num,
                    ))
                else:
                    logger.warning("Could not extract text from %s p%d", filename, page_num)

        # Sort by page number (OCR pages may have been appended out of order)
        chunks.sort(key=lambda c: c.page_number)
        doc.close()
    except Exception as e:
        logger.error("Failed to extract text from %s: %s", filename, e)
    return chunks


def merge_chunks_to_note(chunks: list[PageChunk]) -> str:
    """
    Merge page chunks into a single note string with embedded source markers.
    Markers like ---[filename p2]--- allow the LLM to cite specific pages.
    """
    parts = []
    for chunk in chunks:
        marker = f"---[{chunk.source_file} p{chunk.page_number}]---"
        parts.append(f"{marker}\n{chunk.text}")
    return "\n\n".join(parts)


def extract_text_from_multiple_pdfs(uploaded_files) -> tuple[str, list[PageChunk]]:
    """
    Process multiple Streamlit UploadedFile objects.
    Returns (merged_note_text, all_page_chunks).
    """
    all_chunks = []
    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.read()
        filename = uploaded_file.name
        chunks = extract_text_from_pdf(file_bytes, filename)
        all_chunks.extend(chunks)
        logger.info("Extracted %d pages from %s", len(chunks), filename)

    merged = merge_chunks_to_note(all_chunks) if all_chunks else ""
    return merged, all_chunks


def find_source_for_quote(quote: str, chunks: list[PageChunk]) -> str | None:
    """
    Given an evidence quote, find which page chunk contains it.
    Returns a reference like 'filename.pdf p2' or None.
    """
    if not quote:
        return None
    # Try exact substring match first
    for chunk in chunks:
        if quote in chunk.text:
            return f"{chunk.source_file} p{chunk.page_number}"
    # Fallback: try case-insensitive partial match (first 60 chars)
    query = quote[:60].lower()
    for chunk in chunks:
        if query in chunk.text.lower():
            return f"{chunk.source_file} p{chunk.page_number}"
    return None
