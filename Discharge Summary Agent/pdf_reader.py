# pdf_reader.py
# Handles reading PDFs and extracting their text.
# For scanned / handwritten PDFs (like these patient records)
# standard text extraction returns almost nothing, so we fall
# back to OCR (Optical Character Recognition) automatically.

import os
from typing import Dict, List

# ── OCR imports ───────────────────────────────────────────────
try:
    import pytesseract
    from pdf2image import convert_from_path
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("⚠️  OCR packages missing. Run: pip install pytesseract pdf2image pillow")

# ── Standard PDF text extraction ─────────────────────────────
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    print("⚠️  PyMuPDF missing. Run: pip install pymupdf")

# ─────────────────────────────────────────────────────────────
# !! UPDATE THESE TWO PATHS TO MATCH YOUR SYSTEM !!
# ─────────────────────────────────────────────────────────────

TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
POPPLER_PATH   = r"C:\poppler-26.02.0\Library\bin"

# ─────────────────────────────────────────────────────────────


def setup_tesseract() -> bool:
    """Point pytesseract at the Tesseract executable."""
    if not OCR_AVAILABLE:
        return False
    if os.path.exists(TESSERACT_PATH):
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
        return True
    print(f"⚠️  Tesseract not found at: {TESSERACT_PATH}")
    print("    Download from: https://github.com/UB-Mannheim/tesseract/wiki")
    return False


def extract_text_with_ocr(pdf_path: str) -> str:
    """
    Converts each PDF page to an image then runs Tesseract OCR on it.
    Works on scanned and handwritten documents.
    dpi=250 gives a good balance of accuracy vs speed.
    """
    print(f"  Using OCR: {os.path.basename(pdf_path)}")

    try:
        poppler_kwargs = {}
        if os.path.exists(POPPLER_PATH):
            poppler_kwargs["poppler_path"] = POPPLER_PATH

        pages = convert_from_path(pdf_path, dpi=250, **poppler_kwargs)
        print(f"  Converting {len(pages)} pages to images...")

        full_text = ""
        for i, page_image in enumerate(pages):
            print(f"  OCR page {i + 1}/{len(pages)}...", end="\r")

            # --psm 6  = assume a single uniform block of text
            # --oem 3  = use the best available LSTM engine
            page_text = pytesseract.image_to_string(
                page_image,
                config="--psm 6 --oem 3"
            )
            full_text += f"\n\n--- PAGE {i + 1} ---\n{page_text}"

        print(f"\n  OCR complete: {len(full_text)} characters extracted")
        return full_text

    except Exception as e:
        print(f"\n  OCR failed: {e}")
        return f"OCR_FAILED: {str(e)}"


def extract_text_standard(pdf_path: str) -> str:
    """
    Fast text extraction using PyMuPDF.
    Only works when the PDF contains real selectable text
    (not scanned images).
    """
    if not PYMUPDF_AVAILABLE:
        return ""
    try:
        doc = fitz.open(pdf_path)
        full_text = ""
        for page_num in range(len(doc)):
            page_text = doc[page_num].get_text()
            full_text += f"\n\n--- PAGE {page_num + 1} ---\n{page_text}"
        doc.close()
        return full_text
    except Exception as e:
        return f"STANDARD_EXTRACTION_FAILED: {str(e)}"


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Smart extraction:
      1. Try fast standard text extraction first.
      2. If the extracted text is too short (i.e. the PDF is
         mostly images / scans), fall back to OCR automatically.
    """
    print(f"  Extracting: {os.path.basename(pdf_path)}")

    standard_text = extract_text_standard(pdf_path)

    # Count the pages so we can judge whether we got enough text
    num_pages = 1
    if PYMUPDF_AVAILABLE:
        try:
            doc = fitz.open(pdf_path)
            num_pages = len(doc)
            doc.close()
        except Exception:
            pass

    # Expect at least 500 non-whitespace chars per page for real text
    actual_chars = len(standard_text.replace('\n', '').replace(' ', ''))
    min_expected  = num_pages * 500

    print(
        f"  Pages: {num_pages} | "
        f"Standard extraction: {len(standard_text)} chars | "
        f"Threshold: {min_expected} chars"
    )

    if actual_chars >= min_expected:
        print("  ✅ Standard text extraction sufficient")
        return standard_text

    # Not enough text → scanned / handwritten PDF → use OCR
    print("  📷 Low text content — switching to OCR (this takes a few minutes)")

    if not OCR_AVAILABLE:
        print("  ❌ OCR packages not installed!")
        return standard_text

    if not setup_tesseract():
        print("  ❌ Tesseract not configured — returning partial text")
        return standard_text

    return extract_text_with_ocr(pdf_path)


def extract_pdfs_from_folder(folder_path: str) -> Dict[str, str]:
    """
    Reads every PDF in a folder and returns a dict:
      { filename: extracted_text }
    """
    if not os.path.exists(folder_path):
        print(f"ERROR: Folder '{folder_path}' does not exist")
        return {}

    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]

    if not pdf_files:
        print(f"WARNING: No PDF files found in '{folder_path}'")
        return {}

    print(f"\nFound {len(pdf_files)} PDF file(s):")
    pdf_texts = {}

    for filename in pdf_files:
        print(f"\n  Processing: {filename}")
        pdf_path = os.path.join(folder_path, filename)
        pdf_texts[filename] = extract_text_from_pdf(pdf_path)
        print(f"  Total: {len(pdf_texts[filename])} characters extracted")

    return pdf_texts


def get_relevant_chunk(full_text: str, keywords: List[str],
                       chunk_size: int = 4000) -> str:
    """
    Returns up to chunk_size characters of text that contains
    the given keywords. Used to keep prompts within the LLM
    context window.
    """
    lines = full_text.split('\n')
    relevant_parts = []

    for i in range(len(lines)):
        window = lines[max(0, i - 2): i + 15]
        window_text = '\n'.join(window)
        if any(kw.lower() in window_text.lower() for kw in keywords):
            relevant_parts.append(window_text)

    if not relevant_parts:
        return full_text[:chunk_size]

    return '\n\n'.join(relevant_parts)[:chunk_size]


def categorize_documents(pdf_texts: Dict[str, str]) -> Dict[str, str]:
    """
    Guesses the document type for each PDF based on keywords
    found in the extracted text. Used for logging only.
    """
    categories = {}

    type_keywords = {
        "admission_note":    ["admission record", "chief complaints",
                              "history of present illness", "past history"],
        "lab_results":       ["haemoglobin", "total count", "creatinine",
                              "biochemistry report", "haematology report"],
        "drug_chart":        ["drug chart", "medication", "dose", "route",
                              "frequency", "regular prescription"],
        "nursing_notes":     ["nursing documentation", "nurses notes",
                              "nursing assessment"],
        "discharge_summary": ["discharge", "condition at discharge",
                              "advice on discharge", "follow-up"],
        "consultation":      ["consultation sheet", "care reviewed", "opinion"],
        "investigations":    ["investigations", "usg", "echo", "ecg",
                              "ct scan", "x-ray"],
        "icu_chart":         ["icu-chart", "icu chart", "care plan"],
        "er_chart":          ["er observation", "triage", "casualty"],
    }

    for filename, text in pdf_texts.items():
        text_lower = text.lower()
        best_match, best_score = "unknown", 0
        for doc_type, keywords in type_keywords.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > best_score:
                best_score = score
                best_match = doc_type
        categories[filename] = best_match
        print(f"  '{filename}' → {best_match} (score: {best_score})")

    return categories
