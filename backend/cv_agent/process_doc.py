import pdfplumber
import docx
import subprocess
import tempfile
import os


def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()


def extract_text_from_docx(docx_path):
    text = ""
    try:
        doc = docx.Document(docx_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        print(f"Error extracting DOCX ({docx_path}): {e}")
    return text.strip()


def extract_text_from_doc(doc_path):
    """
    Extract text from old .doc files using libreoffice conversion.
    Converts .doc → .docx → extract as docx
    """
    try:
        temp_dir = tempfile.mkdtemp()
        converted_path = os.path.join(temp_dir, "converted.docx")

        # Convert using LibreOffice (must be installed on system)
        subprocess.run([
            "libreoffice", "--headless", "--convert-to", "docx",
            "--outdir", temp_dir, doc_path
        ], check=True)

        return extract_text_from_docx(converted_path)

    except Exception as e:
        print(f"Error converting DOC file ({doc_path}): {e}")
        return ""
