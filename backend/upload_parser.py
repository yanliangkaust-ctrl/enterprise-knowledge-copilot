"""Bounded extraction into the existing document/chunk model. No OCR or binary storage."""
from pathlib import Path
import re
import subprocess
import sys

from backend.document_ingestion import parse_markdown_content, chunk_documents
from backend.pdf_worker import MAX_FILE_BYTES, MAX_TEXT_BYTES

MAX_DOCUMENTS = 28  # eight samples plus twenty uploaded documents
MAX_CORPUS_BYTES = 2_000_000


class UploadError(ValueError):
    def __init__(self, detail, status=422):
        super().__init__(detail)
        self.status = status


def validate_filename(filename):
    if (not filename or len(filename) > 120 or filename != filename.strip()
            or filename.startswith('.') or re.search(r'[\\/:*?"<>|\x00-\x1f\x7f]', filename)
            or Path(filename).suffix.casefold() not in {'.md', '.txt', '.pdf'}):
        raise UploadError('Use a plain filename (maximum 120 characters) ending in .pdf, .txt, or .md.')
    samples = Path(__file__).resolve().parents[1] / 'data' / 'sample_docs'
    if filename.casefold() in {p.name.casefold() for p in samples.glob('*.md')}:
        raise UploadError('Bundled demo documents cannot be replaced. Choose a new filename.', 409)


def parse_upload(filename, data):
    validate_filename(filename)
    if not data or len(data) > MAX_FILE_BYTES:
        raise UploadError('Choose a nonempty file no larger than 1 MiB.', 413)
    if filename.casefold().endswith('.pdf'):
        if not data.startswith(b'%PDF-'):
            raise UploadError('This file is not a PDF. Check its contents and extension.')
        try:
            # -m backend.pdf_worker executes backend/__init__.py first, loading
            # NumPy/LangGraph before the child memory limit. Run the isolated file.
            result = subprocess.run([sys.executable, str(Path(__file__).with_name('pdf_worker.py'))], input=data,
                                    capture_output=True, timeout=15, check=False,
                                    cwd=Path(__file__).resolve().parents[1])
        except subprocess.TimeoutExpired:
            raise UploadError('PDF parsing timed out. Export a simpler PDF or UTF-8 TXT file.') from None
        import json
        if result.returncode:
            raise UploadError('PDF parsing failed. Upload a valid text PDF or UTF-8 TXT file.')
        payload = json.loads(result.stdout)
        if 'error' in payload:
            raise UploadError(payload['error'], payload.get('status', 422))
        text = payload['text']
    else:
        try:
            text = data.decode('utf-8-sig')
        except UnicodeDecodeError:
            raise UploadError('TXT and Markdown files must use UTF-8 encoding.') from None
    if not text.strip():
        raise UploadError('No readable text found. Scanned PDFs require OCR, which is not supported.')
    if re.search(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', text):
        raise UploadError('File contains binary/control data. Upload readable text.')
    if len(text.encode('utf-8')) > MAX_TEXT_BYTES:
        raise UploadError('Extracted text exceeds 200 KB. Split the document.', 413)
    document = parse_markdown_content(filename, text)
    if not chunk_documents([document]):
        raise UploadError('Document has no indexable text. Add content below its headings.')
    return document
