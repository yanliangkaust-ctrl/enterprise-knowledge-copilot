"""Isolated PDF extraction entry point; stdout contains only a bounded JSON result."""
import json
import sys
from io import BytesIO

MAX_FILE_BYTES = 1024 * 1024
MAX_TEXT_BYTES = 200_000


class PDFValidationError(ValueError):
    def __init__(self, detail, status=422):
        super().__init__(detail)
        self.status = status


def extract_pdf(data):
    """Only pypdf is imported in the resource-bounded child, never the RAG app."""
    from pypdf import PdfReader
    reader = PdfReader(BytesIO(data), strict=True)
    if reader.is_encrypted:
        raise PDFValidationError('Encrypted PDFs are unsupported. Upload an unencrypted text PDF.')
    if not 1 <= len(reader.pages) <= 20:
        raise PDFValidationError('PDFs must contain between 1 and 20 pages.')
    parts = []
    for page in reader.pages:
        contents = page.get_contents()
        if contents is not None and len(contents.get_data()) > 2_000_000:
            raise PDFValidationError('PDF page content is too complex. Export a smaller text PDF or TXT file.')
        parts.append(page.extract_text() or '')
        if len('\n'.join(parts).encode('utf-8')) > MAX_TEXT_BYTES:
            raise PDFValidationError('Extracted text exceeds 200 KB. Split the document.', 413)
    return '\n\n'.join(parts)

if __name__ == '__main__':
    # Render/Linux: bound worker address space in addition to the parent's timeout.
    if sys.platform != 'win32':
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (256 * 1024 * 1024,) * 2)
    try:
        text = extract_pdf(sys.stdin.buffer.read(MAX_FILE_BYTES + 1))
        payload = {'text': text}
    except PDFValidationError as exc:
        payload = {'error': str(exc), 'status': exc.status}
    except Exception:
        payload = {'error': 'PDF parsing failed. Upload a valid text PDF or UTF-8 TXT file.'}
    sys.stdout.buffer.write(json.dumps(payload).encode('utf-8'))
