"""Isolated PDF extraction entry point; stdout contains only a bounded JSON result."""
import json
import sys

if __name__ == '__main__':
    # Render/Linux: bound worker address space in addition to the parent's timeout.
    if sys.platform != 'win32':
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (256 * 1024 * 1024,) * 2)
    from backend.upload_parser import extract_pdf, UploadError, MAX_FILE_BYTES
    try:
        text = extract_pdf(sys.stdin.buffer.read(MAX_FILE_BYTES + 1))
        payload = {'text': text}
    except UploadError as exc:
        payload = {'error': str(exc), 'status': exc.status}
    except Exception:
        payload = {'error': 'PDF parsing failed. Upload a valid text PDF or UTF-8 TXT file.'}
    sys.stdout.buffer.write(json.dumps(payload).encode('utf-8'))
