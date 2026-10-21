from io import BytesIO
import subprocess

import pytest
from fastapi.testclient import TestClient
from pypdf import PdfWriter
from pypdf.generic import NameObject, DictionaryObject, DecodedStreamObject

from backend import api
from backend.knowledge_base import create_shared_knowledge_base
from backend.upload_parser import parse_upload, UploadError, MAX_FILE_BYTES

FACT = 'Violet Beacon uses the Zircon Thread protocol to replicate the archive every 19 minutes.'


def pdf_bytes(text=FACT, encrypted=False):
    writer = PdfWriter()
    page = writer.add_blank_page(width=600, height=800)
    font = DictionaryObject({NameObject('/Type'): NameObject('/Font'), NameObject('/Subtype'): NameObject('/Type1'), NameObject('/BaseFont'): NameObject('/Helvetica')})
    page[NameObject('/Resources')] = DictionaryObject({NameObject('/Font'): DictionaryObject({NameObject('/F1'): writer._add_object(font)})})
    stream = DecodedStreamObject()
    stream.set_data(f'BT /F1 12 Tf 30 700 Td ({text}) Tj ET'.encode('ascii'))
    page[NameObject('/Contents')] = writer._add_object(stream)
    if encrypted:
        writer.encrypt('secret')
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


@pytest.fixture
def library(tmp_path, monkeypatch):
    monkeypatch.setenv('KNOWLEDGE_STORE_DIR', str(tmp_path))
    monkeypatch.setenv('KNOWLEDGE_EMBEDDING_BACKEND', 'hash')
    base = create_shared_knowledge_base()
    monkeypatch.setattr(api, 'get_shared_knowledge_base', lambda: base)
    monkeypatch.setattr(api.upload_limiter, '_clients', {})
    with TestClient(api.app) as client:
        yield client, base


def upload(client, name, data):
    return client.post('/documents', params={'filename': name}, content=data, headers={'Content-Type': 'application/octet-stream'})


@pytest.mark.parametrize('extension', ['md', 'txt', 'pdf'])
def test_upload_then_query_without_restart_and_refusal(library, extension):
    client, base = library
    before = {d.name: d.raw_text for d in base.documents}
    data = pdf_bytes() if extension == 'pdf' else FACT.encode()
    name = f'violet.{extension}'
    result = upload(client, name, data)
    assert result.status_code == 201, result.text
    assert result.json()['status'] == 'ready' and result.json()['chunk_count'] > 0
    assert name in {d['filename'] for d in client.get('/documents').json()['documents']}
    answer = client.post('/query', json={'question': 'What protocol does Violet Beacon use?'})
    assert answer.status_code == 200
    body = answer.json()
    assert body['evidence_sufficient'] and name in body['sources'], body
    assert 'Zircon Thread' in body['answer']
    # Existing generator headings may remain outside literal verifier coverage.
    assert body['verification']['status'] in {'SUPPORTED', 'NOT_VERIFIED'}
    assert not body['verification']['blocked']
    assert any(e['source_document'] == name and '::v1::' in e['chunk_id'] for e in body['provenance'])
    refusal = client.post('/query', json={'question': 'What is the price of Violet Beacon in USD?'})
    assert refusal.status_code == 200 and not refusal.json()['evidence_sufficient']
    assert refusal.json()['sources'] == []
    assert {d.name: d.raw_text for d in base.documents if d.name in before} == before
    restarted = create_shared_knowledge_base()
    assert name in {d.name for d in restarted.documents}


@pytest.mark.parametrize('name,data,status', [
    ('bad.docx', b'hello', 422), ('../bad.txt', b'hello', 422),
    ('bad\\name.txt', b'hello', 422), ('bad.txt', b'\xff', 422),
    ('bad.txt', b'\x00binary', 422), ('empty.md', b'# Heading', 422),
    ('empty.txt', b'', 413), ('huge.txt', b'x' * (MAX_FILE_BYTES + 1), 413),
    ('fake.pdf', b'not a pdf', 422), ('broken.pdf', b'%PDF-1.7\ninvalid', 422),
    ('ADR_Deployment.md', b'# Changed\nNo.', 409),
], ids=['extension','traversal','separator','encoding','binary','heading-only','empty','oversize','fake-pdf','broken-pdf','sample'])
def test_invalid_upload_never_changes_store(library, name, data, status):
    client, base = library
    revision = base.revision
    result = upload(client, name, data)
    assert result.status_code == status, result.text
    assert result.headers.get('x-request-id')
    assert base.revision == revision and len(base.documents) == 8


def test_pdf_encryption_and_no_readable_text():
    with pytest.raises(UploadError, match='Encrypted'):
        parse_upload('encrypted.pdf', pdf_bytes(encrypted=True))
    writer = PdfWriter(); writer.add_blank_page(width=100, height=100)
    output = BytesIO(); writer.write(output)
    with pytest.raises(UploadError, match='No readable text'):
        parse_upload('scan.pdf', output.getvalue())


def test_pdf_timeout_is_actionable(monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired('parser', 15)
    monkeypatch.setattr(subprocess, 'run', timeout)
    with pytest.raises(UploadError, match='timed out'):
        parse_upload('slow.pdf', b'%PDF-1.7')


def test_noop_collision_and_failed_index_rollback(library, monkeypatch):
    client, base = library
    assert upload(client, 'fact.txt', FACT.encode()).status_code == 201
    revision = base.revision
    assert upload(client, 'fact.txt', FACT.encode()).status_code == 201
    assert base.revision == revision
    assert upload(client, 'FACT.txt', b'Different.').status_code == 409
    def fail(*args):
        raise RuntimeError('private parser details')
    monkeypatch.setattr(base, '_prepare', fail)
    result = upload(client, 'another.txt', b'Another fact.')
    assert result.status_code == 503 and 'private' not in result.text
    assert base.revision == revision
    assert 'another.txt' not in {d.name for d in create_shared_knowledge_base().documents}


def test_preamble_and_heading_free_text():
    from backend.document_ingestion import chunk_documents
    document = parse_upload('notes.md', b'Preamble fact.\n# Heading\nBody fact.')
    assert [c.text for c in chunk_documents([document])] == ['Preamble fact.', 'Body fact.']


def test_limits_and_content_type(library, monkeypatch):
    client, base = library
    result = client.post('/documents?filename=fact.txt', content=b'Fact.', headers={'Content-Type': 'image/png'})
    assert result.status_code == 415
    result = upload(client, 'large.txt', ('é' * 100001).encode())
    assert result.status_code == 413
    api.upload_lock.acquire()
    try:
        result = upload(client, 'fact.txt', b'Fact.')
        assert result.status_code == 429 and result.headers['retry-after'] == '5'
    finally:
        api.upload_lock.release()
    monkeypatch.setattr(api.upload_limiter, 'limit', 1)
    api.upload_limiter._clients.clear()
    assert upload(client, 'fact.txt', b'Fact.').status_code == 201
    assert upload(client, 'other.txt', b'Other fact.').status_code == 429
    assert len(base.documents) == 9


def test_capacity_rejects_without_mutating(library, monkeypatch):
    from backend import upload_parser
    client, base = library
    monkeypatch.setattr(upload_parser, 'MAX_DOCUMENTS', 8)
    revision = base.revision
    assert upload(client, 'extra.txt', b'New fact.').status_code == 409
    assert base.revision == revision


def test_pdf_page_limit():
    writer = PdfWriter()
    for _ in range(21):
        writer.add_blank_page(width=100, height=100)
    output = BytesIO(); writer.write(output)
    with pytest.raises(UploadError, match='20 pages'):
        parse_upload('long.pdf', output.getvalue())
