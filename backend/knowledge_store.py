"""Transactional local Markdown storage; derived indexes remain rebuildable."""
from contextlib import contextmanager
import hashlib
import sqlite3
from pathlib import Path
from uuid import uuid4

from backend.document_ingestion import parse_markdown_content


class KnowledgeStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS knowledge_revision (
                    singleton INTEGER PRIMARY KEY CHECK(singleton = 1), revision INTEGER NOT NULL);
                INSERT OR IGNORE INTO knowledge_revision VALUES (1, 0);
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY, filename TEXT NOT NULL UNIQUE,
                    active_version INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS document_versions (
                    document_id TEXT NOT NULL, version INTEGER NOT NULL,
                    content_hash TEXT NOT NULL, markdown TEXT NOT NULL,
                    PRIMARY KEY(document_id, version));
            """)

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.path, timeout=10)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _read(self, connection):
        revision = connection.execute('SELECT revision FROM knowledge_revision').fetchone()[0]
        records = []
        for identifier, filename, version, digest, text in connection.execute("""
            SELECT d.document_id, d.filename, d.active_version, v.content_hash, v.markdown
            FROM documents d JOIN document_versions v
            ON d.document_id = v.document_id AND d.active_version = v.version
            ORDER BY d.rowid
        """):
            record = parse_markdown_content(filename, text)
            record.metadata.update(document_id=identifier, version=version, content_hash=digest)
            records.append(record)
        return revision, records

    def read(self):
        with self.connect() as connection:
            connection.execute('BEGIN')
            return self._read(connection)

    def revision(self):
        with self.connect() as connection:
            return connection.execute('SELECT revision FROM knowledge_revision').fetchone()[0]

    def update(self, documents, prepare, only_missing=False):
        # SQLite serializes writers. Index failure rolls back document/version changes.
        with self.connect() as connection:
            connection.execute('BEGIN IMMEDIATE')
            changed = False
            for document in documents:
                filename = Path(document.name).name
                if not filename or Path(filename).suffix.casefold() != '.md':
                    raise ValueError('Only Markdown (.md) files are supported.')
                digest = hashlib.sha256(document.raw_text.encode('utf-8')).hexdigest()
                existing = connection.execute("""
                    SELECT d.document_id, d.active_version, v.content_hash FROM documents d
                    JOIN document_versions v ON d.document_id=v.document_id
                    AND d.active_version=v.version WHERE filename=?
                """, (filename,)).fetchone()
                if existing and (only_missing or existing[2] == digest):
                    continue
                identifier, version = (existing[0], existing[1] + 1) if existing else (str(uuid4()), 1)
                connection.execute('INSERT INTO document_versions VALUES (?, ?, ?, ?)',
                                   (identifier, version, digest, document.raw_text))
                if existing:
                    connection.execute('UPDATE documents SET active_version=? WHERE document_id=?',
                                       (version, identifier))
                else:
                    connection.execute('INSERT INTO documents VALUES (?, ?, ?)',
                                       (identifier, filename, version))
                changed = True
            if changed:
                connection.execute('UPDATE knowledge_revision SET revision=revision+1')
            revision, records = self._read(connection)
            snapshot = prepare(records, revision)
            return revision, snapshot
