"""Regression: the bounded PDF process must not initialize the backend package."""
import json
from pathlib import Path
import subprocess
import sys

from tests.test_uploads import pdf_bytes, FACT
from backend.upload_parser import parse_upload


def test_pdf_worker_runs_without_backend_or_numpy_imports():
    worker = Path(__file__).resolve().parents[1] / 'backend' / 'pdf_worker.py'
    # A fresh child enforces the import boundary on every OS. On Linux the worker
    # also applies the real RLIMIT_AS limit before importing pypdf.
    code = '''import importlib.abc, runpy, sys
class NoApplicationImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'backend', 'numpy', 'langgraph'}:
            raise ImportError('PDF worker loaded the application: ' + fullname)
sys.meta_path.insert(0, NoApplicationImports())
runpy.run_path(sys.argv[1], run_name='__main__')
'''
    result = subprocess.run([sys.executable, '-c', code, str(worker)], input=pdf_bytes(),
                            capture_output=True, timeout=15, check=True)
    assert FACT in json.loads(result.stdout)['text']


def test_upload_launches_worker_file_directly(monkeypatch):
    run = subprocess.run
    commands = []
    def capture(command, **kwargs):
        commands.append(command)
        return run(command, **kwargs)
    monkeypatch.setattr(subprocess, 'run', capture)
    assert FACT in parse_upload('fact.pdf', pdf_bytes()).raw_text
    assert commands == [[sys.executable, str(Path(__file__).resolve().parents[1] / 'backend' / 'pdf_worker.py')]]
