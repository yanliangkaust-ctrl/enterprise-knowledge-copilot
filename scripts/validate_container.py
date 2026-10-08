"""Validate a prebuilt release image using disposable Docker resources only."""
import argparse
import json
from pathlib import Path
import subprocess
import shutil
import time
import urllib.error
import urllib.request
from uuid import UUID, uuid4


def docker(*args, check=True):
    result = subprocess.run(["docker", *args], capture_output=True, text=True, timeout=120)
    if check and result.returncode:
        raise RuntimeError("docker_command_failed:" + args[0])
    return result.stdout.strip()


def request(url, question=None):
    data = None if question is None else json.dumps({"question": question}).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        response = urllib.request.urlopen(req, timeout=10)
    except urllib.error.HTTPError as exc:
        response = exc
    with response:
        body = response.read().decode()
        return response.status, dict(response.headers), json.loads(body) if body.startswith("{") else body


def wait_ready(name):
    port = docker("inspect", "--format", '{{(index (index .NetworkSettings.Ports "8000/tcp") 0).HostPort}}', name)
    url = "http://127.0.0.1:" + port
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        try:
            if request(url + "/ready")[0] == 200:
                return url
        except (OSError, TimeoutError):
            pass
        time.sleep(1)
    raise RuntimeError("container_readiness_timeout")


UPDATE = '''import json
from backend.knowledge_base import create_shared_knowledge_base
k=create_shared_knowledge_base()
k.ingest_markdown("release_probe.md", "# Release probe\\nQuartz Relay uses Amber Mesh protocol.")
k.ingest_markdown("release_probe.md", "# Release probe\\nQuartz Relay uses Copper Mesh protocol.")
d=next(d for d in k.documents if d.name=="release_probe.md")
print(json.dumps({"revision":k.revision,"id":d.metadata["document_id"],"version":d.metadata["version"]}))
'''
STATE = '''import json
from backend.knowledge_base import create_shared_knowledge_base
k=create_shared_knowledge_base()
d=next(d for d in k.documents if d.name=="release_probe.md")
print(json.dumps({"revision":k.revision,"id":d.metadata["document_id"],"version":d.metadata["version"]}))
'''
DEMO_FILES = {
    "ADR_Deployment.md", "API_Specification.md", "Deployment_Guide.md",
    "Incident_Batch_Failure.md", "OCR_Platform_Architecture.md", "Risk_Register.md",
    "Security_Requirements.md", "UAT_Report.md",
}
CORPUS = '''import json,hashlib
from backend.knowledge_base import create_shared_knowledge_base
k=create_shared_knowledge_base()
print(json.dumps({d.name:{"hash":hashlib.sha256(d.raw_text.encode()).hexdigest(),"version":d.metadata["version"]} for d in k.documents}))
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default="artifacts/container-validation")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    token = "release-" + uuid4().hex
    volume = token + "-knowledge"
    names = []
    report = {"image": args.image, "passed": False, "checks": {}}
    ids = set()

    def checked_request(url, question=None):
        status, headers, body = request(url, question)
        identifier = next(v for k, v in headers.items() if k.lower() == "x-request-id")
        UUID(identifier)
        assert identifier not in ids
        ids.add(identifier)
        return status, body

    def start(suffix, readonly=False, ephemeral=False):
        name = token + suffix
        names.append(name)
        options = ["run", "-d", "--name", name, "-p", "127.0.0.1::8000",
                   "-e", "KNOWLEDGE_STORE_DIR=/knowledge", "-e", "KNOWLEDGE_EMBEDDING_BACKEND=hash"]
        options += ["--read-only"] if readonly else [] if ephemeral else ["--mount", "type=volume,src=" + volume + ",dst=/knowledge"]
        docker(*options, args.image)
        return name

    def stop(name):
        docker("stop", "--time", "30", name)
        assert docker("inspect", "--format", "{{.State.ExitCode}}", name) == "0"
        assert "Application shutdown complete" in docker("logs", name) or "Application shutdown complete" in subprocess.run(
            ["docker", "logs", name], capture_output=True, text=True).stderr

    try:
        docker("version")
        report["image_id"] = docker("image", "inspect", "--format", "{{.Id}}", args.image)
        docker("volume", "create", volume)
        name = start("-first")
        url = wait_ready(name)
        assert checked_request(url + "/health")[1] == {"status": "ok", "service": "enterprise-knowledge-copilot"}
        assert checked_request(url + "/ready")[0] == 200
        status, answer = checked_request(url + "/query", "What depends on Kubernetes?")
        assert status == 200 and answer["evidence_sufficient"]
        status, refusal = checked_request(url + "/query", "What is the OCR Service license price?")
        assert status == 200 and not refusal["evidence_sufficient"]
        report["checks"]["health_ready_supported_refusal_ids"] = True
        expected = json.loads(docker("exec", name, "python", "-c", UPDATE))
        assert expected["version"] == 2
        assert checked_request(url + "/query", "What protocol does Quartz Relay use?")[1]["answer"].find("Copper Mesh") >= 0
        # Make only this disposable database inaccessible, then restore it.
        docker("exec", name, "python", "-c", "import os; os.rename('/knowledge/knowledge.sqlite3','/knowledge/saved.sqlite3'); os.mkdir('/knowledge/knowledge.sqlite3')")
        assert checked_request(url + "/query", "What depends on Kubernetes?")[0] == 503
        assert checked_request(url + "/ready")[0] == 503
        docker("exec", name, "python", "-c", "import os; os.rmdir('/knowledge/knowledge.sqlite3'); os.rename('/knowledge/saved.sqlite3','/knowledge/knowledge.sqlite3')")
        assert checked_request(url + "/query", "What depends on Kubernetes?")[0] == 200
        report["checks"]["sanitized_503_recovery"] = True
        stop(name)
        docker("start", name)
        url = wait_ready(name)
        assert json.loads(docker("exec", name, "python", "-c", STATE)) == expected
        stop(name)
        replacement = start("-replacement")
        url = wait_ready(replacement)
        assert json.loads(docker("exec", replacement, "python", "-c", STATE)) == expected
        assert "Copper Mesh" in checked_request(url + "/query", "What protocol does Quartz Relay use?")[1]["answer"]
        report["persisted_state"] = expected
        report["checks"]["restart_replacement_persistence"] = True
        stop(replacement)
        # Free mode: every NEW container has independent writable ephemeral storage.
        fresh = start("-free-first", ephemeral=True)
        free_url = wait_ready(fresh)
        original_corpus = json.loads(docker("exec", fresh, "python", "-c", CORPUS))
        assert set(original_corpus) == DEMO_FILES
        assert all(d['version'] == 1 for d in original_corpus.values())
        docker("exec", fresh, "python", "-c", UPDATE)
        stop(fresh)
        free_logs = subprocess.run(["docker", "logs", fresh], capture_output=True, text=True, timeout=30)
        (output / (fresh + ".log")).write_text(free_logs.stdout + free_logs.stderr, encoding="utf-8")
        docker("rm", fresh)
        names.remove(fresh)
        rebuilt = start("-free-new", ephemeral=True)
        free_url = wait_ready(rebuilt)
        assert json.loads(docker("exec", rebuilt, "python", "-c", CORPUS)) == original_corpus
        assert checked_request(free_url + "/ready")[1]["document_count"] == 8
        assert checked_request(free_url + "/query", "What depends on Kubernetes?")[1]["evidence_sufficient"]
        report["checks"]["free_eight_document_reconstruction"] = True
        stop(rebuilt)
        bad = start("-unwritable", readonly=True)
        assert docker("wait", bad) != "0"
        report["checks"]["unwritable_startup_fails"] = True
        events = []
        for name in names:
            result = subprocess.run(["docker", "logs", name], capture_output=True, text=True, timeout=30)
            (output / (name + ".log")).write_text(result.stdout + result.stderr, encoding="utf-8")
            for line in result.stdout.splitlines():
                if line.startswith("{"):
                    event = json.loads(line)
                    assert {"event", "timestamp", "request_id", "stage", "status"} <= event.keys()
                    assert not {"question", "answer", "text", "provenance", "body"} & event.keys()
                    events.append(event)
        serialized = json.dumps(events)
        assert "Copper Mesh" not in serialized and "license price" not in serialized
        assert any(e["event"] == "application_error" and e.get("http_status") == 503 for e in events)
        report["checks"]["structured_sanitized_logs_shutdown"] = True
        report["passed"] = True
    except Exception as exc:
        report["error_type"] = type(exc).__name__
    finally:
        for name in names:
            result = subprocess.run(["docker", "logs", name], capture_output=True, text=True, timeout=30) if shutil.which('docker') else None
            if result:
                (output / (name + ".log")).write_text(result.stdout + result.stderr, encoding="utf-8")
                docker("rm", "-f", name, check=False)
        if shutil.which('docker'):
            docker("volume", "rm", volume, check=False)
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
