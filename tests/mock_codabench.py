"""A local stand-in for Codabench's API (v1.33's behaviour on the endpoints the upload helper uses), for the tests.

It follows Codabench's own code where it matters to the client: token authentication (``Authorization: Token <key>``,
401 otherwise), the password exchange of ``/api/api-token-auth/``, the competition's phases with their status and the
caller's ``participant_status``, the data object that returns a storage URL, a storage PUT that refuses Codabench
credentials and any Content-Type but ``application/zip`` (a presigned S3 or GCS URL fails its signature otherwise),
the submission checks in Codabench's order (approved 403, phase open 400, daily and total limits 400, not counting
Failed submissions) and a status that walks Submitted, Running, Scoring, Finished with its scores.
"""

from __future__ import annotations

import json
import re
import threading
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


TOKEN = "0123456789abcdef0123456789abcdef01234567"
USERNAME, PASSWORD = "participant", "correct horse"
COMPETITION = 4242
PHASE_DEV, PHASE_FINAL, TASK_DEV = 71, 72, 81


@dataclass
class State:
    approved: str | None = "approved"  # the caller's participant_status
    dev_status: str = "Current"
    max_per_day: int = 3
    fail_next: bool = False  # the next submission ends Failed
    datasets: dict = field(default_factory=dict)  # key -> {"size", "completed", "blob"}
    submissions: dict = field(default_factory=dict)  # id -> dict
    requests: list = field(default_factory=list)  # (method, path, headers) of every request
    next_id: int = 1000


def competition_doc(state: State) -> dict:
    def phase(pid, index, name, status, task):
        used = sum(1 for s in state.submissions.values() if s["phase"] == pid and s["status"] != "Failed")
        return {
            "id": pid,
            "index": index,
            "name": name,
            "status": status,
            "start": "2026-10-02T15:30:00Z",
            "end": None,
            "tasks": [{"id": task, "name": f"{name} task"}],
            "max_submissions_per_day": state.max_per_day,
            "max_submissions_per_person": 100,
            "used_submissions_per_day": used,
            "used_submissions_per_person": used,
            "hide_output": name == "Final",
            "hide_score_output": name == "Final",
        }

    return {
        "id": COMPETITION,
        "title": "mock competition",
        "participant_status": state.approved,
        "phases": [
            phase(PHASE_DEV, 0, "Development", state.dev_status, TASK_DEV),
            phase(PHASE_FINAL, 1, "Final", "Next", TASK_DEV + 1),
        ],
    }


class Handler(BaseHTTPRequestHandler):
    state: State  # set on the class by serve()

    def log_message(self, *args) -> None:  # quiet
        return

    def _send(self, status: int, body: object) -> None:
        raw = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _body(self) -> bytes:
        return self.rfile.read(int(self.headers.get("Content-Length") or 0))

    def _authed(self) -> bool:
        auth = self.headers.get("Authorization", "")
        parts = auth.split()
        if len(parts) == 2 and parts[0].lower() == "token" and parts[1] == TOKEN:
            return True
        detail = "Invalid token." if auth else "Authentication credentials were not provided."
        self._send(401, {"detail": detail})
        return False

    def _route(self, method: str) -> None:
        st = self.state
        path = self.path
        st.requests.append((method, path, dict(self.headers)))
        body = self._body() if method in ("POST", "PUT") else b""
        if method == "PUT" and path.startswith("/storage/"):
            return self._storage(path.split("/")[2], body)
        if method == "POST" and path == "/api/api-token-auth/":
            data = json.loads(body or b"{}")
            if not data.get("username") or not data.get("password"):
                return self._send(400, {"non_field_errors": ['Must include "username" and "password".']})
            if (data["username"], data["password"]) != (USERNAME, PASSWORD):
                return self._send(400, {"non_field_errors": ["Unable to log in with provided credentials."]})
            return self._send(200, {"token": TOKEN})
        m = re.fullmatch(r"/api/competitions/(\d+)/(\?.*)?", path)
        if method == "GET" and m:
            if int(m.group(1)) != COMPETITION:
                return self._send(404, {"detail": "Not found."})
            return self._send(200, competition_doc(st))
        if not self._authed():
            return None
        m = re.fullmatch(r"/api/can_make_submission/(\d+)/", path)
        if method == "GET" and m:
            if st.approved != "approved":
                return self._send(200, {"can": False, "reason": "User not approved to participate in this competition"})
            if self._used_today(int(m.group(1))) >= st.max_per_day:
                reason = "Reached maximum allowed submissions for today for this phase"
                return self._send(200, {"can": False, "reason": reason})
            return self._send(200, {"can": True, "reason": None})
        if method == "POST" and path == "/api/datasets/":
            return self._dataset(json.loads(body))
        m = re.fullmatch(r"/api/datasets/completed/([0-9a-f-]+)/", path)
        if method == "PUT" and m:
            if m.group(1) not in st.datasets:
                return self._send(404, {"detail": "Not found."})
            st.datasets[m.group(1)]["completed"] = True
            return self._send(200, {"key": m.group(1)})
        if method == "POST" and path == "/api/submissions/":
            return self._submit(json.loads(body))
        m = re.fullmatch(r"/api/submissions/(\d+)/", path)
        if method == "GET" and m:
            return self._poll(int(m.group(1)))
        m = re.fullmatch(r"/api/submissions/\?phase=(\d+)", path)
        if method == "GET" and m:
            rows = [s for s in st.submissions.values() if s["phase"] == int(m.group(1))]
            return self._send(
                200, {"count": len(rows), "next": None, "previous": None, "results": rows, "page_size": 50}
            )
        return self._send(404, {"detail": "Not found."})

    def _used_today(self, phase_id: int) -> int:
        return sum(1 for s in self.state.submissions.values() if s["phase"] == phase_id and s["status"] != "Failed")

    def _dataset(self, data: dict) -> None:
        if "file_size" not in data:
            return self._send(400, {"file_size": "This field is required."})
        if not isinstance(data["file_size"], (int, float)):
            return self._send(400, {"file_size": ["A valid number is required."]})
        if not str(data.get("request_sassy_file_name", "")).endswith(".zip"):
            return self._send(400, {"non_field_errors": ["Only zip files are allowed!"]})
        key = str(uuid.uuid4())
        self.state.datasets[key] = {"size": data["file_size"], "completed": False, "blob": None}
        host = f"http://{self.server.server_address[0]}:{self.server.server_address[1]}"
        return self._send(201, {"key": key, "sassy_url": f"{host}/storage/{key}"})

    def _storage(self, key: str, body: bytes) -> None:
        if "Authorization" in self.headers:
            return self._send(403, {"Code": "SignatureDoesNotMatch", "Message": "unexpected Authorization header"})
        if self.headers.get("Content-Type") != "application/zip":
            return self._send(403, {"Code": "SignatureDoesNotMatch", "Message": "Content-Type is signed"})
        if key not in self.state.datasets:
            return self._send(404, {"Code": "NoSuchUpload"})
        self.state.datasets[key]["blob"] = body
        return self._send(200, {})

    def _submit(self, data: dict) -> None:
        st = self.state
        if st.approved != "approved":
            return self._send(403, {"detail": "You do not have access to this competition to make a submission"})
        phase = data.get("phase")
        if phase not in (PHASE_DEV, PHASE_FINAL):
            return self._send(400, {"phase": ["Invalid pk - object does not exist."]})
        if phase != PHASE_DEV or st.dev_status != "Current":
            return self._send(400, {"non_field_errors": ["This phase is not currently accepting submissions."]})
        if self._used_today(phase) >= st.max_per_day:
            reason = "Reached maximum allowed submissions for today for this phase"
            return self._send(400, {"non_field_errors": [reason]})
        if data.get("tasks") and any(t != TASK_DEV for t in data["tasks"]):
            return self._send(400, {"non_field_errors": ["All tasks must be part of the current phase."]})
        ds = st.datasets.get(data.get("data"))
        if ds is None:
            return self._send(400, {"data": ["Object with key does not exist."]})
        st.next_id += 1
        sub = {
            "id": st.next_id,
            "status": "Submitting",
            "phase": phase,
            "data": data["data"],
            "filename": "submission.zip",
            "scores": [],
            "status_details": None,
            "has_children": False,
            "_polls": 0,
            "_fail": st.fail_next,
        }
        st.fail_next = False
        st.submissions[sub["id"]] = sub
        return self._send(201, {k: v for k, v in sub.items() if not k.startswith("_")})

    def _poll(self, sub_id: int) -> None:
        sub = self.state.submissions.get(sub_id)
        if sub is None:
            return self._send(404, {"detail": "Not found."})
        walk = ["Submitted", "Running", "Scoring", "Failed" if sub["_fail"] else "Finished"]
        sub["status"] = walk[min(sub["_polls"], len(walk) - 1)]
        sub["_polls"] += 1
        if sub["status"] == "Finished":
            sub["scores"] = [
                {"id": 1, "index": 0, "score": "0.4281000000", "column_key": "rss", "precision": 4, "is_primary": True},
                {"id": 2, "index": 1, "score": "0E-10", "column_key": "fallback_count", "precision": 0},
            ]
        if sub["status"] == "Failed":
            sub["status_details"] = "ShockBench-Flow: the ingestion stopped"
        return self._send(200, {k: v for k, v in sub.items() if not k.startswith("_")})

    def do_GET(self) -> None:  # noqa: N802 - http.server's names
        self._route("GET")

    def do_POST(self) -> None:  # noqa: N802
        self._route("POST")

    def do_PUT(self) -> None:  # noqa: N802
        self._route("PUT")


def serve(state: State | None = None) -> tuple[ThreadingHTTPServer, State, str]:
    """Start the mock on a free local port; returns (server, state, base URL). Call ``server.shutdown()`` after."""
    state = state or State()
    handler = type("BoundHandler", (Handler,), {"state": state})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    host, port = server.server_address[:2]
    return server, state, f"http://{host}:{port}"
