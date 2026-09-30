"""A standard-library client of Codabench's REST API, behind ``sbf token``, ``sbf upload`` and ``sbf status``.

It makes the requests of Codabench's own web upload (v1.33), an API not documented for participants. Only GETs are
retried, on a transient server error; an upload or submission request is never repeated, so no daily slot is spent
by accident.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


DEFAULT_URL = "https://www.codabench.org"
TIMEOUT_S = 60.0  # per HTTP request
MIN_POLL_S = 10.0  # between two status reads: a shared service
GET_RETRIES_S = (3.0, 10.0)  # waits before retrying a GET that got a server error
TERMINAL = ("Finished", "Failed", "Cancelled")
USER_AGENT = "shockbench-flow-starter/0.1.0 (+python urllib)"


class CodabenchError(RuntimeError):
    def __init__(self, message: str, status: int | None = None, body: object = None) -> None:
        super().__init__(message)
        self.status = status
        self.body = body


def competition_id(value: str | int) -> tuple[int, str | None]:
    """(id, secret key or None) of an id or a competition URL (``.../competitions/<id>/?secret_key=...``)."""
    text = str(value).strip()
    if text.isdecimal():
        return int(text), None
    m = re.search(r"/competitions/(\d+)", text)
    if not m:
        raise ValueError(f"not a competition id or URL: {text!r}")
    q = urllib.parse.parse_qs(urllib.parse.urlparse(text).query)
    return int(m.group(1)), (q.get("secret_key") or [None])[0]


def _detail(body: object) -> str:
    """Codabench's error in one line: DRF answers ``{"detail": ...}``, ``{"field": [...]}`` or a list."""
    if isinstance(body, dict):
        if "detail" in body:
            return str(body["detail"])
        parts = []
        for k, v in body.items():
            v = "; ".join(map(str, v)) if isinstance(v, list) else str(v)
            parts.append(v if k == "non_field_errors" else f"{k}: {v}")
        return "; ".join(parts)
    if isinstance(body, list):
        return "; ".join(map(str, body))
    return str(body)[:500]


class Client:
    """Codabench's API at ``base_url``; ``repr`` never shows the token."""

    def __init__(self, base_url: str = DEFAULT_URL, token: str = "") -> None:
        self.base_url = base_url.rstrip("/")
        self._token = token

    def __repr__(self) -> str:
        return f"Client({self.base_url!r})"

    def _open(self, req: urllib.request.Request) -> tuple[int, bytes]:
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
                return resp.status, resp.read()
        except urllib.error.HTTPError as err:
            return err.code, err.read()
        except urllib.error.URLError as err:
            raise CodabenchError(f"{req.get_method()} {req.full_url}: cannot connect ({err.reason})") from None

    def request(self, method: str, path: str, body: dict | None = None, *, auth: bool = True, ok=(200,)) -> object:
        """JSON in, JSON out; a status outside ``ok`` raises ``CodabenchError``."""
        url = path if path.startswith("http") else f"{self.base_url}{path}"
        headers = {"Accept": "application/json", "User-Agent": USER_AGENT}
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        if auth:
            headers["Authorization"] = f"Token {self._token}"
        status, raw = self._open(urllib.request.Request(url, data=data, headers=headers, method=method))
        for wait_s in GET_RETRIES_S if method == "GET" else ():
            if status < 500:
                break
            time.sleep(wait_s)
            status, raw = self._open(urllib.request.Request(url, data=data, headers=headers, method=method))
        shown = re.sub(r"secret_key=[^&]+", "secret_key=<secret>", path)
        try:
            parsed = json.loads(raw.decode() or "null")
        except (UnicodeDecodeError, json.JSONDecodeError):
            what = f"not JSON: {raw[:200].decode(errors='replace')!r}"
            if raw.lstrip()[:1] == b"<":
                what = "an HTML error page"
            raise CodabenchError(f"{method} {shown}: HTTP {status}, {what}", status) from None
        if status not in ok:
            raise CodabenchError(f"{method} {shown}: HTTP {status}: {_detail(parsed)}", status, parsed)
        return parsed

    def competition(self, pk: int, secret_key: str | None = None) -> dict:
        """The secret key is sent only when the competition is not visible without it.

        Codabench has answered 500 to a logged-in organiser who also sent it.
        """
        path = f"/api/competitions/{pk}/"
        try:
            out = self.request("GET", path)
        except CodabenchError as err:
            if err.status != 404 or not secret_key:
                raise
            out = self.request("GET", f"{path}?secret_key={urllib.parse.quote(secret_key)}")
        if not isinstance(out, dict) or not isinstance(out.get("phases"), list):
            raise CodabenchError(f"GET {path}: no phases in the answer (has Codabench's API changed?)")
        return out

    def can_submit(self, phase_id: int) -> tuple[bool, str | None]:
        out = self.request("GET", f"/api/can_make_submission/{phase_id}/")
        return bool(out.get("can")), out.get("reason")

    def submission(self, sub_id: int) -> dict:
        return self.request("GET", f"/api/submissions/{sub_id}/")

    def submissions(self, phase_id: int) -> list[dict]:
        out = self.request("GET", f"/api/submissions/?phase={phase_id}")
        return out.get("results", []) if isinstance(out, dict) else list(out)

    def create_dataset(self, competition_pk: int, file_name: str, size: int) -> dict:
        body = {
            "type": "submission",
            "competition": competition_pk,
            "request_sassy_file_name": file_name,
            "file_name": file_name,
            "file_size": size,
        }
        out = self.request("POST", "/api/datasets/", body, ok=(201,))
        if not isinstance(out, dict) or not out.get("key") or not out.get("sassy_url"):
            raise CodabenchError("POST /api/datasets/: no key or upload URL in the answer")
        return out

    def put_blob(self, url: str, data: bytes) -> None:
        """No Codabench credentials here, and the Content-Type the storage URL was signed with."""
        headers = {"Content-Type": "application/zip", "x-ms-blob-type": "BlockBlob", "User-Agent": USER_AGENT}
        status, raw = self._open(urllib.request.Request(url, data=data, headers=headers, method="PUT"))
        if not 200 <= status < 300:
            raise CodabenchError(f"upload to storage: HTTP {status}: {raw[:300].decode(errors='replace')}", status)

    def complete(self, key: str) -> None:
        self.request("PUT", f"/api/datasets/completed/{key}/")

    def create_submission(self, key: str, phase_id: int, task_ids: list[int] | None) -> dict:
        body = {"data": key, "phase": phase_id, "fact_sheet_answers": None, "organization": None, "queue": None}
        if task_ids:
            body["tasks"] = task_ids
        return self.request("POST", "/api/submissions/", body, ok=(201,))


def server_url(competition: str | None = None) -> str:
    """A competition URL's host; www.codabench.org for a bare id or none."""
    parsed = urllib.parse.urlparse(str(competition or "").strip())
    if parsed.scheme in ("http", "https") and parsed.netloc:
        return f"{parsed.scheme}://{parsed.netloc}"
    return DEFAULT_URL


def get_token(base_url: str, username: str, password: str) -> str:
    body = {"username": username, "password": password}
    out = Client(base_url).request("POST", "/api/api-token-auth/", body, auth=False)
    if not isinstance(out, dict) or not isinstance(out.get("token"), str):
        raise CodabenchError("POST /api/api-token-auth/: no token in the answer")
    return out["token"]


def from_env(competition: str | None = None) -> tuple[Client, int, str | None]:
    """(client, competition id, secret key) from CODABENCH_COMPETITION and CODABENCH_TOKEN; nothing is sent here."""
    value = str(competition or os.environ.get("CODABENCH_COMPETITION") or "").strip()
    if not value:
        raise CodabenchError("no competition: set CODABENCH_COMPETITION to the competition's URL (see .env.example)")
    token = (os.environ.get("CODABENCH_TOKEN") or "").strip()
    if not token:
        raise CodabenchError("no token: set CODABENCH_TOKEN in .env (uv run sbf token gets it for you)")
    try:
        pk, secret = competition_id(value)
    except ValueError as err:
        raise CodabenchError(str(err)) from None
    return Client(server_url(value), token), pk, secret


def pick_phase(comp: dict, phase: str | None = None) -> dict:
    """The phase named ``phase`` (any case), else the one open now."""
    phases = comp.get("phases", [])
    if phase:
        hits = [p for p in phases if str(p.get("name", "")).lower() == phase.lower()]
        if not hits:
            raise CodabenchError(f"no phase named {phase!r}; the phases are {[p.get('name') for p in phases]}")
        return hits[0]
    current = [p for p in phases if p.get("status") == "Current"]
    listed = ", ".join(f"{p.get('name')} ({p.get('status')})" for p in phases)
    if not current:
        raise CodabenchError(f"no phase is open now: {listed}; --phase names one")
    if len(current) > 1:
        raise CodabenchError(f"{len(current)} phases are open now; name one with --phase: {listed}")
    return current[0]


def preflight(client: Client, pk: int, secret_key: str | None, phase: str | None) -> tuple[dict, dict]:
    """(competition, phase) after the checks the web page makes before an upload."""
    comp = client.competition(pk, secret_key)
    status = comp.get("participant_status")
    if status != "approved":
        what = {
            None: "you are not registered: register on the competition page",
            "pending": "your registration is pending: the organisers approve it, then you can submit",
        }.get(status, f"your registration status is {status!r}")
        raise CodabenchError(f"cannot submit to competition {pk}: {what}")
    ph = pick_phase(comp, phase)  # a named phase may be closed: Codabench decides (organisers submit early)
    can, reason = client.can_submit(ph["id"])
    if not can:
        raise CodabenchError(f"Codabench says you cannot submit to {ph.get('name')!r} now: {reason}")
    return comp, ph


def upload(client: Client, zip_path: Path, comp: dict, ph: dict) -> dict:
    """The new submission: data object, storage PUT, completion, submission request."""
    data = zip_path.read_bytes()
    name = zip_path.name if len(zip_path.name) <= 64 and zip_path.suffix == ".zip" else "submission.zip"
    ds = client.create_dataset(comp["id"], name, len(data))
    client.put_blob(ds["sassy_url"], data)
    client.complete(ds["key"])
    tasks = [t["id"] if isinstance(t, dict) else t for t in ph.get("tasks", [])] or None
    try:
        return client.create_submission(ds["key"], ph["id"], tasks)
    except CodabenchError as err:
        if err.status is not None and err.status >= 500:
            raise CodabenchError(
                f"{err} - the submission may have been created anyway: run `status` before submitting again",
                err.status,
            ) from None
        raise


def scores(sub: dict) -> dict[str, float]:
    out = {}
    for s in sub.get("scores", []) or []:
        try:
            out[str(s.get("column_key"))] = float(s.get("score"))
        except (TypeError, ValueError):
            continue
    return out


def wait(client: Client, sub_id: int, poll_s: float = 30.0, timeout_s: float = 4 * 3600.0, echo=print) -> dict:
    """Poll until the submission is Finished, Failed or Cancelled."""
    poll = max(float(poll_s), MIN_POLL_S)
    deadline = time.monotonic() + timeout_s
    last = None
    while True:
        sub = client.submission(sub_id)
        status = sub.get("status")
        if status != last:
            echo(f"submission {sub_id}: {status}")
            last = status
        if status in TERMINAL:
            return sub
        if time.monotonic() > deadline:
            raise CodabenchError(f"submission {sub_id} is still {status!r} after {timeout_s:.0f} s")
        time.sleep(poll)
