"""A small client of Codabench's REST API: log in, find the phase, upload a submission zip, read its status and score.

Standard library only. It makes the same requests, in the same order, as Codabench's own web upload (Codabench v1.33,
``src/static/riot/competitions/detail/submission_upload.tag``): create a data object (``POST /api/datasets/``), put the
zip's bytes to the storage URL it returns (with no Codabench credentials), mark the upload complete, create the
submission (``POST /api/submissions/``), then read ``GET /api/submissions/<id>/``. This API is the one Codabench's
pages use; it is not documented for participants and may change, so every unexpected answer stops with the raw
status and message.

Credentials come from environment variables only and are never printed or written anywhere:

- ``CODABENCH_TOKEN``: your API token, or
- ``CODABENCH_USERNAME`` and ``CODABENCH_PASSWORD``: exchanged for the token at each run (``POST /api/api-token-auth/``;
  the username field also takes your account's email). An account created with "Sign in with GitHub" has no password
  until you set one with Codabench's password reset.

Nothing here retries a refusal: a 400 or 403 (the daily limit, a closed phase, an unapproved registration) is shown
as Codabench wrote it, and a failure after the submission request is reported without resubmitting, so a script
never spends a daily slot you did not mean to spend.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path


DEFAULT_URL = "https://www.codabench.org"
TIMEOUT_S = 60.0  # seconds per HTTP request
MIN_POLL_S = 10.0  # the shortest interval between two status reads (be gentle with a shared service)
TERMINAL = ("Finished", "Failed", "Cancelled")
USER_AGENT = "shockbench-flow-starter/0.1.0 (+python urllib)"


class CodabenchError(RuntimeError):
    """An answer the client cannot use: the HTTP status and Codabench's own message."""

    def __init__(self, message: str, status: int | None = None, body: object = None) -> None:
        super().__init__(message)
        self.status = status
        self.body = body


def competition_id(value: str | int) -> tuple[int, str | None]:
    """(competition id, secret key or None) of an id or a competition URL (``.../competitions/<id>/?secret_key=``)."""
    text = str(value).strip()
    if text.isdecimal():
        return int(text), None
    m = re.search(r"/competitions/(\d+)", text)
    if not m:
        raise ValueError(f"not a competition id or URL: {text!r}")
    q = urllib.parse.parse_qs(urllib.parse.urlparse(text).query)
    return int(m.group(1)), (q.get("secret_key") or [None])[0]


def _detail(body: object) -> str:
    """Codabench's error message in one line (DRF answers ``{"detail": ...}``, ``{"field": [...]}`` or a list)."""
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


@dataclass
class Credentials:
    """What the environment gives: a token, or a username and password. ``repr`` never shows a secret."""

    token: str | None = None
    username: str | None = None
    password: str | None = None

    @classmethod
    def from_env(cls, environ: dict | None = None) -> Credentials:
        env = os.environ if environ is None else environ
        return cls(
            token=env.get("CODABENCH_TOKEN") or None,
            username=env.get("CODABENCH_USERNAME") or None,
            password=env.get("CODABENCH_PASSWORD") or None,
        )

    def describe(self) -> str:
        if self.token:
            return "CODABENCH_TOKEN"
        if self.username and self.password:
            return "CODABENCH_USERNAME and CODABENCH_PASSWORD"
        return "none"

    def __repr__(self) -> str:
        return f"Credentials({self.describe()})"


class Client:
    """Codabench's API at ``base_url`` with a token (``login`` gets one from a username and password)."""

    def __init__(self, base_url: str = DEFAULT_URL, credentials: Credentials | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.credentials = credentials or Credentials()
        self._token = self.credentials.token

    # ----- HTTP ------------------------------------------------------------------------------------------------------
    def _open(self, req: urllib.request.Request) -> tuple[int, bytes]:
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
                return resp.status, resp.read()
        except urllib.error.HTTPError as err:
            return err.code, err.read()
        except urllib.error.URLError as err:
            raise CodabenchError(f"{req.get_method()} {req.full_url}: cannot connect ({err.reason})") from None

    def request(self, method: str, path: str, body: dict | None = None, *, auth: bool = True, ok=(200,)) -> object:
        """JSON in, JSON out; a status outside ``ok`` raises ``CodabenchError`` with Codabench's message."""
        url = path if path.startswith("http") else f"{self.base_url}{path}"
        headers = {"Accept": "application/json", "User-Agent": USER_AGENT}
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        if auth:
            headers["Authorization"] = f"Token {self.token()}"
        status, raw = self._open(urllib.request.Request(url, data=data, headers=headers, method=method))
        try:
            parsed = json.loads(raw.decode() or "null")
        except (UnicodeDecodeError, json.JSONDecodeError):
            snippet = raw[:200].decode(errors="replace")
            raise CodabenchError(f"{method} {path}: HTTP {status}, not JSON: {snippet!r}", status) from None
        if status not in ok:
            raise CodabenchError(f"{method} {path}: HTTP {status}: {_detail(parsed)}", status, parsed)
        return parsed

    # ----- authentication --------------------------------------------------------------------------------------------
    def token(self) -> str:
        if self._token is None:
            self.login()
        return self._token

    def login(self) -> None:
        """Exchange the username and password for the account's token, kept in memory only."""
        c = self.credentials
        if not (c.username and c.password):
            raise CodabenchError(
                "no credentials: set CODABENCH_TOKEN, or CODABENCH_USERNAME and CODABENCH_PASSWORD, in the environment"
            )
        out = self.request("POST", "/api/api-token-auth/", {"username": c.username, "password": c.password}, auth=False)
        if not isinstance(out, dict) or not isinstance(out.get("token"), str):
            raise CodabenchError("POST /api/api-token-auth/: no token in the answer")
        self._token = out["token"]

    # ----- read ------------------------------------------------------------------------------------------------------
    def competition(self, pk: int, secret_key: str | None = None) -> dict:
        path = f"/api/competitions/{pk}/" + (f"?secret_key={urllib.parse.quote(secret_key)}" if secret_key else "")
        out = self.request("GET", path)
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

    # ----- write -----------------------------------------------------------------------------------------------------
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
        """The zip's bytes to the storage URL: no Codabench credentials, the Content-Type the URL was signed with."""
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


# ----- the steps the CLI runs ----------------------------------------------------------------------------------------
def pick_phase(comp: dict, phase: str | None = None) -> dict:
    """The phase named ``phase`` (case-insensitive), else the one whose status is Current."""
    phases = comp.get("phases", [])
    if phase:
        hits = [p for p in phases if str(p.get("name", "")).lower() == phase.lower()]
        if not hits:
            raise CodabenchError(f"no phase named {phase!r}; the phases are {[p.get('name') for p in phases]}")
        return hits[0]
    current = [p for p in phases if p.get("status") == "Current"]
    if len(current) != 1:
        raise CodabenchError(
            f"{len(current)} phases are open now; name one with --phase ({[p.get('name') for p in phases]})"
        )
    return current[0]


def preflight(client: Client, pk: int, secret_key: str | None, phase: str | None) -> tuple[dict, dict]:
    """(competition, phase) after the checks the web page makes before an upload; raises with what to do."""
    comp = client.competition(pk, secret_key)
    status = comp.get("participant_status")
    if status != "approved":
        what = {
            None: "you are not registered: register on the competition page",
            "pending": "your registration is pending: the organisers approve it, then you can submit",
        }.get(status, f"your registration status is {status!r}")
        raise CodabenchError(f"cannot submit to competition {pk}: {what}")
    ph = pick_phase(comp, phase)
    if ph.get("status") != "Current":
        raise CodabenchError(f"phase {ph.get('name')!r} is not open now (status {ph.get('status')!r})")
    can, reason = client.can_submit(ph["id"])
    if not can:
        raise CodabenchError(f"Codabench says you cannot submit to {ph.get('name')!r} now: {reason}")
    return comp, ph


def upload(client: Client, zip_path: Path, comp: dict, ph: dict) -> dict:
    """Create the data object, put the bytes, complete it and create the submission; returns the submission."""
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
    """{column key: score} of a submission (or of its children, for a multi-task one)."""
    out = {}
    for s in sub.get("scores", []) or []:
        try:
            out[str(s.get("column_key"))] = float(s.get("score"))
        except (TypeError, ValueError):
            continue
    return out


def wait(client: Client, sub_id: int, poll_s: float = 30.0, timeout_s: float = 4 * 3600.0, echo=print) -> dict:
    """Poll a submission until it is Finished, Failed or Cancelled (at most every ``MIN_POLL_S`` seconds)."""
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
