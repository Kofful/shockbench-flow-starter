"""upload and status against a local mock of Codabench's API (tests/mock_codabench.py); nothing leaves this machine."""

import pytest

from sbf_starter import cli as sbf
from sbf_starter import codabench
from tests import mock_codabench as mock
from tests.conftest import ROOT


@pytest.fixture
def server(monkeypatch):
    srv, state, url = mock.serve()
    monkeypatch.setenv("CODABENCH_URL", url)
    monkeypatch.setenv("CODABENCH_COMPETITION", f"{url}/competitions/{mock.COMPETITION}/")
    monkeypatch.setattr(codabench, "MIN_POLL_S", 0.0)
    yield state
    srv.shutdown()


@pytest.fixture
def zip_path(tmp_path):
    return sbf.pack(str(ROOT / "src" / "sbf_starter" / "agents" / "template"), out=str(tmp_path / "submission.zip"))


def login_env(monkeypatch, password=mock.PASSWORD):
    monkeypatch.setenv("CODABENCH_USERNAME", mock.USERNAME)
    monkeypatch.setenv("CODABENCH_PASSWORD", password)


def test_competition_id_from_url():
    assert codabench.competition_id("https://www.codabench.org/competitions/1234/") == (1234, None)
    assert codabench.competition_id("https://www.codabench.org/competitions/77/?secret_key=abc-1") == (77, "abc-1")
    assert codabench.competition_id("99") == (99, None)
    with pytest.raises(ValueError):
        codabench.competition_id("https://example.org/nothing")


def test_upload_and_wait_with_a_password(server, zip_path, monkeypatch, capsys):
    login_env(monkeypatch)
    sbf.upload(zip_path, wait=True, poll_s=0)
    out = capsys.readouterr().out
    assert "phase 'Development'" in out and "Finished" in out and "rss 0.4281" in out
    (ds,) = server.datasets.values()
    assert ds["completed"] and ds["blob"] == open(zip_path, "rb").read()
    storage = [h for m, p, h in server.requests if p.startswith("/storage/")]
    assert storage and all("Authorization" not in h for h in storage)
    assert mock.TOKEN not in out and mock.PASSWORD not in out  # credentials are never printed


def test_upload_with_a_token_and_status(server, zip_path, monkeypatch, capsys):
    monkeypatch.setenv("CODABENCH_TOKEN", mock.TOKEN)
    sbf.upload(zip_path)
    (sub_id,) = server.submissions
    sbf.status(sub_id, wait=True, poll_s=0)
    sbf.status()
    out = capsys.readouterr().out
    assert f"status {sub_id} --wait" in out and "Finished" in out and "1 submission(s)" in out
    assert not any(p == "/api/api-token-auth/" for _m, p, _h in server.requests)  # the token needs no login


def test_dry_run_uploads_nothing(server, zip_path, monkeypatch, capsys):
    login_env(monkeypatch)
    sbf.upload(zip_path, dry_run=True)
    assert "dry run: nothing uploaded" in capsys.readouterr().out
    assert not server.datasets and not server.submissions


def test_bad_password_stops(server, zip_path, monkeypatch, capsys):
    login_env(monkeypatch, password="wrong")
    with pytest.raises(SystemExit) as exc:
        sbf.upload(zip_path)
    assert exc.value.code == sbf.FAILED
    out = capsys.readouterr().out
    assert "Unable to log in with provided credentials" in out and "wrong" not in out


def test_no_credentials_stops(server, zip_path, capsys):
    with pytest.raises(SystemExit):
        sbf.upload(zip_path)
    assert "no credentials" in capsys.readouterr().out


def test_pending_registration_is_not_retried(server, zip_path, monkeypatch, capsys):
    login_env(monkeypatch)
    server.approved = "pending"
    with pytest.raises(SystemExit):
        sbf.upload(zip_path)
    assert "pending" in capsys.readouterr().out
    assert not server.datasets


def test_daily_limit_stops_before_uploading(server, zip_path, monkeypatch, capsys):
    login_env(monkeypatch)
    server.max_per_day = 1
    sbf.upload(zip_path)
    with pytest.raises(SystemExit):
        sbf.upload(zip_path)
    assert "Reached maximum allowed submissions for today" in capsys.readouterr().out
    assert len(server.datasets) == 1 and len(server.submissions) == 1


def test_closed_phase_stops(server, zip_path, monkeypatch, capsys):
    login_env(monkeypatch)
    server.dev_status = "Previous"
    with pytest.raises(SystemExit):
        sbf.upload(zip_path)
    assert "open now" in capsys.readouterr().out and not server.datasets


def test_failed_submission_shows_its_details(server, zip_path, monkeypatch, capsys):
    login_env(monkeypatch)
    server.fail_next = True
    sbf.upload(zip_path, wait=True, poll_s=0)
    out = capsys.readouterr().out
    assert "Failed" in out and "the ingestion stopped" in out


def test_upload_refuses_an_invalid_zip(server, tmp_path, monkeypatch, capsys):
    login_env(monkeypatch)
    bad = tmp_path / "bad.zip"
    bad.write_bytes(b"not a zip")
    with pytest.raises(SystemExit) as exc:
        sbf.upload(str(bad))
    assert exc.value.code == sbf.REFUSED and not server.requests


def test_storage_contract_of_the_mock(server):
    """The mock refuses what a presigned storage URL refuses, so the client's own headers are what the tests pass."""
    import os
    import urllib.request

    server.datasets["k"] = {"size": 3, "completed": False, "blob": None}
    url = f"{os.environ['CODABENCH_URL']}/storage/k"
    client = codabench.Client()
    headers = {"Content-Type": "application/octet-stream"}
    assert client._open(urllib.request.Request(url, data=b"abc", headers=headers, method="PUT"))[0] == 403
    headers = {"Content-Type": "application/zip", "Authorization": "Token x"}
    assert client._open(urllib.request.Request(url, data=b"abc", headers=headers, method="PUT"))[0] == 403
    client.put_blob(url, b"abc")
    assert server.datasets["k"]["blob"] == b"abc"


def server_host(monkeypatch) -> str:
    """The mock's host:port, and CODABENCH_URL unset (the competition URL alone names the server)."""
    import os

    monkeypatch.delenv("CODABENCH_URL")
    return os.environ["CODABENCH_COMPETITION"].split("/")[2]


def test_a_competition_url_names_the_server(server, zip_path, monkeypatch, capsys):
    """Without CODABENCH_URL, the competition URL's own host is used (never www.codabench.org behind its back)."""
    url = f"http://{server_host(monkeypatch)}"
    login_env(monkeypatch)
    sbf.upload(zip_path, dry_run=True)
    assert f"Codabench at {url}," in capsys.readouterr().out and server.requests
    monkeypatch.setenv("CODABENCH_URL", "https://www.codabench.org")
    with pytest.raises(SystemExit, match="unset one of them"):
        sbf.upload(zip_path, dry_run=True)
