"""upload and status against a local mock of Codabench's API (tests/mock_codabench.py); nothing leaves this machine."""

import pytest
from dotenv import dotenv_values

from sbf_starter import cli as sbf
from sbf_starter import codabench
from tests import mock_codabench as mock
from tests.conftest import write_agent


@pytest.fixture
def server(monkeypatch):
    srv, state, url = mock.serve()
    monkeypatch.setenv("CODABENCH_COMPETITION", f"{url}/competitions/{mock.COMPETITION}/")
    monkeypatch.setenv("CODABENCH_TOKEN", mock.TOKEN)
    monkeypatch.setattr(codabench, "MIN_POLL_S", 0.0)
    monkeypatch.setattr(codabench, "GET_RETRIES_S", (0.0, 0.0))
    yield state
    srv.shutdown()


@pytest.fixture
def zip_path(tmp_path):
    return sbf.pack("template", out=str(tmp_path / "submission.zip"))


@pytest.fixture
def login(monkeypatch):
    """Answers `sbf token`'s two prompts: the mock's username, then a password (the right one by default)."""

    def answer(password: str = mock.PASSWORD) -> None:
        monkeypatch.setattr("builtins.input", lambda prompt="": mock.USERNAME)
        monkeypatch.setattr("getpass.getpass", lambda prompt="": password)

    return answer


def test_token_is_saved_in_env(server, tmp_path, monkeypatch, capsys, login):
    monkeypatch.chdir(tmp_path)
    env = tmp_path / ".env"
    env.write_text("# my settings\nCODABENCH_COMPETITION=keep-me\nCODABENCH_TOKEN=\n")
    login()
    sbf.token()
    assert dotenv_values(env) == {"CODABENCH_COMPETITION": "keep-me", "CODABENCH_TOKEN": mock.TOKEN}
    assert "# my settings" in env.read_text()
    out = capsys.readouterr().out
    assert "CODABENCH_TOKEN saved in .env" in out and mock.TOKEN not in out and mock.PASSWORD not in out


def test_token_creates_env_from_the_example(server, tmp_path, monkeypatch, login):
    import dotenv

    import sbf_starter

    monkeypatch.setattr(sbf_starter, "ROOT", tmp_path)  # the repository's root, where .env goes when there is none
    monkeypatch.setattr(dotenv, "find_dotenv", lambda **kwargs: "")
    (tmp_path / ".env.example").write_text("# the example\nCODABENCH_COMPETITION=\nCODABENCH_TOKEN=\n")
    login()
    sbf.token()
    env = tmp_path / ".env"
    assert dotenv_values(env) == {"CODABENCH_COMPETITION": "", "CODABENCH_TOKEN": mock.TOKEN}
    assert "# the example" in env.read_text() and env.stat().st_mode & 0o777 == 0o600


def test_a_wrong_password_saves_nothing(server, tmp_path, monkeypatch, capsys, login):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("CODABENCH_TOKEN=\n")
    login("not-my-password-3e9a")
    with pytest.raises(SystemExit) as exc:
        sbf.token()
    assert exc.value.code == sbf.FAILED
    out = capsys.readouterr().out
    assert "Unable to log in with provided credentials" in out and "not-my-password-3e9a" not in out
    assert dotenv_values(tmp_path / ".env") == {"CODABENCH_TOKEN": ""}


def test_competition_id_from_url():
    assert codabench.competition_id("https://www.codabench.org/competitions/1234/") == (1234, None)
    assert codabench.competition_id("https://www.codabench.org/competitions/77/?secret_key=abc-1") == (77, "abc-1")
    assert codabench.competition_id("99") == (99, None)
    with pytest.raises(ValueError):
        codabench.competition_id("https://example.org/nothing")


def test_upload_and_wait(server, zip_path, capsys):
    sbf.upload(zip_path, wait=True, poll_s=0)
    out = capsys.readouterr().out
    assert "phase 'Development'" in out and "Finished" in out and "rss 0.5000" in out
    (ds,) = server.datasets.values()
    assert ds["completed"] and ds["blob"] == open(zip_path, "rb").read()
    storage = [h for m, p, h in server.requests if p.startswith("/storage/")]
    assert storage and all("Authorization" not in h for h in storage)
    assert mock.TOKEN not in out  # the token is never printed


def test_upload_by_name_packs_it_under_outputs(server, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    sbf.upload("heuristic")
    assert (tmp_path / "outputs" / "heuristic.zip").is_file()
    (sub_id,) = server.submissions
    sbf.status(sub_id, wait=True, poll_s=0)
    sbf.status()
    out = capsys.readouterr().out
    assert f"status {sub_id} --wait" in out and "Finished" in out and "1 submission(s)" in out


def test_upload_refuses_an_import_the_image_lacks(server, tmp_path, capsys):
    folder = write_agent(
        tmp_path / "pandas_agent",
        """
        import pandas

        class Agent:
            def __init__(self, config=None):
                pass

            def act(self, observation):
                return {"flows": observation["action_mask"] * 0.0}
        """,
    )
    with pytest.raises(SystemExit) as exc:
        sbf.upload(str(folder))
    assert exc.value.code == sbf.FAILED
    assert "agent.py imports pandas" in capsys.readouterr().out and not server.requests


def test_dry_run_uploads_nothing(server, zip_path, capsys):
    sbf.upload(zip_path, dry_run=True)
    assert "dry run: nothing uploaded" in capsys.readouterr().out
    assert not server.datasets and not server.submissions


def test_a_wrong_token_stops(server, zip_path, monkeypatch, capsys):
    monkeypatch.setenv("CODABENCH_TOKEN", "not-the-token-7c1f")
    with pytest.raises(SystemExit) as exc:
        sbf.upload(zip_path)
    assert exc.value.code == sbf.FAILED
    out = capsys.readouterr().out
    assert "Invalid token" in out and "not-the-token-7c1f" not in out


@pytest.mark.parametrize("unset", ["CODABENCH_TOKEN", "CODABENCH_COMPETITION"])
def test_a_missing_variable_stops_before_any_request(server, zip_path, monkeypatch, capsys, unset):
    monkeypatch.delenv(unset)
    with pytest.raises(SystemExit):
        sbf.upload(zip_path)
    assert unset in capsys.readouterr().out and not server.requests


def test_pending_registration_is_not_retried(server, zip_path, capsys):
    server.approved = "pending"
    with pytest.raises(SystemExit):
        sbf.upload(zip_path)
    assert "pending" in capsys.readouterr().out
    assert not server.datasets


def test_daily_limit_stops_before_uploading(server, zip_path, capsys):
    server.max_per_day = 1
    sbf.upload(zip_path)
    with pytest.raises(SystemExit):
        sbf.upload(zip_path)
    assert "Reached maximum allowed submissions for today" in capsys.readouterr().out
    assert len(server.datasets) == 1 and len(server.submissions) == 1


def test_closed_phase_stops(server, zip_path, capsys):
    server.dev_status = "Previous"
    with pytest.raises(SystemExit):
        sbf.upload(zip_path)
    assert "open now" in capsys.readouterr().out and not server.datasets


def test_a_named_closed_phase_stops_a_participant(server, zip_path, capsys):
    server.dev_status = "Next"
    with pytest.raises(SystemExit):
        sbf.upload(zip_path, phase="Development")
    assert "not currently accepting submissions" in capsys.readouterr().out and not server.datasets


def test_an_organiser_submits_to_a_phase_before_it_opens(server, zip_path, capsys):
    server.dev_status, server.organiser = "Next", True
    sbf.upload(zip_path, phase="Development", wait=True, poll_s=0)
    out = capsys.readouterr().out
    assert "(id 71, Next)" in out and "Finished" in out


def test_failed_submission_shows_its_details(server, zip_path, capsys):
    server.fail_next = True
    sbf.upload(zip_path, wait=True, poll_s=0)
    out = capsys.readouterr().out
    assert "Failed" in out and "the ingestion stopped" in out


def with_secret(monkeypatch, secret: str = "s3cr3t-key") -> None:
    import os

    monkeypatch.setenv("CODABENCH_COMPETITION", os.environ["CODABENCH_COMPETITION"] + f"?secret_key={secret}")


def test_a_member_never_sends_the_secret_key(server, zip_path, monkeypatch, capsys):
    """Codabench has answered 500 when a logged-in organiser also sent the key, so it is left out."""
    with_secret(monkeypatch)
    sbf.upload(zip_path, dry_run=True)
    assert "dry run: nothing uploaded" in capsys.readouterr().out
    assert not [p for _m, p, _h in server.requests if "secret_key" in p]


def test_a_private_competition_is_found_by_its_secret_key(server, zip_path, monkeypatch, capsys):
    server.member = False
    with_secret(monkeypatch)
    sbf.upload(zip_path, dry_run=True)
    assert "dry run: nothing uploaded" in capsys.readouterr().out
    assert [p for _m, p, _h in server.requests if "secret_key" in p]


def test_no_open_phase_says_so(server, zip_path, capsys):
    server.dev_status = "Next"
    with pytest.raises(SystemExit):
        sbf.upload(zip_path, dry_run=True)
    assert "no phase is open now: Development (Next), Final (Next)" in capsys.readouterr().out


def test_an_html_error_page_is_reported_briefly_and_hides_the_key(server, monkeypatch):
    with_secret(monkeypatch)
    client, pk, secret = codabench.from_env()
    with pytest.raises(codabench.CodabenchError) as exc:
        client.request("GET", f"/api/competitions/{pk}/?secret_key={secret}")
    assert exc.value.status == 500 and "an HTML error page" in str(exc.value) and secret not in str(exc.value)


def test_a_transient_server_error_on_a_get_is_retried(server, zip_path, capsys):
    server.flaky_gets = 2
    sbf.upload(zip_path, dry_run=True)
    assert "dry run: nothing uploaded" in capsys.readouterr().out


def test_a_lasting_server_error_stops(server, zip_path, capsys):
    server.flaky_gets = 3
    with pytest.raises(SystemExit):
        sbf.upload(zip_path, dry_run=True)
    assert "HTTP 502, an HTML error page" in capsys.readouterr().out and not server.datasets


def test_upload_refuses_an_invalid_zip(server, tmp_path):
    bad = tmp_path / "bad.zip"
    bad.write_bytes(b"not a zip")
    with pytest.raises(SystemExit) as exc:
        sbf.upload(str(bad))
    assert exc.value.code == sbf.REFUSED and not server.requests


def test_storage_contract_of_the_mock(server):
    """The mock refuses what a presigned storage URL refuses, so the client's own headers are what the tests pass."""
    import urllib.request

    client, _pk, _secret = codabench.from_env()
    server.datasets["k"] = {"size": 3, "completed": False, "blob": None}
    url = f"{client.base_url}/storage/k"
    headers = {"Content-Type": "application/octet-stream"}
    assert client._open(urllib.request.Request(url, data=b"abc", headers=headers, method="PUT"))[0] == 403
    headers = {"Content-Type": "application/zip", "Authorization": "Token x"}
    assert client._open(urllib.request.Request(url, data=b"abc", headers=headers, method="PUT"))[0] == 403
    client.put_blob(url, b"abc")
    assert server.datasets["k"]["blob"] == b"abc"


def test_the_competition_url_names_the_server(server, zip_path, monkeypatch, capsys):
    """The server is the competition URL's host; a bare id means www.codabench.org."""
    import os

    host = os.environ["CODABENCH_COMPETITION"].split("/")[2]
    sbf.upload(zip_path, dry_run=True)
    assert f"Codabench at http://{host}," in capsys.readouterr().out and server.requests
    client, pk, _secret = codabench.from_env("4242")
    assert client.base_url == codabench.DEFAULT_URL and pk == 4242
