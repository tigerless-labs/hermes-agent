"""scripts/lara_track_upstream.sh: the newest upstream release by date is the target, tagged vYYYY.M.D or vX.Y.Z."""

import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "lara_track_upstream.sh"
IDENTITY = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com",
            "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com"}


def _git(repo: Path, *args: str, date: str | None = None) -> str:
    env = {**os.environ, **IDENTITY}
    if date:
        env["GIT_AUTHOR_DATE"] = env["GIT_COMMITTER_DATE"] = date
    return subprocess.run(["git", "-C", str(repo), *args], env=env, check=True, capture_output=True, text=True).stdout


def _commit(repo: Path, name: str, date: str) -> None:
    (repo / name).write_text(name)
    _git(repo, "add", name)
    _git(repo, "commit", "-qm", name, date=date)


@pytest.fixture
def fork(tmp_path: Path) -> Path:
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    _git(upstream, "init", "-q", "-b", "main")
    for name, date, tag in [("a", "2026-09-24T00:00:00Z", "v2026.9.24"), ("b", "2026-10-08T00:00:00Z", "v0.21.6"),
                            ("c", "2026-10-15T00:00:00Z", "v0.22.2"),
                            ("d", "2026-10-16T00:00:00Z", "v0.22.2+canary.20261016T065950Z")]:
        _commit(upstream, name, date)
        _git(upstream, "tag", tag)
    _git(upstream, "tag", "rc.1-v0.22.3")
    fork = tmp_path / "fork"
    subprocess.run(["git", "clone", "-q", "--no-tags", str(upstream), str(fork)], check=True)
    _git(fork, "remote", "add", "upstream", str(upstream))
    _git(fork, "fetch", "-q", "--tags", "upstream")
    _git(fork, "checkout", "-q", "-b", "lara", "v2026.9.24")
    _commit(fork, "fork-change", "2026-10-01T00:00:00Z")
    return fork


def _track(fork: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(SCRIPT), "--upstream", "upstream", "--lara", "lara", *args],
                          cwd=fork, env={**os.environ, **IDENTITY}, capture_output=True, text=True)


def test_newest_release_by_date_is_the_target_across_tag_schemes(fork):
    result = _track(fork)
    assert result.returncode == 0, result.stderr
    assert "base: v2026.9.24  target: v0.22.2" in result.stdout
    assert "green: lara-v0.22.2" in result.stdout
    assert _git(fork, "merge-base", "--is-ancestor", "v0.22.2", "lara-v0.22.2") == ""


def test_a_version_tagged_release_is_accepted_as_target(fork):
    result = _track(fork, "--target", "v0.21.6")
    assert result.returncode == 0, result.stderr
    assert "target: v0.21.6" in result.stdout


@pytest.mark.parametrize("tag", ["rc.1-v0.22.3", "v0.22.2+canary.20261016T065950Z"])
def test_a_tag_that_is_not_a_release_is_refused(fork, tag):
    assert _track(fork, "--target", tag).returncode == 64


def test_lara_on_a_version_tagged_release_is_up_to_date(fork):
    _git(fork, "checkout", "-q", "-B", "lara", "v0.22.2")
    _commit(fork, "fork-change", "2026-10-16T00:00:00Z")
    result = _track(fork)
    assert result.returncode == 0, result.stderr
    assert "base: v0.22.2  target: v0.22.2" in result.stdout
    assert "up to date" in result.stdout
