"""``security.document_parsing_in_sandbox``: under a non-local backend read_file parses a document inside the
session's own sandbox with the engine's own extraction code, so the host never parses a document's contents."""

from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

import tools.read_extract as read_extract
from tools import file_tools

_NS_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def _docx(path: Path, text: str) -> Path:
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("[Content_Types].xml", "<Types/>")
        z.writestr("word/document.xml", f'<w:document xmlns:w="{_NS_W}"><w:body><w:p><w:r><w:t>{text}</w:t>'
                                        f'</w:r></w:p></w:body></w:document>')
    return path


def test_the_extraction_code_runs_on_its_own_outside_the_engine(tmp_path: Path) -> None:
    document = _docx(tmp_path / "report.docx", "Quarterly widgets: 1337")
    source = Path(read_extract.__file__).read_text(encoding="utf-8")
    run = subprocess.run([sys.executable, "-I", "-", str(document)], input=source, capture_output=True, text=True,
                         cwd=tmp_path, timeout=60, check=False)
    answer = json.loads(run.stdout.strip().splitlines()[-1])
    assert "Quarterly widgets: 1337" in answer["text"] and answer["size"] == document.stat().st_size


def test_a_document_the_code_cannot_read_is_an_error_not_a_crash(tmp_path: Path) -> None:
    broken = tmp_path / "broken.docx"
    broken.write_bytes(b"not a zip")
    source = Path(read_extract.__file__).read_text(encoding="utf-8")
    run = subprocess.run([sys.executable, "-I", "-", str(broken)], input=source, capture_output=True, text=True,
                         cwd=tmp_path, timeout=60, check=False)
    assert "error" in json.loads(run.stdout.strip().splitlines()[-1])


class _Sandbox:
    """Runs what read_file sends to the sandbox as a local process — a stand-in for the container."""

    def __init__(self) -> None:
        self.commands: list[str] = []

    def execute(self, command: str, cwd: str = "", *, timeout=None, stdin_data=None):
        self.commands.append(command)
        run = subprocess.run(["bash", "-c", command.replace("python3", sys.executable + " -I", 1)],
                             input=stdin_data, capture_output=True, text=True, timeout=60, check=False)
        return {"output": run.stdout + run.stderr, "returncode": run.returncode}


@pytest.fixture
def sandboxed(monkeypatch, tmp_path):
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    sandbox = _Sandbox()
    monkeypatch.setattr(file_tools, "_get_file_ops", lambda task_id="default": SimpleNamespace(
        env=sandbox, _add_line_numbers=lambda text, offset: text,
        read_file_bytes=mock.Mock(side_effect=AssertionError("the host fetched the document's bytes"))))
    monkeypatch.setattr(read_extract, "extract_document_bytes",
                        mock.Mock(side_effect=AssertionError("the host parsed the document")))
    with mock.patch("hermes_cli.config.load_config",
                    return_value={"security": {"document_parsing_in_sandbox": True}}):
        yield sandbox


def test_read_file_parses_the_document_in_the_sandbox(sandboxed, tmp_path) -> None:
    document = _docx(tmp_path / "it's a $(report).docx", "Secret word: pelican")
    result = json.loads(file_tools._read_extracted_document(str(document), document, 1, 100, "t1"))
    assert "Secret word: pelican" in result["content"] and result["extracted_document"] is True
    [command] = sandboxed.commands
    assert "'\"'\"'" in command and "$(report)" in command


def test_without_a_sandbox_the_document_is_not_parsed_on_the_host(sandboxed, monkeypatch, tmp_path) -> None:
    def no_sandbox(task_id="default"):
        raise RuntimeError("no sandbox")

    monkeypatch.setattr(file_tools, "_get_file_ops", no_sandbox)
    document = _docx(tmp_path / "report.docx", "x")
    result = json.loads(file_tools._read_extracted_document(str(document), document, 1, 100, "t1"))
    assert "error" in result


def test_a_sandbox_that_cannot_run_the_parser_is_an_error(sandboxed, monkeypatch, tmp_path) -> None:
    sandboxed.execute = lambda command, cwd="", *, timeout=None, stdin_data=None: {
        "output": "bash: python3: command not found", "returncode": 127}
    document = _docx(tmp_path / "report.docx", "x")
    result = json.loads(file_tools._read_extracted_document(str(document), document, 1, 100, "t1"))
    assert "error" in result and "python3" in result["error"]


@pytest.mark.parametrize("backend, enabled", [("local", True), ("docker", False)])
def test_the_host_parses_as_before_when_off_or_local(monkeypatch, tmp_path, backend, enabled) -> None:
    monkeypatch.setenv("TERMINAL_ENV", backend)
    with mock.patch("hermes_cli.config.load_config",
                    return_value={"security": {"document_parsing_in_sandbox": enabled}}):
        assert not file_tools._documents_parsed_in_sandbox()
