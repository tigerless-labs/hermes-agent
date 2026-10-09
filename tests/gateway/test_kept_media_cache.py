"""Cache files kept under a stable key: one folder per key in the cache dir, found again by that key,
swept by age like any cached file."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from gateway.platforms import base

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32


@pytest.fixture
def caches(tmp_path, monkeypatch):
    for constant, folder in (("IMAGE_CACHE_DIR", "images"), ("AUDIO_CACHE_DIR", "audio"),
                             ("VIDEO_CACHE_DIR", "videos"), ("DOCUMENT_CACHE_DIR", "documents")):
        monkeypatch.setattr(base, constant, tmp_path / folder)
    return tmp_path


def test_a_document_kept_under_a_key_is_found_again_by_that_key(caches):
    path = base.cache_document_from_bytes(b"a,b\n", "stock.csv", kept="slack-F0STOCK")
    assert Path(path).name == "stock.csv" and Path(path).read_bytes() == b"a,b\n"
    assert base.find_kept_cache_file(base.get_document_cache_dir(), "slack-F0STOCK") == path


def test_media_kept_under_a_key_keep_their_stem_and_the_writer_s_extension(caches):
    path = base.cache_image_from_bytes(PNG, ".png", kept="slack-F0CHART", stem="chart")
    assert Path(path).name == "chart.png"
    assert base.find_kept_cache_file(base.get_image_cache_dir(), "slack-F0CHART") == path


def test_an_unknown_key_finds_nothing(caches):
    assert base.find_kept_cache_file(base.get_document_cache_dir(), "slack-F0NEVER") is None


def test_finding_a_kept_file_keeps_it_from_the_age_sweep(caches):
    path = Path(base.cache_document_from_bytes(b"x", "old.txt", kept="slack-F0OLD"))
    long_ago = time.time() - 10 * 3600
    os.utime(path, (long_ago, long_ago))
    base.find_kept_cache_file(base.get_document_cache_dir(), "slack-F0OLD")
    assert base.cleanup_document_cache(max_age_hours=1) == 0 and path.exists()


def test_the_age_sweep_removes_old_kept_files_and_their_empty_folders(caches):
    path = Path(base.cache_document_from_bytes(b"x", "old.txt", kept="slack-F0OLD"))
    long_ago = time.time() - 10 * 3600
    os.utime(path, (long_ago, long_ago))
    assert base.cleanup_document_cache(max_age_hours=1) == 1
    assert not path.exists() and not path.parent.exists()


def test_without_a_key_names_stay_random(caches):
    first, second = (base.cache_document_from_bytes(b"x", "same.txt") for _ in range(2))
    assert first != second and Path(first).parent == base.get_document_cache_dir()


@pytest.mark.parametrize("key", ["../escape", "a/b", "", ".", "..", "x" * 300, "slack-F0\x00"])
def test_redteam_a_key_that_is_not_one_plain_name_is_refused(caches, key):
    with pytest.raises(ValueError):
        base.cache_document_from_bytes(b"x", "f.txt", kept=key)
    with pytest.raises(ValueError):
        base.find_kept_cache_file(base.get_document_cache_dir(), key)


def test_redteam_a_kept_image_must_still_be_an_image(caches):
    with pytest.raises(ValueError):
        base.cache_image_from_bytes(b"<html>login</html>", ".png", kept="slack-F0FAKE", stem="fake")
    assert base.find_kept_cache_file(base.get_image_cache_dir(), "slack-F0FAKE") is None


def test_redteam_a_symlinked_kept_folder_is_not_followed(caches, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("secret")
    folder = base.get_document_cache_dir() / f"{base.KEPT_MEDIA_DIR_PREFIX}slack-F0LINK"
    folder.symlink_to(outside, target_is_directory=True)
    assert base.find_kept_cache_file(base.get_document_cache_dir(), "slack-F0LINK") is None
