# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import hashlib
import os
import pathlib
import shutil

import pooch  # type: ignore[import-untyped]

from icon4py.model.testing import config, locking


def download_and_extract(
    uri: str,
    dst: pathlib.Path,
) -> None:
    """
    Download and extract a tar file with locking.

    Args:
        uri: download url for archived data
        dst: the archive is extracted at this path

    Downloads the archive to a temporary cache directory (configured via
    ``ICON4PY_DOWNLOAD_CACHE``, defaulting to a subdirectory of the system
    temp directory), extracts to ``dst``, and deletes the archive. If
    extraction fails the archive is left in the cache so that a subsequent
    run can reuse it without re-downloading.
    """
    dst.mkdir(parents=True, exist_ok=True)
    cache = config.DOWNLOAD_CACHE_PATH
    cache.mkdir(parents=True, exist_ok=True)

    completion_marker = dst / ".extraction_complete"
    lockfile = "filelock.lock"

    if completion_marker.exists():
        return

    with locking.lock(dst, lockfile=lockfile):
        if completion_marker.exists():
            return
        # Clean up any partial data from previous failed attempts
        # (except for the lockfile)
        for item in dst.iterdir():
            if item.name != lockfile:
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
        # Use a URI-derived filename to avoid collisions between different downloads
        uri_hash = hashlib.sha256(uri.encode()).hexdigest()[:16]
        archive_fname = f"archive_{uri_hash}.tar.gz"
        pooch.retrieve(
            url=uri,
            known_hash=None,
            path=str(cache),
            fname=archive_fname,
            processor=pooch.Untar(extract_dir=str(dst.resolve())),
        )
        completion_marker.touch()
        (cache / archive_fname).unlink(missing_ok=True)


def download_test_data(dst: pathlib.Path, uri: str) -> None:
    if config.ENABLE_TESTDATA_DOWNLOAD:
        download_and_extract(uri, dst)
    else:
        # If test data download is disabled, we check if the directory exists
        # and isn't empty without locking. We assume the location is managed by the user
        # and avoid locking shared directories (e.g. on CI).
        if not dst.exists():
            raise RuntimeError(f"Test data {dst} does not exist, and downloading is disabled.")
        elif not any(os.scandir(dst)):
            raise RuntimeError(f"Test data {dst} exists but is empty, and downloading is disabled.")
