# mypy: allow-untyped-defs
import errno
import hashlib
import os
import re
import shutil
import sys
import tempfile
import uuid
import warnings
import zipfile
from pathlib import Path
from typing import Any
from warnings import deprecated
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import torch
from torch.serialization import MAP_LOCATION


class _Faketqdm:  # type: ignore[no-redef]
    def __init__(self, total=None, disable=False, unit=None, *args, **kwargs):
        self.total = total
        self.disable = disable
        self.n = 0
        # Ignore all extra *args and **kwargs lest you want to reinvent tqdm

    def update(self, n):
        if self.disable:
            return

        self.n += n
        if self.total is None:
            sys.stderr.write(f"\r{self.n:.1f} bytes")
        else:
            sys.stderr.write(f"\r{100 * self.n / float(self.total):.1f}%")
        sys.stderr.flush()

    # Don't bother implementing; use real tqdm if you want
    def set_description(self, *args, **kwargs):
        pass

    def write(self, s):
        sys.stderr.write(f"{s}\n")

    def close(self):
        self.disable = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.disable:
            return

        sys.stderr.write("\n")


try:
    from tqdm import tqdm  # If tqdm is installed use it, otherwise use the fake wrapper
except ImportError:
    tqdm = _Faketqdm

__all__ = [
    "download_url_to_file",
    "get_dir",
    "load_state_dict_from_url",
    "set_dir",
]

ENV_TORCH_HOME = "TORCH_HOME"
ENV_XDG_CACHE_HOME = "XDG_CACHE_HOME"
DEFAULT_CACHE_DIR = "~/.cache"
READ_DATA_CHUNK = 128 * 1024
# matches bfd8deac from resnet18-bfd8deac.pth
HASH_REGEX = re.compile(r"-([a-f0-9]*)\.")
_PATH_SEP_PATTERN = re.compile(r"[/\\]")
_hub_dir: str | None = None


def _safe_extract_zip(zip_file, extract_to):
    """
    Safely extract a zip file, preventing zipslip attacks.

    Args:
        zip_file: ZipFile object to extract
        extract_to: Directory to extract to

    Raises:
        ValueError: If any archive entry contains unsafe paths
    """
    # Normalize the extraction directory path
    extract_to = Path(extract_to).resolve(strict=False)

    for member in zip_file.infolist():
        # Get the normalized path
        filename = os.path.normpath(member.filename)

        # Check for directory traversal attempts
        if filename.startswith(("/", "\\")):
            raise ValueError(f"Archive entry has absolute path: {member.filename}")

        if len(filename) >= 2 and filename[1] == ":" and filename[0].isalpha():
            raise ValueError(f"Archive entry has absolute path: {member.filename}")

        if ".." in re.split(_PATH_SEP_PATTERN, filename):
            raise ValueError(
                f"Archive entry contains directory traversal: {member.filename}"
            )

        # Construct the full extraction path and verify it's within extract_to
        out = (extract_to / filename).resolve(strict=False)

        if not out.is_relative_to(extract_to):
            raise ValueError(
                f"Archive entry escapes target directory: {member.filename}"
            )

        # Extract the member safely
        zip_file.extract(member, extract_to)


def _get_torch_home():
    torch_home = os.path.expanduser(
        os.getenv(
            ENV_TORCH_HOME,
            os.path.join(os.getenv(ENV_XDG_CACHE_HOME, DEFAULT_CACHE_DIR), "torch"),
        )
    )
    return torch_home


def get_dir() -> str:
    r"""
    Get the Torch Hub cache directory used for storing downloaded models & weights.

    If :func:`~torch.hub.set_dir` is not called, default path is ``$TORCH_HOME/hub`` where
    environment variable ``$TORCH_HOME`` defaults to ``$XDG_CACHE_HOME/torch``.
    ``$XDG_CACHE_HOME`` follows the X Design Group specification of the Linux
    filesystem layout, with a default value ``~/.cache`` if the environment
    variable is not set.
    """
    # Issue warning to move data if old env is set
    if os.getenv("TORCH_HUB"):
        warnings.warn(
            "TORCH_HUB is deprecated, please use env TORCH_HOME instead", stacklevel=2
        )

    if _hub_dir is not None:
        return _hub_dir
    return os.path.join(_get_torch_home(), "hub")


def set_dir(d: str | os.PathLike) -> None:
    r"""
    Optionally set the Torch Hub directory used to save downloaded models & weights.

    Args:
        d (str): path to a local folder to save downloaded models & weights.
    """
    global _hub_dir
    _hub_dir = os.path.expanduser(d)


def download_url_to_file(
    url: str,
    dst: str,
    hash_prefix: str | None = None,
    progress: bool = True,
) -> None:
    r"""Download object at the given URL to a local path.

    Args:
        url (str): URL of the object to download
        dst (str): Full path where object will be saved, e.g. ``/tmp/temporary_file``
        hash_prefix (str, optional): If not None, the SHA256 downloaded file should start with ``hash_prefix``.
            Default: None
        progress (bool, optional): whether or not to display a progress bar to stderr
            Default: True

    Example:
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_HUB)
        >>> # xdoctest: +REQUIRES(POSIX)
        >>> torch.hub.download_url_to_file(
        ...     "https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth",
        ...     "/tmp/temporary_file",
        ... )

    """
    # We deliberately save it in a temp file and move it after
    # download is complete. This prevents a local working checkpoint
    # being overridden by a broken download.
    # We deliberately do not use NamedTemporaryFile to avoid restrictive
    # file permissions being applied to the downloaded file.
    dst = os.path.expanduser(dst)
    for _ in range(tempfile.TMP_MAX):
        tmp_dst = dst + "." + uuid.uuid4().hex + ".partial"
        try:
            f = open(tmp_dst, "w+b")  # noqa: SIM115
        except FileExistsError:
            continue
        break
    else:
        raise FileExistsError(errno.EEXIST, "No usable temporary file name found")
    req = Request(url, headers={"User-Agent": "torch.hub"})
    try:
        with urlopen(req) as u:
            meta = u.info()
            if hasattr(meta, "getheaders"):
                content_length = meta.getheaders("Content-Length")
            else:
                content_length = meta.get_all("Content-Length")
            file_size = None
            if content_length is not None and len(content_length) > 0:
                file_size = int(content_length[0])

            sha256 = hashlib.sha256() if hash_prefix is not None else None
            with tqdm(
                total=file_size,
                disable=not progress,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                while True:
                    buffer = u.read(READ_DATA_CHUNK)
                    if len(buffer) == 0:
                        break
                    f.write(buffer)
                    if sha256 is not None:
                        sha256.update(buffer)
                    pbar.update(len(buffer))

            f.close()
            if sha256 is not None and hash_prefix is not None:
                digest = sha256.hexdigest()
                if digest[: len(hash_prefix)] != hash_prefix:
                    raise RuntimeError(
                        f'invalid hash value (expected "{hash_prefix}", got "{digest}")'
                    )
        shutil.move(f.name, dst)
    finally:
        f.close()
        if os.path.exists(f.name):
            os.remove(f.name)


# Hub used to support automatically extracts from zipfile manually compressed by users.
# The legacy zip format expects only one file from torch.save() < 1.6 in the zip.
# We should remove this support since zipfile is now default zipfile format for torch.save().
def _is_legacy_zip_format(filename: str) -> bool:
    if zipfile.is_zipfile(filename):
        with zipfile.ZipFile(filename) as zf:
            infolist = zf.infolist()
        return len(infolist) == 1 and not infolist[0].is_dir()
    return False


@deprecated(
    "Falling back to the old format < 1.6. This support will be "
    "deprecated in favor of default zipfile format introduced in 1.6. "
    "Please redo torch.save() to save it in the new zipfile format.",
    category=FutureWarning,
)
def _legacy_zip_load(
    filename: str,
    model_dir: str,
    map_location: MAP_LOCATION,
    weights_only: bool,
) -> dict[str, Any]:
    # Note: extractall() defaults to overwrite file if exists. No need to clean up beforehand.
    #       We deliberately don't handle tarfile here since our legacy serialization format was in tar.
    #       E.g. resnet18-5c106cde.pth which is widely used.
    with zipfile.ZipFile(filename) as f:
        members = f.infolist()
        if len(members) != 1:
            raise RuntimeError("Only one file(not dir) is allowed in the zipfile")
        # Use safe extraction to prevent zipslip attacks
        _safe_extract_zip(f, model_dir)
        extraced_name = members[0].filename
        extracted_file = os.path.join(model_dir, extraced_name)
    return torch.load(
        extracted_file, map_location=map_location, weights_only=weights_only
    )


def load_state_dict_from_url(
    url: str,
    model_dir: str | None = None,
    map_location: MAP_LOCATION = None,
    progress: bool = True,
    check_hash: bool = False,
    file_name: str | None = None,
    weights_only: bool = False,
) -> dict[str, Any]:
    r"""Loads the Torch serialized object at the given URL.

    If downloaded file is a zip file, it will be automatically
    decompressed.

    If the object is already present in `model_dir`, it's deserialized and
    returned.
    The default value of ``model_dir`` is ``<hub_dir>/checkpoints`` where
    ``hub_dir`` is the directory returned by :func:`~torch.hub.get_dir`.

    Args:
        url (str): URL of the object to download
        model_dir (str, optional): directory in which to save the object
        map_location (optional): a function or a dict specifying how to remap storage locations (see torch.load)
        progress (bool, optional): whether or not to display a progress bar to stderr.
            Default: True
        check_hash(bool, optional): If True, the filename part of the URL should follow the naming convention
            ``filename-<sha256>.ext`` where ``<sha256>`` is the first eight or more
            digits of the SHA256 hash of the contents of the file. The hash is used to
            ensure unique names and to verify the contents of the file.
            Default: False
        file_name (str, optional): name for the downloaded file. Filename from ``url`` will be used if not set.
        weights_only(bool, optional): If True, only weights will be loaded and no complex pickled objects.
            Recommended for untrusted sources. See :func:`~torch.load` for more details.

    Example:
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_HUB)
        >>> state_dict = torch.hub.load_state_dict_from_url(
        ...     "https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth"
        ... )

    """
    # Issue warning to move data if old env is set
    if os.getenv("TORCH_MODEL_ZOO"):
        warnings.warn(
            "TORCH_MODEL_ZOO is deprecated, please use env TORCH_HOME instead",
            stacklevel=2,
        )

    if model_dir is None:
        hub_dir = get_dir()
        model_dir = os.path.join(hub_dir, "checkpoints")

    os.makedirs(model_dir, exist_ok=True)

    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    if file_name is not None:
        filename = file_name
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        sys.stdout.write(f'Downloading: "{url}" to {cached_file}\n')
        hash_prefix = None
        if check_hash:
            r = HASH_REGEX.search(filename)  # r is Optional[Match[str]]
            hash_prefix = r.group(1) if r else None
        download_url_to_file(url, cached_file, hash_prefix, progress=progress)

    if _is_legacy_zip_format(cached_file):
        return _legacy_zip_load(cached_file, model_dir, map_location, weights_only)
    return torch.load(cached_file, map_location=map_location, weights_only=weights_only)
