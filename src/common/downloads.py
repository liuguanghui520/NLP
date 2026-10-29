"""Hash-locked transfers with a local cache, bounded retries and retained failures.

Only complete verified bytes are promoted to the requested filename. No GitHub
API token or third-party mirror is needed; a mirror is an explicit user option.
"""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import shutil
import subprocess
from urllib.request import Request, urlopen
from uuid import uuid4


def safe_path(root, relative):
    """Check portable names lexically, without resolving a file being created.

    Also reject existing symlink/junction parents. Resolving the target during
    concurrent Windows file creation can produce inconsistent extended paths.
    """
    name = PurePosixPath(relative)
    windows = PureWindowsPath(relative)
    if (not relative or "\\" in relative or name.is_absolute() or windows.drive
            or windows.root or ".." in name.parts or ":" in relative):
        raise ValueError(f"Path outside working directory: {relative}")
    root = Path(root).resolve()
    target = root.joinpath(*name.parts)
    current = root
    for part in name.parts:
        current = current / part
        if current.is_symlink() or getattr(os.path, "isjunction", lambda p: False)(current):
            raise ValueError(f"Linked path outside working directory: {relative}")
    return target


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def record(log, **event):
    if log is not None:
        log = Path(log)
        log.parent.mkdir(parents=True, exist_ok=True)
        event["time_utc"] = datetime.now(timezone.utc).isoformat()
        with log.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")


def quarantine(path):
    """Keep corrupt or incomplete bytes for diagnosis rather than overwriting."""
    path = Path(path)
    target = path.with_name(path.name + ".failed-" + uuid4().hex[:10])
    path.rename(target)
    return target


def download_verified(url, target, expected, *, expected_bytes=None,
                      cache_file=None, prefix="", log=None, attempts=2):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)

    def verify(path):
        if expected_bytes is not None and path.stat().st_size != expected_bytes:
            raise ValueError(f"Size mismatch: {target.name}")
        if sha256(path) != expected:
            raise ValueError(f"SHA256 mismatch: {target.name}")

    if target.exists():
        try:
            verify(target)
            record(log, file=target.name, transport="existing", status="verified")
            return target
        except ValueError:
            failed = quarantine(target)
            record(log, file=target.name, status="quarantined", retained=str(failed))

    if cache_file is not None and Path(cache_file).is_file():
        cache_file = Path(cache_file)
        verify(cache_file)
        shutil.copyfile(cache_file, target)
        verify(target)
        record(log, file=target.name, transport="local_cache", status="verified")
        return target

    urls = [url]
    if prefix:
        if not prefix.startswith("https://") or not prefix.endswith("/") or "@" in prefix:
            raise ValueError("--download-prefix must be an HTTPS URL ending in /, without credentials")
        urls.insert(0, prefix + url)
    curl = shutil.which("curl.exe" if os.name == "nt" else "curl")
    failures = []
    for source in urls:
        for transport in (["urllib", "curl"] if curl else ["urllib"]):
            for attempt in range(1, attempts + 1):
                partial = target.with_name(target.name + ".partial-" + uuid4().hex[:10])
                try:
                    if transport == "urllib":
                        request = Request(source, headers={"User-Agent": "nlp-course-reproduction"})
                        with urlopen(request, timeout=30) as response, partial.open("xb") as output:
                            shutil.copyfileobj(response, output, 1024 * 1024)
                    else:
                        result = subprocess.run(
                            [curl, "-fL", "--silent", "--show-error", "--connect-timeout", "20",
                             "--max-time", "600", "--output", str(partial), source],
                            capture_output=True, timeout=620)
                        if result.returncode:
                            raise RuntimeError(result.stderr.decode("utf-8", errors="replace").strip())
                    verify(partial)
                    partial.rename(target)
                    record(log, file=target.name, transport=transport, attempt=attempt,
                           mirrored=source != url, status="verified", sha256=expected)
                    return target
                except (Exception, KeyboardInterrupt) as error:
                    retained = str(quarantine(partial)) if partial.exists() else None
                    record(log, file=target.name, transport=transport, attempt=attempt,
                           mirrored=source != url, status="failed", error=str(error), retained=retained)
                    if isinstance(error, KeyboardInterrupt):
                        raise
                    failures.append(f"{transport} attempt {attempt}: {error}")
    raise RuntimeError(f"Download failed: {target.name}; " + "; ".join(failures)
                       + ". Use --input-cache for a verified checkout or supply a working HTTPS proxy.")
