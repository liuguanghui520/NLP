"""Upload verified packages to a GitHub Release; never print or persist credentials.

Authentication uses the existing Git credential helper for github.com. Requests
are restricted to GitHub API/upload hosts, with authorization removed on redirects.
"""

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request


ROOT = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def existing_github_credential():
    """Use the authorized service's normal helper; secret values stay in memory."""
    env = os.environ.copy()
    env.update(GIT_TERMINAL_PROMPT="0", GCM_INTERACTIVE="Never")
    process = subprocess.run(["git", "credential", "fill"],
                             input="protocol=https\nhost=github.com\n\n",
                             capture_output=True, text=True, env=env)
    if process.returncode:
        raise RuntimeError("Existing GitHub credential helper is unavailable; log in normally before retrying")
    values = dict(line.split("=", 1) for line in process.stdout.splitlines() if "=" in line)
    token = values.get("password")
    if not token:
        raise RuntimeError("GitHub credential helper did not provide authentication")
    return token


class SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parsed = urllib.parse.urlparse(newurl)
        if parsed.scheme != "https":
            raise ValueError("Refusing a non-HTTPS redirect")
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None:
            redirected.remove_header("Authorization")
        return redirected


class Reader:
    """Streaming request body with a client-side digest and bounded progress logs."""
    def __init__(self, stream, name, size, event):
        self.stream, self.name, self.size, self.event = stream, name, size, event
        self.count = 0
        self.digest = hashlib.sha256()
        self.last = time.monotonic()

    def read(self, size=-1):
        block = self.stream.read(size)
        self.count += len(block)
        self.digest.update(block)
        if time.monotonic() - self.last >= 10:
            self.event("upload_progress", name=self.name, sent_bytes=self.count, total_bytes=self.size)
            self.last = time.monotonic()
        return block


class GitHub:
    def __init__(self, token, proxy, repo):
        self.token, self.repo = token, repo
        handlers = [SafeRedirect()]
        if proxy:
            handlers.append(urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
        self.opener = urllib.request.build_opener(*handlers)

    def request(self, method, url, *, payload=None, body=None, headers=None, auth=True, timeout=300):
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme != "https":
            raise ValueError("HTTPS is required")
        values = {"Accept": "application/vnd.github+json", "User-Agent": "NLP-trained-artifact-publisher",
                  "X-GitHub-Api-Version": "2022-11-28"}
        if auth:
            if parsed.hostname not in {"api.github.com", "uploads.github.com"}:
                raise ValueError("Authentication is restricted to GitHub API hosts")
            values["Authorization"] = "Bearer " + self.token
        if headers:
            values.update(headers)
        if payload is not None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            values["Content-Type"] = "application/json"
        request = urllib.request.Request(url, data=body, headers=values, method=method)
        return self.opener.open(request, timeout=timeout)

    def json(self, method, suffix, payload=None, missing_ok=False):
        url = f"https://api.github.com/repos/{self.repo}" + ("/" + suffix if suffix else "")
        try:
            with self.request(method, url, payload=payload) as response:
                if response.status == 204:
                    return None
                return json.load(response)
        except urllib.error.HTTPError as error:
            if missing_ok and error.code == 404:
                return None
            try:
                message = json.loads(error.read()).get("message", "API request failed")
            except (ValueError, AttributeError):
                message = "API request failed"
            raise RuntimeError(f"GitHub HTTP {error.code}: {message}") from None

    def remote_digest(self, asset, expected, event):
        """Prefer GitHub's computed digest; otherwise verify the download stream."""
        digest = asset.get("digest")
        if digest:
            if digest != "sha256:" + expected:
                raise ValueError(f"GitHub digest mismatch: {asset['name']}")
            return "github_server_sha256"
        url = f"https://api.github.com/repos/{self.repo}/releases/assets/{asset['id']}"
        actual = hashlib.sha256()
        count = 0
        event("download_hash_verification", name=asset["name"])
        with self.request("GET", url, headers={"Accept": "application/octet-stream"}) as response:
            for block in iter(lambda: response.read(8 * 1024 * 1024), b""):
                actual.update(block)
                count += len(block)
        if actual.hexdigest() != expected or count != asset["size"]:
            raise ValueError(f"Downloaded asset integrity mismatch: {asset['name']}")
        return "authenticated_download_sha256"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="liuguanghui520/NLP")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--target", required=True, help="Exact existing commit SHA")
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--notes", type=Path, required=True)
    parser.add_argument("--state-directory", type=Path, required=True)
    parser.add_argument("--proxy")
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", args.repo):
        parser.error("Invalid repository")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.tag) or not re.fullmatch(r"[a-f0-9]{40}", args.target):
        parser.error("Use a portable tag and a full commit SHA")
    directory = args.directory.resolve()
    state_dir = args.state_directory.resolve()
    if not state_dir.is_relative_to(ROOT) or state_dir == ROOT:
        parser.error("Store the upload receipt under the current project")
    state_dir.mkdir(parents=True, exist_ok=True)
    status_path = state_dir / "status.json"
    state = load(status_path) if status_path.exists() else dict(status="preflight", assets=[], created_asset_ids=[])
    attempt = 1 + len(list(state_dir.glob("attempt-*.json")))
    state.update(repo=args.repo, tag=args.tag, target=args.target, attempt=attempt,
                 started_at=now(), training=False, model_loading=False, inference=False)

    def event(kind, **fields):
        item = dict(at=now(), event=kind, **fields)
        with (state_dir / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(json.dumps(item, ensure_ascii=True), flush=True)
        state["last_event"] = item
        save(status_path, state)

    try:
        catalog = load(directory / "catalog.json")
        if catalog["status"] != "completed" or len(catalog["artifacts"]) != 9:
            raise ValueError("Expected a completed nine-artifact package catalog")
        files = []
        for row in catalog["artifacts"]:
            name = row["archive"]
            if Path(name).name != name or not name.endswith(".zip"):
                raise ValueError("Invalid package filename")
            path = directory / name
            if path.stat().st_size != row["archive_bytes"] or sha256(path) != row["archive_sha256"]:
                raise ValueError(f"Local package changed: {name}")
            files.append((path, row["archive_sha256"]))
        for name in ["README.md", "SHA256SUMS", "catalog.json"]:
            files.append((directory / name, sha256(directory / name)))
        event("local_files_verified", files=len(files), bytes=sum(p.stat().st_size for p, _ in files))
        api = GitHub(existing_github_credential(), args.proxy, args.repo)
        repo_info = api.json("GET", "")
        if not repo_info.get("permissions", {}).get("push"):
            raise PermissionError("Existing GitHub login lacks repository write permission")
        release = api.json("GET", "releases/tags/" + args.tag, missing_ok=True)
        if release is None:
            release = next((item for item in api.json("GET", "releases?per_page=100") if item["tag_name"] == args.tag), None)
        if release is None:
            release = api.json("POST", "releases", dict(tag_name=args.tag, target_commitish=args.target,
                               name="中文情感分析训练产物 · " + args.tag,
                               body=args.notes.read_text(encoding="utf-8"), draft=True, prerelease=False))
            state["release_id"] = release["id"]
            event("draft_created", release_id=release["id"], url=release["html_url"])
        elif state.get("release_id") != release["id"]:
            raise ValueError("An existing release is not owned by this upload receipt")
        state["release_id"] = release["id"]
        state["release_url"] = release["html_url"]
        save(status_path, state)
        upload_url = release["upload_url"].split("{", 1)[0]
        if urllib.parse.urlparse(upload_url).hostname != "uploads.github.com":
            raise ValueError("Unexpected GitHub upload host")
        for position, (path, expected) in enumerate(files, 1):
            assets = api.json("GET", f"releases/{release['id']}/assets?per_page=100")
            existing = next((asset for asset in assets if asset["name"] == path.name), None)
            if existing is not None and existing["state"] != "uploaded":
                if existing["id"] not in state["created_asset_ids"]:
                    raise ValueError(f"Existing unfinished asset is not owned by this receipt: {path.name}")
                api.json("DELETE", f"releases/assets/{existing['id']}")
                existing = None
            if existing is None:
                event("upload_started", index=position, count=len(files), name=path.name, bytes=path.stat().st_size)
                url = upload_url + "?" + urllib.parse.urlencode({"name": path.name})
                try:
                    with path.open("rb") as stream:
                        reader = Reader(stream, path.name, path.stat().st_size, event)
                        with api.request("POST", url, body=reader,
                                         headers={"Content-Type": "application/octet-stream", "Content-Length": str(path.stat().st_size)},
                                         timeout=900) as response:
                            existing = json.load(response)
                    if reader.count != path.stat().st_size or reader.digest.hexdigest() != expected:
                        raise ValueError("Client upload stream differs from the expected file")
                except BaseException:
                    latest = api.json("GET", f"releases/{release['id']}/assets?per_page=100")
                    failed = next((asset for asset in latest if asset["name"] == path.name), None)
                    if failed is not None and failed["id"] not in state["created_asset_ids"]:
                        state["created_asset_ids"].append(failed["id"])
                    save(status_path, state)
                    raise
                state["created_asset_ids"].append(existing["id"])
            if existing["size"] != path.stat().st_size:
                raise ValueError(f"Remote asset size mismatch: {path.name}")
            verified_by = api.remote_digest(existing, expected, event)
            record = dict(name=path.name, asset_id=existing["id"], bytes=existing["size"],
                          sha256=expected, server_digest=existing.get("digest"),
                          verified_by=verified_by, download_url=existing["browser_download_url"])
            state["assets"] = [a for a in state["assets"] if a["name"] != path.name] + [record]
            event("asset_verified", **record)
        if len(state["assets"]) != len(files):
            raise ValueError("Asset verification is incomplete")
        if args.publish:
            release = api.json("PATCH", f"releases/{release['id']}", dict(draft=False))
            state.update(published=not release["draft"], release_url=release["html_url"])
            event("release_published", url=release["html_url"])
            published_assets = {asset["id"]: asset for asset in api.json("GET", f"releases/{release['id']}/assets?per_page=100")}
            for asset in state["assets"]:
                asset["download_url"] = published_assets[asset["asset_id"]]["browser_download_url"]
                with api.request("GET", asset["download_url"], auth=False,
                                 headers={"Range": "bytes=0-0", "Accept": "application/octet-stream"}) as response:
                    first = response.read(1)
                    if len(first) != 1 or response.status not in (200, 206):
                        raise ValueError(f"Anonymous download probe failed: {asset['name']}")
                    asset["anonymous_get_status"] = response.status
                    asset["content_range"] = response.headers.get("Content-Range")
                event("anonymous_download_verified", name=asset["name"])
        state.update(status="completed", completed_at=now())
        save(status_path, state)
        save(state_dir / f"attempt-{attempt:02d}.json", state)
        return 0
    except BaseException as error:
        # Never serialize request objects, headers or the in-memory credential.
        state.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                     error=type(error).__name__ + ": " + str(error), completed_at=now())
        save(status_path, state)
        save(state_dir / f"attempt-{attempt:02d}.json", state)
        event("upload_failed", error=state["error"])
        return 130 if isinstance(error, KeyboardInterrupt) else 1


if __name__ == "__main__":
    raise SystemExit(main())
