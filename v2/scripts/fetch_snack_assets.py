"""Fetch the kitchen subset of NVIDIA's Residential USD pack, including textures and licence.

Run from v2: .venv-seg/bin/python scripts/fetch_snack_assets.py
Downloads about 122 MB using HTTP byte ranges, rather than the whole 22.9 GB archive.
Downloaded content stays in the shared, ignored data/assets/nvidia_residential directory.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "assets/snack_kitchen/nvidia_manifest.json"
DESTINATION = ROOT / "data/assets/nvidia_residential"
CHUNK = 2 * 1024 * 1024


class RemoteArchive(io.RawIOBase):
    """Seekable read-only ZIP input; reject servers that ignore byte-range requests."""

    def __init__(self, url: str, size: int):
        self.url, self.size, self.position = url, size, 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        if self.position < 0:
            raise ValueError("negative archive position")
        return self.position

    def read(self, size=-1):
        remaining = self.size - self.position
        count = remaining if size < 0 else min(size, remaining)
        chunks = []
        while count > 0:
            length = min(count, CHUNK)
            end = self.position + length - 1
            request = Request(self.url, headers={"Range": f"bytes={self.position}-{end}"})
            for attempt in range(3):
                try:
                    with urlopen(request, timeout=45) as response:
                        expected = f"bytes {self.position}-{end}/{self.size}"
                        if response.status != 206 or response.headers.get("Content-Range") != expected:
                            raise RuntimeError("server did not honor the bounded archive range")
                        content = response.read(length + 1)
                    if len(content) != length:
                        raise RuntimeError("incomplete archive range")
                    break
                except OSError:
                    if attempt == 2:
                        raise
                    time.sleep(attempt + 1)
            self.position += length
            count -= length
            chunks.append(content)
        return b"".join(chunks)


def valid(path: Path, spec: dict) -> bool:
    if not path.is_file() or path.stat().st_size != spec["bytes"]:
        return False
    return not spec.get("sha256") or hashlib.sha256(path.read_bytes()).hexdigest() == spec["sha256"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="verify local files without network access")
    args = ap.parse_args(argv)
    manifest = json.loads(MANIFEST.read_text())
    pending = [spec for spec in manifest["files"] if not valid(DESTINATION / spec["path"], spec)]
    if args.check:
        for spec in pending:
            print(f"missing or invalid: {spec['path']}")
        print(f"[snack assets] {len(manifest['files']) - len(pending)}/{len(manifest['files'])} files verified")
        return int(bool(pending))
    if not pending:
        print("[snack assets] all files already verified")
        return 0
    with zipfile.ZipFile(RemoteArchive(manifest["archive_url"], manifest["archive_bytes"])) as archive:
        for number, spec in enumerate(pending, 1):
            path = DESTINATION / spec["path"]
            if not path.resolve().is_relative_to(DESTINATION.resolve()):
                raise ValueError("asset path leaves the download directory")
            info = archive.getinfo(spec["path"])
            if info.file_size != spec["bytes"] or info.file_size > 40 * 1024 * 1024:
                raise ValueError(f"unexpected asset size: {spec['path']}")
            content = archive.read(info)
            if spec.get("sha256") and hashlib.sha256(content).hexdigest() != spec["sha256"]:
                raise ValueError(f"asset checksum mismatch: {spec['path']}")
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name + ".partial")
            temporary.write_bytes(content)
            temporary.replace(path)
            print(f"[snack assets] {number}/{len(pending)} {spec['path']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
