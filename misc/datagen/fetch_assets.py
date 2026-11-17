#!/usr/bin/env python3
"""Download object assets for synthetic data from NVIDIA's official Isaac Sim asset server ONLY,
keeping the server's folder layout (so the USDs' relative material references resolve), and write a
manifest with URL, size and ETag per file.

    /usr/bin/python3 datagen/fetch_assets.py          # -> data/assets/isaac_6.1/... + ASSETS_MANIFEST.json
"""

from __future__ import annotations

import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUCKET = "https://omniverse-content-production.s3-us-west-2.amazonaws.com"
VERSION_PREFIX = "Assets/Isaac/6.1/"  # the asset root this Isaac Sim install is configured with
PREFIXES = ["Isaac/Props/YCB/Axis_Aligned/", "Isaac/Props/Shapes/"]
OUT = ROOT / "data" / "assets" / "isaac_6.1"
LICENCE = {
    "Isaac/Props/YCB/": "YCB object set (Calli et al., ycbbenchmarks.com; original scans commonly cited as CC BY 4.0 - "
                        "UNVERIFIED) converted to USD by NVIDIA; use governed by NVIDIA Isaac Sim asset terms (UNVERIFIED: "
                        "no licence file on the server for this folder)",
    "Isaac/Props/Shapes/": "NVIDIA Isaac Sim asset (procedural primitive); NVIDIA Isaac Sim asset terms (UNVERIFIED: no licence "
                           "file on the server for this folder)",
}


def list_keys(prefix: str) -> list[dict]:
    keys, token = [], None
    while True:
        q = {"list-type": "2", "prefix": VERSION_PREFIX + prefix, "max-keys": "1000"}
        if token:
            q["continuation-token"] = token
        xml = urllib.request.urlopen(f"{BUCKET}/?{urllib.parse.urlencode(q)}", timeout=60).read().decode()
        for block in re.findall(r"<Contents>(.*?)</Contents>", xml, re.S):
            key = re.search(r"<Key>(.*?)</Key>", block).group(1)
            if "/.thumbs/" in key or key.endswith("/"):
                continue
            keys.append({"key": key, "size": int(re.search(r"<Size>(\d+)</Size>", block).group(1)),
                         "etag": re.search(r"<ETag>(.*?)</ETag>", block).group(1).replace("&quot;", "")})
        m = re.search(r"<NextContinuationToken>(.*?)</NextContinuationToken>", xml)
        if not m:
            return keys
        token = m.group(1)


def main() -> int:
    manifest = []
    for prefix in PREFIXES:
        for item in list_keys(prefix):
            rel = item["key"][len(VERSION_PREFIX):]
            dest = OUT / rel
            if not (dest.exists() and dest.stat().st_size == item["size"]):
                dest.parent.mkdir(parents=True, exist_ok=True)
                urllib.request.urlretrieve(f"{BUCKET}/{urllib.parse.quote(item['key'])}", dest)
            manifest.append({"file": str(dest.relative_to(ROOT)), "url": f"{BUCKET}/{item['key']}", "size": item["size"],
                             "etag": item["etag"], "licence": next(v for k, v in LICENCE.items() if rel.startswith(k))})
        print(f"{prefix}: {sum(1 for m in manifest if prefix in m['file'])} files", flush=True)
    (OUT / "ASSETS_MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    print(f"{len(manifest)} files, {sum(m['size'] for m in manifest) / 1e6:.1f} MB -> {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
