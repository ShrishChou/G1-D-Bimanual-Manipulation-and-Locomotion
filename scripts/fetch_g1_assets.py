#!/usr/bin/env python3
"""Fetch the public Unitree G1 MJCF + meshes so the MuJoCo digital twin can build.

    python scripts/fetch_g1_assets.py

The robot model is NOT vendored in this repository. It is pulled from Google DeepMind's
MuJoCo Menagerie (BSD-3, see the LICENSE placed alongside the downloaded files) into
``assets/g1/``, which is where ``sim/scene.py`` looks by default. Set ``G1_XML`` to
override with your own copy (for example the wheeled G1-D model, which is not public).

Downloads ~42 MB of STL meshes on first run and is a no-op afterwards.
"""
from __future__ import annotations

import json
import shutil
import ssl
import subprocess
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = "google-deepmind/mujoco_menagerie"
MODEL_DIR = "unitree_g1"
DEST = Path(__file__).resolve().parents[1] / "assets" / "g1"
RAW = f"https://raw.githubusercontent.com/{REPO}/main/{MODEL_DIR}"
API = f"https://api.github.com/repos/{REPO}/contents/{MODEL_DIR}/assets"


def _ctx():
    """Prefer certifi's CA bundle; some Python installs ship without a usable one."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:  # noqa: BLE001
        return ssl.create_default_context()


def fetch(url: str) -> bytes:
    """urllib first, then curl -- a Python without a working CA bundle is common enough to plan for."""
    try:
        with urllib.request.urlopen(url, timeout=60, context=_ctx()) as r:
            return r.read()
    except Exception:  # noqa: BLE001
        if not shutil.which("curl"):
            raise
        r = subprocess.run(["curl", "-fsSL", url], capture_output=True)
        if r.returncode != 0:
            raise RuntimeError(f"curl failed for {url}: {r.stderr[:200]!r}")
        return r.stdout


def get(url: str, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size:
        return
    out.write_bytes(fetch(url))


def main() -> int:
    print(f"fetching the public Unitree G1 model into {DEST} ...")
    for name in ("g1.xml", "LICENSE"):
        get(f"{RAW}/{name}", DEST / name)

    try:
        entries = json.loads(fetch(API))
    except Exception as e:  # noqa: BLE001
        print(f"could not list the mesh directory ({e}).\n"
              f"Download {RAW}/assets manually into {DEST / 'assets'}.", file=sys.stderr)
        return 1

    meshes = [(e["download_url"], DEST / "assets" / e["name"]) for e in entries if e["type"] == "file"]
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda a: get(*a), meshes))

    xml = DEST / "g1.xml"
    if not xml.exists():
        print(f"g1.xml missing at {xml}", file=sys.stderr)
        return 1
    mb = sum(p.stat().st_size for p in (DEST / "assets").glob("*")) / 1e6
    print(f"done: {xml} + {len(meshes)} meshes ({mb:.0f} MB)")
    print("the digital twin is now runnable:  python sim/render_demo.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
