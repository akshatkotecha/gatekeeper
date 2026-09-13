"""Fetches FastAPI's documentation into data/raw_docs via a sparse git clone."""
import os
import shutil
import stat
import subprocess
from pathlib import Path

REPO_URL = "https://github.com/fastapi/fastapi.git"
SPARSE_PATH = "docs/en/docs"
CLONE_DIR = Path("data/_fastapi_repo")
OUTPUT_DIR = Path("data/raw_docs")


def _force_remove(func, path, exc_info):
    # Windows marks git pack files read-only, which breaks plain rmtree.
    os.chmod(path, stat.S_IWRITE)
    func(path)


def _rmtree(path: Path):
    shutil.rmtree(path, onexc=_force_remove)


def fetch():
    if CLONE_DIR.exists():
        _rmtree(CLONE_DIR)

    subprocess.run(
        ["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", REPO_URL, str(CLONE_DIR)],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(CLONE_DIR), "sparse-checkout", "set", SPARSE_PATH],
        check=True,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    src_dir = CLONE_DIR / SPARSE_PATH

    count = 0
    for md_file in src_dir.rglob("*.md"):
        rel = md_file.relative_to(src_dir)
        # flatten subfolders: tutorial/first-steps.md -> tutorial__first-steps.md
        dest_name = str(rel).replace("/", "__").replace("\\", "__")
        dest = OUTPUT_DIR / dest_name
        dest.write_text(md_file.read_text(encoding="utf-8"), encoding="utf-8")
        count += 1

    _rmtree(CLONE_DIR)
    print(f"Fetched {count} markdown files into {OUTPUT_DIR}")


if __name__ == "__main__":
    fetch()
