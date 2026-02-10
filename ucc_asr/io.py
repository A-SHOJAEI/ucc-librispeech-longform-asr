from __future__ import annotations

import hashlib
import tarfile
from pathlib import Path
from typing import Dict, Iterable, Optional

import requests
from tqdm import tqdm


def md5_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.md5()
    with Path(path).open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def download_file(url: str, dst: str | Path, timeout_s: int = 60) -> Path:
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")

    with requests.get(url, stream=True, timeout=timeout_s) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with tmp.open("wb") as f, tqdm(
            total=total if total > 0 else None,
            unit="B",
            unit_scale=True,
            desc=f"download {dst.name}",
        ) as pbar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                pbar.update(len(chunk))
    tmp.replace(dst)
    return dst


def parse_md5sum_txt(text: str) -> Dict[str, str]:
    """
    Parses OpenSLR-style md5sum.txt lines:
      <md5> <filename>
      <md5>  <filename>   (sometimes two spaces)
    """
    out: Dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        md5, filename = parts[0], parts[-1]
        if len(md5) == 32:
            out[filename] = md5.lower()
    return out


def verify_md5(path: str | Path, expected_md5: str) -> None:
    got = md5_file(path)
    exp = expected_md5.lower()
    if got.lower() != exp:
        raise RuntimeError(f"MD5 mismatch for {path}: expected {exp}, got {got}")


def extract_tar_gz(src: str | Path, dst_dir: str | Path) -> None:
    src = Path(src)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(src, "r:gz") as tf:
        # Guard against path traversal (CVE-2007-4559 class).
        for member in tf.getmembers():
            member_path = (dst_dir / member.name).resolve()
            if not str(member_path).startswith(str(dst_dir.resolve()) + "/"):
                raise RuntimeError(f"Refusing to extract outside target dir: {member.name}")
        tf.extractall(path=dst_dir)


def ensure_download_and_verify(
    urls: Iterable[str],
    md5_url: Optional[str],
    raw_dir: str | Path,
    verify: bool = True,
) -> Dict[str, Path]:
    """
    Downloads files into raw_dir. If md5_url is provided, downloads it and verifies
    known files by filename.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    md5_map: Dict[str, str] = {}
    if md5_url:
        md5_path = raw_dir / Path(md5_url).name
        if not md5_path.exists():
            download_file(md5_url, md5_path)
        md5_text = md5_path.read_text(encoding="utf-8", errors="replace")
        md5_map = parse_md5sum_txt(md5_text)

    out: Dict[str, Path] = {}
    for url in urls:
        name = Path(url).name
        dst = raw_dir / name
        if not dst.exists():
            download_file(url, dst)
        if verify and name in md5_map:
            verify_md5(dst, md5_map[name])
        out[name] = dst
    return out
