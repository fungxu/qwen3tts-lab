#!/usr/bin/env python3
"""从 ModelScope 下载 Qwen3-TTS 模型（支持断点续传 + 校验）。

用法:
    python download_model.py Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice ./models/Qwen3-TTS-12Hz-1.7B-CustomVoice
"""
import json
import os
import sys
import urllib.request
import urllib.parse
import hashlib
import time

BASE = "https://modelscope.cn"
API = BASE + "/api/v1/models/{repo}/repo/files?Revision=master&Root={root}"
RAW = BASE + "/models/{repo}/resolve/master/{path}"


def list_files(repo, root=""):
    url = API.format(repo=repo, root=urllib.parse.quote(root))
    with urllib.request.urlopen(url, timeout=30) as r:
        data = json.load(r)
    return data.get("Data", {}).get("Files", [])


def walk(repo, root="", out=None):
    out = out if out is not None else []
    for f in list_files(repo, root):
        if f.get("Type") == "tree":
            walk(repo, f["Path"], out)
        else:
            out.append((f["Path"], f.get("Size", 0)))
    return out


def download(url, dest, expect_size=0):
    tmp = dest + ".part"
    have = os.path.getsize(tmp) if os.path.exists(tmp) else 0
    if expect_size and have == expect_size:
        os.replace(tmp, dest)
        return have
    headers = {"Range": f"bytes={have}-"} if have else {}
    req = urllib.request.Request(url, headers=headers)
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "ab") as fp:
        total = expect_size or int(r.headers.get("Content-Length", 0)) + have
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            fp.write(chunk)
            have += len(chunk)
            if total:
                pct = have * 100 / total
                sp = have / max(time.time() - t0, 1e-6) / 1e6
                sys.stdout.write(f"\r  {os.path.basename(dest):<28} {pct:5.1f}%  {have/1e6:8.1f}/{total/1e6:.1f} MB  {sp:5.1f} MB/s")
                sys.stdout.flush()
    sys.stdout.write("\n")
    if expect_size and have != expect_size:
        raise RuntimeError(f"size mismatch {have} != {expect_size}")
    os.replace(tmp, dest)
    return have


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    repo = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    dest_root = sys.argv[2] if len(sys.argv) > 2 else "./models/" + repo.split("/")[-1]
    files = walk(repo)
    total = sum(s for _, s in files)
    print(f"{repo}: {len(files)} 个文件, 共 {total/1e9:.2f} GB -> {dest_root}")
    os.makedirs(dest_root, exist_ok=True)
    for path, size in files:
        dest = os.path.join(dest_root, path)
        os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
        if os.path.exists(dest) and size and os.path.getsize(dest) == size:
            print(f"  跳过 {path} (已存在)")
            continue
        download(RAW.format(repo=repo, path=path), dest, size)
    # 记录校验值
    manifest = {}
    for path, size in files:
        p = os.path.join(dest_root, path)
        manifest[path] = {"size": os.path.getsize(p), "sha256": sha256(p)}
    with open(os.path.join(dest_root, "CHECKSUMS.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("完成，校验值写入 CHECKSUMS.json")


if __name__ == "__main__":
    main()
