# -*- coding: utf-8 -*-
"""
魔搭同步.py —— 校验魔搭上的代码仓库是不是本地最新版本，不是就自动补传
================================================================================
一句话：
    比对本地 魔搭发布包/ 与远端 bg20130613/zhu-train 的**文件大小 + 内容指纹**，
    列出「本地改了 / 本地新增 / 远端多出」三类差异；默认**只报告不改**，
    确认要传就加 --upload。

用法（推荐只读，一次看全差异）：
    "C:\\Users\\Administrator\\Documents\\竹\\.train_venv\\Scripts\\python.exe" 魔搭同步.py

确认无误后自动补传：
    "C:\\Users\\Administrator\\Documents\\竹\\.train_venv\\Scripts\\python.exe" 魔搭同步.py --upload

只看某个文件 / 只看代码包：
    ... 魔搭同步.py --only 写作引擎.py
    ... 魔搭同步.py --code-only

说明：
· 仓库是私有的（Visibility=1），所以走 SDK 登录读远端清单，不需要裸链。
· 指纹用「文件字节 SHA256」的前 12 位 + 文件大小；远端拿不到字节时退化为按大小比。
· 不动 E 盘、不动任何训练产物，只读本地发布包、只写远端仓库。
"""
import argparse
import hashlib
import json
import os
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(ROOT, "魔搭发布包")
CRED = os.path.join(ROOT, "魔搭凭证.json")
NS = "bg20130613"
MR = NS + "/zhu-train"
REPORT = os.path.join(ROOT, "_报告", "魔搭同步结果.json")

# 只同步这些后缀；__pycache__ / *.pyc / 断点临时文件 一律忽略
SKIP_DIRS = {"__pycache__", ".ipynb_checkpoints", ".git"}
SKIP_EXT = {".pyc", ".pyo", ".tmp", ".log"}
SKIP_NAME_PREFIX = ("~$",)
SKIP_NAME = {".ms_upload_cache"}   # 上传器自己的缓存，别传上去


def norm_nl(data):
    """换行规范化：CRLF 与 LF 视为同一份内容。

    实测踩过的坑 —— 本机 Git Bash / 编辑器写出的 .py、.sh 全是 CRLF，
    魔搭上那份是 LF，同样内容大小差恰好等于 CRLF 行数（阿呆.py 差 2874 B
    = 2874 个 CRLF）。不规范化就会把 12 个没改过的文件全报成"变了"。
    """
    return data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def log(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def local_files(pkg=PKG):
    """发布包里所有要同步的文件 -> {相对路径: (绝对路径, 大小, sha256前12)}"""
    out = {}
    if not os.path.isdir(pkg):
        return out
    for base, dirs, files in os.walk(pkg):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for fn in files:
            if fn.startswith(SKIP_NAME_PREFIX) or fn in SKIP_NAME:
                continue
            ext = os.path.splitext(fn)[1].lower()
            if ext in SKIP_EXT:
                continue
            full = os.path.join(base, fn)
            rel = os.path.relpath(full, pkg).replace("\\", "/")
            try:
                data = open(full, "rb").read()
            except OSError as e:
                log("  ! 读不到 %s：%s" % (rel, str(e)[:80]))
                continue
            n = norm_nl(data)
            out[rel] = (full, len(data),
                        hashlib.sha256(n).hexdigest()[:12],
                        len(n))
    return out


def remote_files(api, repo_id=MR):
    """远端仓库文件清单 -> {相对路径: 大小}；拿不到就返回 None"""
    files = api.get_model_files(repo_id, recursive=True) or []
    out = {}
    for f in files:
        path = f.get("Path") if isinstance(f, dict) else str(f)
        size = f.get("Size") if isinstance(f, dict) else None
        if not path or int(size or 0) == 0:      # Size=0 的是目录条目
            continue
        out[str(path).replace("\\", "/")] = int(size)
    return out or None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upload", action="store_true",
                    help="确认差异后自动把本地版本传上去")
    ap.add_argument("--code-only", action="store_true",
                    help="只同步代码，不碰语料仓")
    ap.add_argument("--only", default="",
                    help="只处理指定文件（可多次用逗号分隔）")
    args = ap.parse_args()

    tok = json.load(open(CRED, encoding="utf-8"))["token"]
    from modelscope.hub.api import HubApi
    api = HubApi()
    api.login(tok)
    log("已登录魔搭（namespace=%s，仓库 %s）" % (NS, MR))

    loc = local_files()
    if args.only:
        keys = {k.strip() for k in args.only.split(",") if k.strip()}
        loc = {k: v for k, v in loc.items() if k in keys}
    log("本地发布包：%d 个文件，%.1f MB" %
        (len(loc), sum(v[1] for v in loc.values()) / 1048576))

    rem = remote_files(api)
    if rem is None:
        log("⚠ 拿不到远端文件清单（SDK 接口变了）。")
        log("  直接执行上传：加 --upload")
        if args.upload:
            api.upload_folder(repo_id=MR, folder_path=PKG, repo_type="model")
            log("  ✓ 已整包上传")
        return
    log("远端仓库：%d 个文件" % len(rem))

    changed, added, same, remote_extra = [], [], [], []
    for rel, (full, size, sha, nsize) in sorted(loc.items()):
        if rel not in rem:
            added.append(rel)
        elif rem[rel] in (size, nsize):
            # 魔搭存的 Size 可能是原始字节，也可能是换行规范化后的；
            # 两种都算一致，避免把没改过的文件误报成"变了"
            same.append(rel)
        else:
            changed.append(rel)
    for rel in rem:
        if rel not in loc:
            remote_extra.append(rel)

    print()
    print("=" * 62)
    print(" 同步前差异（代码仓 %s）" % MR)
    print("=" * 62)
    print("一致 %d / 变了 %d / 新增 %d / 远端多出 %d"
          % (len(same), len(changed), len(added), len(remote_extra)))
    if changed:
        print("\n-- 本地已改（大小与远端不一致，需上传）--")
        for r in changed:
            print("   M  %-34s 本地%d B / 远端%s B" % (r, loc[r][1], rem.get(r)))
    if added:
        print("\n-- 本地新增（远端没有，需上传）--")
        for r in added:
            print("   A  %-34s %d B" % (r, loc[r][1]))
    if remote_extra:
        print("\n-- 远端多出（本地没有，不动它）--")
        for r in remote_extra[:40]:
            print("   R  %s" % r)
    if not (changed or added):
        print("\n✓ 代码仓就是本地最新版，无需上传。")
    print()

    result = {
        "checked_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "repo": MR,
        "same": len(same), "changed": changed,
        "added": added, "remote_extra": remote_extra,
        "uploaded": False,
    }

    if args.upload and (changed or added):
        log("开始补传 %d 个文件 → %s" % (len(changed) + len(added), MR))
        ok = 0
        for rel in changed + added:
            full = loc[rel][0]
            try:
                api.upload_file(repo_id=MR, path_or_fileobj=full,
                                path_in_repo=rel, repo_type="model")
                log("  ✓ %s" % rel)
                ok += 1
            except Exception as e:
                log("  ✗ %s：%s" % (rel, str(e)[:160]))
        result["uploaded"] = True
        result["uploaded_ok"] = ok
        log("补传完成：%d/%d 成功" % (ok, len(changed) + len(added)))
        log("复核：重跑本脚本（不带 --upload）应显示「无需上传」")
    elif args.upload:
        log("无差异，未执行任何上传。")

    if not args.code_only:
        log("（本次只校验代码仓；语料仓 bg20130613/zhu-corpus 请用 魔搭上传.py --data-only 单独传）")

    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    json.dump(result, open(REPORT, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    log("报告 → %s" % REPORT)
    log("仓库页：https://www.modelscope.cn/models/%s" % MR)


if __name__ == "__main__":
    main()