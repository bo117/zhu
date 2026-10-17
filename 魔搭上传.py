# -*- coding: utf-8 -*-
"""
魔搭上传.py —— 把「竹」的训练代码与语料传到魔搭 ModelScope
================================================================================
· 代码 → 模型仓库  bg20130613/zhu-train
· 语料 → 数据集仓库 bg20130613/zhu-corpus

令牌读 魔搭凭证.json（不硬编码）。断点续传靠 SDK（已传的可跳过）。
运行：<训练venv python> 魔搭上传.py [--code-only|--data-only]
"""
import argparse
import json
import os
import time

ROOT = r"D:\竹"
NS = "bg20130613"
MR = NS + "/zhu-train"
DS = NS + "/zhu-corpus"


def log(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code-only", action="store_true")
    ap.add_argument("--data-only", action="store_true")
    args = ap.parse_args()

    tok = json.load(open(os.path.join(ROOT, "魔搭凭证.json"),
                         encoding="utf-8"))["token"]
    from modelscope.hub.api import HubApi
    api = HubApi()
    api.login(tok)
    log("已登录魔搭（namespace=%s）" % NS)

    # ── 1. 代码 → zhu-train ──
    if not args.data_only:
        pkg = os.path.join(ROOT, "魔搭发布包")
        log("上传代码包 → %s" % MR)
        try:
            api.upload_folder(repo_id=MR, folder_path=pkg, repo_type="model")
            log("  ✓ 代码上传完成")
        except Exception as e:
            log("  ✗ 代码上传失败：%s" % str(e)[:300])

    # ── 2. 语料 → zhu-corpus ──
    if not args.code_only:
        files = [
            "训练语料_外部小说.jsonl",     # 785MB / 30万条（主力）
            "训练语料_精编.jsonl",         # 29MB
            "作文范文_qa.jsonl",           # 0.8MB
            # ★ 干净集：--distill-only 的作文主料，零错别字（错字分流.py 产出）。
            #   16.6MB 的 训练语料_真人作文.jsonl 是带错字的考场原稿，
            #   云端训练必须用这份 _干净 的，否则「确认健」「再接再励」会被学走。
            "训练语料_真人作文_干净.jsonl",  # 13MB / 6518 篇（用户可见主料）
            # ★ 2026-10-14 新增三份（都是零乱码、已校验）
            #扩增：6518 篇真人作文 × 3~4 条不同角度指令 → 24039 条
            #   成文一字不改（0.4% 复读率全保留），只拉开指令维度 → 治「听不懂话」
            "训练语料_真人作文_扩增.jsonl",      # 47MB / 24039 条
            # 高考真题：题目 + 满分作文 + 阅卷评语（人工审过千万遍）
            "训练语料_高考真题_成文.jsonl",      # 86 条 / 中位 870 字
            "训练语料_高考真题_诊断.jsonl",      # 86 条 / 成文→阅卷诊断
            # 小说提纯：30 万条按文体分层（对话/心理/描写/叙事密度打分）
            #   A 叙事心理 21% 全留 / D 纯情节 31% 只留 8% → 13 万条
            #   旧做法把 4997 本只用了 1/60，金子一起被稀释了
            "训练语料_小说提纯.jsonl",           # 350MB / 130784 条
        ]
        for f in files:
            p = os.path.join(ROOT, f)
            if not os.path.exists(p):
                log("跳过（不存在）：%s" % f)
                continue
            mb = os.path.getsize(p) / 1048576
            log("上传 %s（%.0f MB）→ %s" % (f, mb, DS))
            t0 = time.time()
            try:
                api.upload_file(repo_id=DS, path_or_fileobj=p,
                                path_in_repo=f, repo_type="dataset")
                log("  ✓ 完成，用时 %.0f 秒" % (time.time() - t0))
            except Exception as e:
                log("  ✗ 失败：%s" % str(e)[:300])
        # distill 目录（API 蒸馏语料，小）
        dd = os.path.join(ROOT, "distill")
        if os.path.isdir(dd):
            log("上传 distill/ → %s" % DS)
            try:
                api.upload_folder(repo_id=DS, folder_path=dd,
                                  path_in_repo="distill", repo_type="dataset")
                log("  ✓ distill 上传完成")
            except Exception as e:
                log("  ✗ distill 失败：%s" % str(e)[:300])

    log("=== 全部完成 ===")
    log("代码仓库：https://www.modelscope.cn/models/%s" % MR)
    log("数据仓库：https://www.modelscope.cn/datasets/%s" % DS)


if __name__ == "__main__":
    main()
