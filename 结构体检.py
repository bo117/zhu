# -*- coding: utf-8 -*-
"""
结构体检.py —— 竹 · 写作结构体检器
========================================================
定位：**只量，不改**。不生成、不训练、不替换任何东西。
     把墨田那套"用结构说话"的度量方法抽成一个独立工具，
     随时可以给现有系统做体检。

五个指标（都是传统评估看不见的）：
  ① 篇间相似度   —— 任意两篇的 MinHash Jaccard，暴露「换汤不换药」
  ② 骨架多样性   —— 段落形状的种类数与集中度，暴露「模板化」
  ③ 句级重复率   —— 全局句子去重率，暴露「复读」
  ④ 段落数分布   —— 结构僵化程度的粗指标
  ⑤ 头尾重合     —— 篇内前 1/3 与后 1/3 的重复，暴露「首尾雷同」

用法：
  python 结构体检.py --in 训练语料_精编.jsonl
  python 结构体检.py --gen 40                 # 用 写作引擎 现场生成 40 篇
  python 结构体检.py --in a.jsonl --out 体检报告.md
"""
import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = os.path.dirname(os.path.abspath(__file__)) or "."
sys.path.insert(0, ROOT)

import numpy as np

_PUNCT = re.compile(r"[^\u4e00-\u9fff\u0041-\u005a\u0061-\u007a\u0030-\u0039]")
_SENT = re.compile(r"[。！？；\n]+")


def hanzi_len(t):
    return len(_PUNCT.sub("", t or ""))


def sentences(t, min_len=10):
    out = []
    for s in _SENT.split(t or ""):
        s = _PUNCT.sub("", s)
        if len(s) >= min_len:
            out.append(s)
    return out


def paras(t):
    return [p for p in re.split(r"\n+", t or "") if p.strip()]


def skeleton(t, max_paras=8):
    """骨架指纹：段数 + 各段相对长度（粗粒化 3 档）"""
    ps = paras(t)
    if not ps:
        return "空"
    lens = [hanzi_len(p) for p in ps]
    tot = max(1, sum(lens))
    shape = "".join(str(min(2, int(x / tot * 6))) for x in lens[:max_paras])
    return "%d|%s" % (len(ps), shape)


# ─────────────────────── 取样本 ───────────────────────
def load_jsonl(path, key=("output", "text", "content"), cap=20000):
    out = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except Exception:
                continue
            for k in key:
                if isinstance(o.get(k), str) and o[k].strip():
                    out.append(o[k])
                    break
            if len(out) >= cap:
                break
    return out


def gen_with_engine(n):
    """用 写作引擎 现场生成 n 篇（纯规则引擎，秒级）"""
    import 写作引擎 as w
    opts = w.options()
    themes = [x["name"] for x in opts["themes"]]
    endings = [e["name"] for e in opts["endings"]]
    masters = [m["name"] for m in opts["masters"]]
    rng = random.Random(20260915)
    out = []
    for i in range(n):
        th = themes[i % len(themes)]
        words = rng.choice([600, 700, 800, 900, 1000, 1200])
        q = "帮我写一篇关于%s的作文，%d字" % (th, words)
        if rng.random() < 0.4:
            q += "，结局" + rng.choice(endings)
        if rng.random() < 0.3:
            q += "，模仿" + rng.choice(masters)
        try:
            brief = w.parse_brief(q)
            brief["words"] = words
            brief["theme"] = th
            t, x = w.write_essay2(brief)
            out.append("《%s》\n\n%s" % (t, x))
        except Exception as e:
            print("  生成失败：%s" % e)
    return out


# ─────────────────────── 体检 ───────────────────────
def check(docs, name, sample=400):
    import 语料指纹 as F
    res = {"name": name, "n": len(docs)}
    if not docs:
        return res

    res["avg_zi"] = sum(hanzi_len(d) for d in docs) / len(docs)

    # ① 篇间相似度
    rng = random.Random(7)
    ss = rng.sample(docs, min(sample, len(docs)))
    h = F.MinHasher()
    sigs = [h.signature(d) for d in ss]
    sigs = [s for s in sigs if s is not None]
    if len(sigs) >= 2:
        M = np.stack(sigs)
        sims = []
        for i in range(len(M)):
            sims.append((M[i + 1:] == M[i][None, :]).sum(axis=1) / M.shape[1])
        sims = np.concatenate(sims)
        res["pair_mean"] = float(sims.mean())
        res["pair_p50"] = float(np.percentile(sims, 50))
        res["pair_p90"] = float(np.percentile(sims, 90))
        res["pair_p99"] = float(np.percentile(sims, 99))
        res["pair_max"] = float(sims.max())
        res["pair_ge30"] = float((sims >= 0.3).mean())
        res["pair_ge40"] = float((sims >= 0.4).mean())
        res["n_pairs"] = int(len(sims))

    # ② 骨架多样性
    sk = [skeleton(d) for d in docs]
    c = Counter(sk)
    res["n_skel"] = len(c)
    res["skel_top1"] = c.most_common(1)[0][1] / len(docs)
    res["skel_top3"] = sum(v for _, v in c.most_common(3)) / len(docs)
    res["skel_top5"] = sum(v for _, v in c.most_common(5)) / len(docs)
    res["skel_top_list"] = c.most_common(5)

    # ③ 句级重复率
    seen, tot = set(), 0
    for d in docs:
        for s in sentences(d):
            tot += 1
            seen.add(s)
    res["sent_total"] = tot
    res["sent_uniq"] = len(seen)
    res["sent_dup_rate"] = 1 - len(seen) / max(1, tot)

    # ④ 段落数分布
    pc = Counter(len(paras(d)) for d in docs)
    res["para_dist"] = pc.most_common(6)

    # ⑤ 头尾重合
    ov = []
    for d in docs:
        ps = paras(d)
        if len(ps) >= 3:
            k = max(1, len(ps) // 3)
            head = set(_PUNCT.sub("", p) for p in ps[:k])
            tail = set(_PUNCT.sub("", p) for p in ps[-k:])
            ov.append(len(head & tail) / max(1, len(tail)))
    res["headtail"] = sum(ov) / max(1, len(ov)) if ov else 0.0
    return res


def report(results, out_path=None):
    L = []
    A = L.append
    A("# 竹 · 写作结构体检报告")
    A("")
    A("时间：%s" % time.strftime("%Y-%m-%d %H:%M:%S"))
    A("")
    A("> 只看结构，不看文采。五个指标都是传统评估（loss/困惑度/人工翻几篇）看不见的。")
    A("")
    A("| 数据源 | 篇数 | 均字 | 篇间相似度均值 | p50 | ≥0.3 占比 | 骨架种类 | 前3骨架集中度 | 句级重复率 | 首尾重合 |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        if r.get("n", 0) == 0:
            continue
        A("| %s | %d | %.0f | **%.3f** | %.3f | **%.1f%%** | %d | **%.1f%%** | %.1f%% | %.1f%% |"
          % (r["name"], r["n"], r.get("avg_zi", 0),
             r.get("pair_mean", 0), r.get("pair_p50", 0), r.get("pair_ge30", 0) * 100,
             r.get("n_skel", 0), r.get("skel_top3", 0) * 100,
             r.get("sent_dup_rate", 0) * 100, r.get("headtail", 0) * 100))
    A("")
    for r in results:
        if r.get("n", 0) == 0:
            continue
        A("## %s（%d 篇）" % (r["name"], r["n"]))
        A("")
        if "n_pairs" in r:
            A("- 篇间相似度（%d 对）：均值 %.3f，p50 %.3f，p90 %.3f，p99 %.3f，"
              "最大 %.3f" % (r["n_pairs"], r["pair_mean"], r["pair_p50"],
                             r["pair_p90"], r["pair_p99"], r["pair_max"]))
            A("  - ≥0.3 占 %.1f%%，≥0.4 占 %.1f%%" % (r["pair_ge30"] * 100,
                                                     r["pair_ge40"] * 100))
        A("- 骨架：%d 种；最高 1 种占 %.1f%%，前 3 种占 %.1f%%，前 5 种占 %.1f%%"
          % (r["n_skel"], r["skel_top1"] * 100, r["skel_top3"] * 100,
             r["skel_top5"] * 100))
        A("  - 最常见的 5 种：")
        for k, v in r["skel_top_list"]:
            A("    - `%s` —— %d 篇（%.1f%%）" % (k, v, v / r["n"] * 100))
        A("- 句级：共 %d 句，去重后 %d 句，重复率 **%.1f%%**"
          % (r["sent_total"], r["sent_uniq"], r["sent_dup_rate"] * 100))
        A("- 段落数分布（前 6）：%s"
          % "，".join("%d段:%d篇" % (a, b) for a, b in r["para_dist"]))
        A("- 篇内首尾重合：**%.1f%%**" % (r["headtail"] * 100))
        A("")
    A("## 指标怎么读")
    A("")
    A("| 指标 | 健康 | 危险 | 说明 |")
    A("|---|---|---|---|")
    A("| 篇间相似度均值 | <0.10 | >0.20 | 任意两篇有多少内容重合 |")
    A("| 前 3 骨架集中度 | <20% | >35% | 多少文章挤在少数形状里 |")
    A("| 句级重复率 | 看池子大小 | 越高越像 | 全局有多少句子是重复的 |")
    A("| 首尾重合 | <15% | >30% | 一篇里开头和结尾说了同样的话 |")
    A("")
    txt = "\n".join(L)
    if out_path:
        with open(out_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(txt)
    return txt


def main():
    ap = argparse.ArgumentParser(description="竹 · 写作结构体检器（只量不改）")
    ap.add_argument("--in", dest="src", default=None, help="jsonl 语料")
    ap.add_argument("--gen", type=int, default=0, help="用写作引擎现场生成 N 篇")
    ap.add_argument("--gen2", type=int, default=0, help="再用写作引擎生成另一批 N 篇（看批次间重复）")
    ap.add_argument("--sample", type=int, default=400, help="篇间比对采样数")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    results = []
    if args.src:
        d = load_jsonl(args.src)
        results.append(check(d, os.path.basename(args.src), args.sample))

    if args.gen:
        t0 = time.time()
        print("用 写作引擎 生成 %d 篇……" % args.gen, flush=True)
        batch1 = gen_with_engine(args.gen)
        print("  完成（%.1f 秒）" % (time.time() - t0), flush=True)
        results.append(check(batch1, "写作引擎 · 第1批（%d 篇）" % len(batch1), args.sample))

        if args.gen2:
            print("再生成 %d 篇（第2批，用于看批次间是否互相像）……" % args.gen2, flush=True)
            batch2 = gen_with_engine(args.gen2)
            results.append(check(batch2, "写作引擎 · 第2批（%d 篇）" % len(batch2), args.sample))
            results.append(check(batch1 + batch2,
                                 "写作引擎 · 两批合并（%d 篇）" % (len(batch1) + len(batch2)),
                                 args.sample))

    if not results:
        print("没有数据源。用 --in 或 --gen。")
        return 1

    txt = report(results, args.out)
    print(txt)
    if args.out:
        print("\n[已写入 %s]" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
