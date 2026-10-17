# -*- coding: utf-8 -*-
"""
语料指纹.py —— 竹 · 语料指纹与近重查重引擎
============================================================
一句话：给每篇作文算一枚"指纹"，指纹像的就算重复，从语料里剔掉。

为什么不用最笨的两两比对：
  3 万篇两两比 = 4.5 亿对，纯 CPython 要跑几小时。
  这里走 MinHash + LSH 两段式，把"像不像"压到"可能像的少数候选"再精算，
  3 万篇量级从"小时级"降到"分钟级"，且不吃内存（流式入库）。

指纹怎么算：
  1) 归一化：去标点、空白，只留汉字/字母/数字
  2) 取 k 元组（默认 k=3，即连续 3 个字）当特征
  3) MinHash：用 n 个随机哈希把特征集压成 n 维签名
     —— 两个签名逐位相等的比例 ≈ 两个集合的 Jaccard 相似度（无偏估计）
  4) LSH 分带：签名切成 b 带 × r 行，命中同一带即入候选（近重的先筛出来）
  5) 候选里精算签名相等率，超过阈值判重

阈值经验值（在 训练语料_3万.jsonl 上实测校准）：
  无关篇目相似度中位数 0.42、p99 0.75、最大 0.89 —— 这批模板稿本身就高度同源
  >= 0.90  几乎同文
  >= 0.80  同模板套壳（保守线，上一版报告用的是 0.84）
  >= 0.72  同模板同句式大面积重合（"尽量删"推荐线）
  >= 0.65  更激进，收益递减且有误杀风险

用法：
  python 语料指纹.py --in 训练语料_3万.jsonl --out 训练语料_精编.jsonl --threshold 0.72
  python 语料指纹.py --selfcheck
"""
import argparse
import json
import os
import re
import sys
import time
import zlib
from collections import Counter

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np

# ─────────────────────── 归一化 / 切句 ───────────────────────
_PUNCT = re.compile(
    r"[，。！？；：、“”‘’（）《》【】〈〉「」『』·…—－\-_~～\s"
    r",.;:!?\"'()\[\]{}<>/\\|@#$%^&*+=`]"
)
_SENT_SPLIT = re.compile(r"[。！？；\n]+")


def normalize(s):
    """去标点空白，只留实字"""
    return _PUNCT.sub("", s or "")


def sentences(text, min_len=8):
    """切句，太短的丢掉（短句做指纹噪声大）"""
    out = []
    for s in _SENT_SPLIT.split(text or ""):
        s = s.strip()
        if len(normalize(s)) >= min_len:
            out.append(s)
    return out


def sent_key(s):
    """句级指纹键：归一化后 32 位哈希，用于算句级重合率"""
    n = normalize(s)
    if len(n) < 10:
        return None
    return zlib.crc32(n.encode("utf-8", "ignore"))


# ─────────────────────── shingle / MinHash ───────────────────────
def shingles(text, k=3):
    """k 元组特征集（字符级，中文友好）"""
    t = normalize(text)
    if not t:
        return ()
    if len(t) <= k:
        return (t,)
    return {t[i:i + k] for i in range(len(t) - k + 1)}


def _h64(s):
    """稳定 64 位哈希（crc32 拼双段，跨进程一致，不用内置 hash）"""
    b = s.encode("utf-8", "ignore")
    return (zlib.crc32(b) | (zlib.crc32(b, 0x1F123BB5) << 32)) & 0xFFFFFFFFFFFFFFFF


class MinHasher:
    """签名生产机器。A/B 用固定种子生成，保证同语料得到同指纹。"""

    def __init__(self, n=64, k=3, seed=20260915):
        self.n = n
        self.k = k
        rng = np.random.default_rng(seed)
        # 通用哈希 h(x) = (x*a + b) mod 2^64（uint64 自然回绕）
        self.A = (rng.integers(1, 1 << 62, size=n, dtype=np.uint64) * np.uint64(2)
                  + np.uint64(1))
        self.B = rng.integers(0, 1 << 62, size=n, dtype=np.uint64)
        self.A[0] = np.uint64(1)
        self.B[0] = np.uint64(0)          # 第一路退化成恒等，便于调试

    def signature(self, text):
        """返回 (n,) uint64 签名；空文本返回 None"""
        sh = shingles(text, self.k)
        if not sh:
            return None
        arr = np.fromiter((_h64(s) for s in sh), dtype=np.uint64, count=len(sh))
        # (m,n) 广播：m 个特征 × n 路哈希，取每路最小值
        mat = arr[:, None] * self.A[None, :] + self.B[None, :]
        return mat.min(axis=0)

    @staticmethod
    def similarity(sig_a, sig_b):
        """签名相等率 ≈ Jaccard 相似度"""
        if sig_a is None or sig_b is None:
            return 1.0 if sig_a is None and sig_b is None else 0.0
        return float(np.count_nonzero(sig_a == sig_b)) / len(sig_a)


# ─────────────────────── LSH 索引 ───────────────────────
class CorpusIndex:
    """
    流式指纹库。add() 入一篇，query() 取候选。
    每带 r 行、共 b 带；r 越大越严格，b 越大召回越高。
    用命中带数给候选排序并截断（max_cand），避免模板语料把候选集撑爆。
    """

    def __init__(self, hasher=None, bands=8, rows=8):
        self.hasher = hasher or MinHasher()
        assert bands * rows == self.hasher.n, "bands*rows 必须等于签名长度"
        self.bands = bands
        self.rows = rows
        self.tables = [dict() for _ in range(bands)]
        self.sigs = []          # 顺序保存签名
        self.metas = []         # 顺序保存附带信息
        self.keysets = []       # 句级指纹集合（算句级重合率用）
        self.count = 0

    def _band_keys(self, sig):
        r = self.rows
        for i in range(self.bands):
            yield i, zlib.crc32(sig[i * r:(i + 1) * r].tobytes())

    def add(self, text, meta=None):
        sig = self.hasher.signature(text)
        idx = self.count
        self.sigs.append(sig)
        self.metas.append(meta)
        self.keysets.append(
            {k for k in (sent_key(s) for s in sentences(text)) if k is not None}
        )
        if sig is not None:
            for b, key in self._band_keys(sig):
                self.tables[b].setdefault(key, []).append(idx)
        self.count += 1
        return idx

    def query(self, text, threshold=0.8, max_cand=256):
        """返回 (最佳相似度, 命中的 meta 或 None, 最佳候选下标)"""
        sig = self.hasher.signature(text)
        if sig is None:
            return 0.0, None, None
        hits = Counter()
        for b, key in self._band_keys(sig):
            for i in self.tables[b].get(key, ()):      # 每带命中都给候选加分
                hits[i] += 1
        if not hits:
            return 0.0, None, None
        cands = [i for i, _ in hits.most_common(max_cand)]
        best_sim, best_i = 0.0, None
        s_all = np.stack([self.sigs[i] for i in cands])
        eq = (s_all == sig[None, :]).sum(axis=1) / len(sig)
        k = int(eq.argmax())
        best_sim, best_i = float(eq[k]), cands[k]
        if best_sim >= threshold:
            return best_sim, self.metas[best_i], best_i
        return best_sim, None, best_i

    def sentence_overlap(self, text, idx):
        """新稿与某篇已库稿的句级重合率（越低越新鲜）"""
        cur = {k for k in (sent_key(s) for s in sentences(text)) if k is not None}
        if not cur or idx is None or idx >= len(self.keysets):
            return 0.0
        return len(cur & self.keysets[idx]) / len(cur)


# ─────────────────────── 去重主流程 ───────────────────────
def _iter_jsonl(path, text_keys=("output", "text", "content")):
    with open(path, encoding="utf-8", errors="replace") as f:
        for no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            text = ""
            for k in text_keys:
                if isinstance(obj.get(k), str) and obj[k].strip():
                    text = obj[k]
                    break
            yield no, obj, text


def _title_of(text):
    m = re.match(r"\s*《(.+?)》", text)
    if m:
        return m.group(1)
    return text.strip().split("\n", 1)[0][:24]


def dedupe(src, dst, threshold=0.72, hasher=None, log_every=4000,
           report=None):
    """
    流式去重：与"已保留库"近重的丢弃，不重复的留下。
    返回统计 dict（含最像的前 20 条丢弃样例）。
    """
    index = CorpusIndex(hasher or MinHasher())
    t0 = time.time()
    seen = kept = 0
    dropped = []
    with open(dst, "w", encoding="utf-8", newline="\n") as out:
        for no, obj, text in _iter_jsonl(src):
            seen += 1
            if not text.strip():
                continue
            sim, meta, hit = index.query(text, threshold=threshold)
            if meta is not None:
                dropped.append((no, round(sim, 3), meta.get("title"),
                                round(index.sentence_overlap(text, hit), 3)))
                continue
            index.add(text, {"no": no, "title": _title_of(text)})
            out.write(json.dumps(obj, ensure_ascii=False) + "\n")
            kept += 1
            if log_every and seen % log_every == 0:
                print("    … 已读 %d，保留 %d，去重 %d，用时 %.1f 分钟"
                      % (seen, kept, len(dropped), (time.time() - t0) / 60), flush=True)
    stat = {
        "src": src, "dst": dst, "seen": seen, "kept": kept,
        "dropped": len(dropped), "threshold": threshold,
        "rate": round(len(dropped) / max(1, seen) * 100, 1),
        "seconds": round(time.time() - t0, 1),
        "samples": sorted(dropped, key=lambda x: -x[1])[:20],
    }
    if report:
        with open(report, "w", encoding="utf-8", newline="\n") as f:
            json.dump({k: v for k, v in stat.items() if k != "samples"},
                      f, ensure_ascii=False, indent=1)
    return stat


# ─────────────────────── 自检 ───────────────────────
def selfcheck():
    """不依赖任何文件的最小验证：同源改写要像、无关文本要不像"""
    h = MinHasher()
    a = ("冬天的清晨，屋檐下挂着细长的冰凌，风一吹，就叮叮当当地响。"
         "外婆蹲在灶台前添柴，火光把她的影子拉得很长，投在斑驳的土墙上。")
    b = a.replace("细长", "细瘦").replace("叮叮当当", "叮叮咚咚").replace("外婆", "奶奶")
    c = "端午的粽子要泡米，粽叶先在开水里滚过，糯米才带得住香气，五花肉要腌足一夜。"
    sa, sb, sc = h.signature(a), h.signature(b), h.signature(c)
    s_ab = MinHasher.similarity(sa, sb)
    s_ac = MinHasher.similarity(sa, sc)
    print("自检：同源改写相似度 = %.2f（应 >0.55）" % s_ab)
    print("自检：无关文本相似度 = %.2f（应 <0.35）" % s_ac)
    ok = s_ab > 0.55 and s_ac < 0.35
    print("自检", "通过" if ok else "未通过")
    return ok


def main():
    ap = argparse.ArgumentParser(description="语料指纹 / 近重查重")
    ap.add_argument("--in", dest="src", help="输入 jsonl")
    ap.add_argument("--out", dest="dst", help="输出 jsonl（保留下来的）")
    ap.add_argument("--threshold", type=float, default=0.72)
    ap.add_argument("--report", default=None, help="统计写到这里（json）")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()

    if args.selfcheck or not args.src:
        return 0 if selfcheck() else 1
    st = dedupe(args.src, args.dst, threshold=args.threshold, report=args.report)
    print("读 %d 篇 → 保留 %d 篇，去近重 %d 篇（%.1f%%），用时 %.1f 分钟"
          % (st["seen"], st["kept"], st["dropped"], st["rate"], st["seconds"] / 60))
    return 0


if __name__ == "__main__":
    sys.exit(main())
