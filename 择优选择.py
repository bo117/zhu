# -*- coding: utf-8 -*-
"""
择优选择.py —— 把墨田 v1.4「序数选择压」落到竹的真实产出上

【为什么要有这个模块】
墨田 v1.4（README §8.6）是一个诚实负结果：四种选择压（无生态／二进制／线性序数／
截断式序数）质量全部不爬升（Δ +0.004 ~ +0.015）。它自己给出的两条根因：

  ① 评分维度饱和 —— 流动项两轮内触顶、新颖性项被有限池枯竭拖低，
     分数被结构性地压平，任何选择压都无从区分；
  ② 规则裁判太弱 —— 质量不是纯规则可判的标量。
  结论：**生态的价值上限 = 裁判的上限。**

竹这边恰好就是它说的那一档：质检只有 overlap_check 一个**二值闸门**
（重合率 ≤0.15 放行，否则重写，最多 3 次）—— 对应 v1.4 里 Δ 只有 +0.015 的
「二进制生死簿」。没有任何梯度，也没有任何东西被记下来。

【本模块只对症根因 ①（可工程化的那条）】
  · 区分度自测：每个维度的标准差先量出来，**饱和维度自动降到 0 权**
    （v1.4 是拍死权重 0.35/0.25/0.25/0.15，两个饱和维度还占着 0.5 的权重，
      分数自然被压平 —— 这是它无区分度的直接机制）
  · 批内排名分：维度分先转成批内百分位排名再组合。**排名天然不饱和**：
    无论绝对分怎么挤在一起，排名一定铺满 0~1
  · 序数选择压：K 个候选择优，好候选胜出 —— 而不是「不合格就重写」

根因 ②（真裁判）不是本模块能解决的，见 墨田v1.4对照_裁判上限.md 的「诚实边界」。

【怎么用】
    import 择优选择 as pick
    hist = pick.Hist()
    title, text, info = pick.best_of_k(brief, k=8, mode="rank", hist=hist)
    hist.add(text)                      # 把成品纳入全局历史（新颖度的基准）

不想要择优就 k=1，行为与改造前等价。
"""

import os
import re
import sys
import math
import random

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import 写作引擎 as w

try:
    import 写作素材库 as suc
except Exception:          # 素材库缺失时退化为只用主题词
    suc = None

_norm = w._norm
_SPLIT = w._SPLIT

# 四个维度名（与墨田 v1.4 的 纯度/流动/新颖/多样 一一对应）
DIMS = ("纯度", "连贯", "新颖", "多样")

# v1.4 的原始固定权重（仅作对照基线，正式打分不用它）
V14_W = {"纯度": 0.35, "连贯": 0.25, "新颖": 0.25, "多样": 0.15}

# ── 结构磨损 κ（治择优带来的副作用）──────────────────────────────
# K 候选择优有个内生偏差：裁判隐含偏好某些结构（段数/深度），
# 于是胜者的结构签名分布会集中——A/B 实测「骨架前3集中度」从 28.2% 涨到 37.0%。
# 这是墨田 v1.2 的磨损场（O1）要治的东西，这里把它加在**结构字段**上：
#   抽样权重 w ∝ 1/(1 + κ·该字段已用次数)
# 字段级而不是签名级——骨架由段数决定，段数只由 depth/structure 决定，
# 在 1080 个完整签名上做磨损太稀，压不住边际集中。
# κ 实测（200 篇 × 2 次重复）：κ=0 前3集中度 32.2%，κ=1 → 32.2%，κ=4 → 28.2%
# （现状 k=1 是 31.5%）。κ=4 用「篇间相似度少降 0.008」换来集中度显著低于现状，
# 因为用户的主要病是骨架层模板化，取 κ=4；保守一点可以设 1.0。
STRUCT_WEAR = 4.0
_FIELD_USE = {}

_MIN_SENT = 8          # 短于此字数的句子不参与判分（与 overlap_check 同口径）


# ─────────────────────────── 主题词库 ───────────────────────────
_LEX = {}


def _grams(s, n=2):
    return {s[i:i + n] for i in range(max(0, len(s) - n + 1))}


def lexicon(theme):
    """该主题的「不跑题」词库：类别素材池切出的 2-gram + 主题词本身。

    注意：非核心主题没有专属池子，词库会退化到只有主题词，绝对纯度偏低。
    但这不影响择优 —— best_of_k 只在**同一主题的 K 个候选**之间排名，
    主题带来的基线差异会被完全抵消。
    """
    if theme in _LEX:
        return _LEX[theme]
    words = set()
    if suc is not None:
        cat = getattr(suc, "THEME_CATEGORY", {}).get(theme, theme)
        for pname in ("CAT_BITS", "CAT_OPENERS", "CAT_CLOSERS",
                      "CAT_DEPTHS", "CAT_TITLES"):
            pool = getattr(suc, pname, None) or {}
            for key in (cat, theme):
                for s in (pool.get(key) or []):
                    words |= _grams(_norm(s), 2)
    if theme:
        words.add(theme)
    _LEX[theme] = words
    return words


# ─────────────────────────── 全局历史 ───────────────────────────
class Hist(object):
    """新颖度的基准：已生成句 ∪ 范文语料句。

    用 __contains__ 而不是暴露一个 set —— 语料有 3 万+ 句，
    每篇候选都去复制一个并集会把择优的开销从毫秒推到秒。
    """

    def __init__(self, with_corpus=True):
        self.sents = set()
        self.corpus = set()
        if with_corpus:
            try:
                self.corpus = w._corpus_sentences()
            except Exception:
                self.corpus = set()

    def __contains__(self, s):
        return (s in self.sents) or (s in self.corpus)

    def add(self, text):
        for s in _SPLIT.split(text or ""):
            ns = _norm(s)
            if len(ns) >= _MIN_SENT:
                self.sents.add(ns)

    def __len__(self):
        return len(self.sents)


# ─────────────────────────── 维度分 ───────────────────────────
def dims(text, theme, hist=None):
    """把一篇真实作文量成四个维度分（都是 0~1）。

    纯度：命中主题词库的句子占比            ← 对应 v1.4 的流域纯度
    连贯：相邻句 2-gram Jaccard 均值        ← 对应 v1.4 的连接乘子（流动）
    新颖：1 - 与历史/语料重合句占比          ← 对应 v1.4 的全局新颖
    多样：篇内不同句占比                    ← 对应 v1.4 的章内多样
    """
    sents = [s for s in (_norm(x) for x in _SPLIT.split(text or ""))
             if len(s) >= _MIN_SENT]
    n = len(sents)
    if n == 0:
        return {d: 0.0 for d in DIMS}

    lex = lexicon(theme or "")
    hits = 0
    for s in sents:
        if (theme and theme in s) or (_grams(s, 2) & lex):
            hits += 1
    pur = hits / n

    nov = 1.0
    if hist is not None:
        nov = 1.0 - sum(1 for s in sents if s in hist) / n

    div = len(set(sents)) / n

    if n > 1:
        acc = 0.0
        for a, b in zip(sents, sents[1:]):
            A, B = _grams(a, 2), _grams(b, 2)
            U = A | B
            if U:
                acc += len(A & B) / len(U)
        flu = acc / (n - 1)
    else:
        flu = 0.0

    return {"纯度": pur, "连贯": flu, "新颖": nov, "多样": div}


# ─────────────────────────── 区分度与权重 ───────────────────────────
def _std(v):
    n = len(v)
    if n < 2:
        return 0.0
    m = sum(v) / n
    return math.sqrt(sum((x - m) ** 2 for x in v) / n)


def discrim(batch):
    """逐维度区分度体检：标准差 + 顶部集中度（多少篇挤在最高分 5% 内）。

    这是 v1.4 缺的那张表 —— 它只报总质量分，没报「哪一维根本没在动」。
    """
    out = {}
    n = max(1, len(batch))
    for d in DIMS:
        v = [b[d] for b in batch]
        if not v:
            out[d] = {"mean": 0.0, "std": 0.0, "max": 0.0, "ceiling": 0.0, "cv": 0.0}
            continue
        mx = max(v)
        mean = sum(v) / n
        ceiling = sum(1 for x in v if x >= mx - 0.05) / n   # 挤在顶端 5% 内
        out[d] = {"mean": mean, "std": _std(v), "max": mx,
                  "ceiling": ceiling, "cv": (_std(v) / mean if mean > 1e-9 else 0.0)}
    return out


def auto_weights(batch, floor=0.0):
    """按实测标准差定权 —— 饱和维度（std≈0）自动归零。

    v1.4 的固定权重让两个饱和维度占了 0.5 权重，这就是分数被压平的机制。
    """
    n = len(batch)
    if n < 2:
        return {d: 1.0 / len(DIMS) for d in DIMS}
    sd = {d: _std([b[d] for b in batch]) for d in DIMS}
    tot = sum(sd.values())
    if tot <= 1e-9:
        return {d: 1.0 / len(DIMS) for d in DIMS}
    raw = {d: sd[d] / tot for d in DIMS}
    if floor > 0:
        raw = {d: max(floor, raw[d]) for d in DIMS}
        s = sum(raw.values())
        raw = {d: raw[d] / s for d in DIMS}
    return raw


# ─────────────────────────── 打分 ───────────────────────────
def _ranks(vals):
    """批内百分位排名（并列取平均），值域恒为 [0,1] 且必然铺满 —— 不会饱和。"""
    n = len(vals)
    if n < 2:
        return [0.5] * n
    order = sorted(range(n), key=lambda i: vals[i])
    r = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg = ((i + j) / 2.0) / (n - 1)
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def score(batch, mode="rank", weights=None):
    """mode='rank' 批内排名分（不饱和）；mode='abs' 绝对分（对照基线）。"""
    n = len(batch)
    if n == 0:
        return [], {}
    if mode == "rank":
        ws = weights or auto_weights(batch)
        cols = {d: _ranks([b[d] for b in batch]) for d in DIMS}
    else:
        ws = weights or dict(V14_W)
        cols = {d: [b[d] for b in batch] for d in DIMS}
    return [sum(ws[d] * cols[d][i] for d in DIMS) for i in range(n)], ws


# ─────────────────────────── K 候选择优 ───────────────────────────
def _rotate_wear(brief, rng):
    """结构与写作引擎的 rotate_structure 同义（尊重 _explicit、尊重折返期），
    但抽样权重带上字段级磨损，抵消择优对某些结构的隐含偏好。"""
    b = dict(brief)
    ex = brief.get("_explicit") or {}
    pools = (("opening", (brief.get("opening"),) if ex.get("opening") else tuple(w.OPENINGS)),
             ("structure", (brief.get("structure"),) if ex.get("structure") else tuple(w.STRUCTURES)),
             ("technique", (brief.get("technique"),) if ex.get("technique") else (None,) + tuple(w.TECHNIQUES)),
             ("depth", (brief.get("depth"),) if ex.get("depth") else ("快速", "标准", "深度")))
    for _ in range(40):
        for field, opts in pools:
            ws = [1.0 / (1.0 + STRUCT_WEAR * _FIELD_USE.get((field, str(o)), 0))
                  for o in opts]
            tot = sum(ws)
            r, acc, pickv = rng.random() * tot, 0.0, opts[-1]
            for o, wt in zip(opts, ws):
                acc += wt
                if r <= acc:
                    pickv = o
                    break
            b[field] = pickv
        if w._struct_sig(b) not in w._STRUCT_HIST:
            return b
    return b


def _gen_one(brief, rng):
    """抽一个没撞冷却期的结构，然后写一篇。

    结构折返期只在「抽」这一步开，写的时候关掉，
    这样历史里只会留下本轮胜者的签名，K 个候选不会互相挤掉冷却位。
    """
    if STRUCT_WEAR > 0:
        b = _rotate_wear(brief, rng)
    else:
        w.STRUCT_ROTATE = True
        b = w.rotate_structure(brief, rng)
    w.STRUCT_ROTATE = False
    t, x = w.write_essay2(b)
    return b, t, x


def reset_struct_state():
    """清空折返期历史与磨损计数（实验对照用）"""
    w._STRUCT_HIST.clear()
    _FIELD_USE.clear()


_GLOBAL_HIST = None


def global_hist():
    """进程内共用的新颖度基准。生产里择优要跨篇记忆才有效——
    墨田 v1.3 的结论就是「治模板化的药方在结构上要求全局历史，滑窗看不见它」。"""
    global _GLOBAL_HIST
    if _GLOBAL_HIST is None:
        _GLOBAL_HIST = Hist()
    return _GLOBAL_HIST


def best_of_k(brief, k=8, mode="rank", hist=None, weights=None):
    """生成 K 个候选，按裁判排名取最好的一个；只让胜者进折返期历史。

    返回 (title, text, info)。k=1 时行为与改造前等价（可一行回退）。
    hist=None 时自动用进程内共用的 global_hist()，并把成品记进去。
    """
    auto_hist = hist is None
    if auto_hist:
        hist = global_hist()
    base = list(w._STRUCT_HIST)
    cands = []
    rng = random.Random(random.random())
    try:
        for _ in range(max(1, k)):
            w._STRUCT_HIST[:] = list(base)      # 每个候选都从同一冷却状态出发
            b, t, x = _gen_one(brief, rng)
            cands.append((b, t, x, dims(x, b.get("theme") or brief.get("theme"), hist)))
    finally:
        w.STRUCT_ROTATE = True
        w._STRUCT_HIST[:] = list(base)          # 候选不污染历史

    scores, ws = score([c[3] for c in cands], mode=mode, weights=weights)
    best = max(range(len(cands)), key=lambda i: scores[i])
    bb, bt, bx, bd = cands[best]

    w._struct_push(w._struct_sig(bb))           # 只提交胜者的结构签名
    for field in ("opening", "structure", "technique", "depth"):
        key = (field, str(bb.get(field)))
        _FIELD_USE[key] = _FIELD_USE.get(key, 0) + 1
    if auto_hist:
        hist.add(bx)                            # 成品纳入全局历史
    info = {"k": len(cands), "mode": mode, "chosen": best,
            "scores": scores, "weights": ws, "dims": bd,
            "chosen_sig": w._struct_sig(bb),
            "dim_table": [c[3] for c in cands],
            "wrote_structures": [w._struct_sig(c[0]) for c in cands]}
    return bt, bx, info


__all__ = ["DIMS", "V14_W", "STRUCT_WEAR", "Hist", "global_hist", "lexicon",
           "dims", "discrim", "auto_weights", "score", "best_of_k",
           "reset_struct_state"]
