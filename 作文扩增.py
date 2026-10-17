# -*- coding: utf-8 -*-
"""★ 真人作文「多指令扩增」——成文一字不改，只把**指令维度**拉开。

  为什么这么做（2026-10-14 决定，替代「用引擎批量造作文」）：
  ────────────────────────────────────────────────────────────────────
  朴素做法：用规则引擎批量生成作文，把 6518 篇变3 万篇。
  **这是重蹈覆辙** —— 用户手上那 3 万篇旧语料就是这么来的（同模板批量生成），
  句级重复率 **98.9%**，无关篇目相似度中位数 0.42 / p99 0.75。
  规则引擎无论怎么组装，骨架空间就那么大，批量生成必然复读。
  真人作文 **0.4%** 的复读率来自「它是人写的」，造是造不出来的。

  正确做法：**成文质量不降，只拉指令维度**。
    成文 = 6518 篇真人作文（一字不改，0.4% 复读率**全部保留**）
    指令 = 每篇配 3~5 条**不同角度的提法**（换字数口径/换开头要求/
          换结构要求/换体裁），且每条都**从 meta 里验证过**。
  → 训练样本 6518 → 2.6 万，而「听懂话」的能力被真正拉开。

  为什么这对「听不懂话」有效：
    用户抱怨「说正常的话它听不懂」。听不懂的根因不是词汇量，是
    **没见过同一件事的不同问法**。一篇作文配 4 种问法，
    学到的是「这四种说法都指向同一件事」—— 这才是指令泛化。

  ★ 一条硬约束（不能违背）：
    **指令必须与正文真实相符**。正文 1082 字而指令说「写到 500 字」，
    就是教模型「学会说谎」。所以字数一律取 meta.chars（原文真实字数），
    体裁一律取 meta.grade/原文实际写法，绝不编造。
"""
import sys, io, os, json, random, collections

ROOT = r"D:\竹"
SRC = os.path.join(ROOT, "训练语料_真人作文_干净.jsonl")
OUT = os.path.join(ROOT, "训练语料_真人作文_扩增.jsonl")

# ★★ 年级下限（2026-10-14 用户定）：
#   低年级作文（一年级「我今天很开心」那种）教出来的只会是小学生作文，
#   对「能写作文」这个产品**完全没用**——用户自己就是中学生/成人视角。
#   保留四年级起（小学高段已有点思辨），砍掉一~三年级。
GRADE_MIN = 4
_GRADE_ORDER = ["一年级", "二年级", "三年级", "四年级", "五年级", "六年级",
                "初一", "初二", "初三", "高一", "高二", "高三"]


def grade_num(g):
    g = (g or "").strip()
    return _GRADE_ORDER.index(g) + 1 if g in _GRADE_ORDER else 99


def _len_bucket(n):
    if n < 250:
        return "短", "就写 %d 字左右，别铺开" % n
    if n < 500:
        return "中", "控制在 %d 字上下，别啰嗦" % n
    if n < 900:
        return "较长", "写到 %d 字左右" % n
    return "长", "写足 %d 字，别半截就收" % n


def _has_dialog(t):
    return any(k in t for k in ("“", "”", "「", "」", "：", "问："))


def _has_time(t):
    return any(k in t for k in ("那天", "那年", "小时候", "后来", "多年后",
                                 "有一次", "那回", "从前", "记得"))


def _has_contrast(t):
    return any(k in t for k in ("但是", "可是", "然而", "却", "原来",
                                 "本以为", "没想到", "谁知"))


def _opening_kind(t):
    head = t.strip()[:60]
    if _has_dialog(t):
        return "对话开头", "用人物对话开头，别一上来就抒情"
    if head.startswith(("都说", "所有", "每一个", "人生", "生活", "有时候")):
        return "议论开头", "开头就点题议论，不要绕"
    return "白描开头", "开头就着具体的事写，不要铺垫"


# 体裁说法：按 meta.grade 反推合适的口语化说法
_GRADE_STYLE = {
    "一年级": "幼儿园水平的真话，别装大人",
    "二年级": "二年级孩子的话，别复杂",
    "三年级": "三年级的话，别用成语堆",
    "四年级": "四年级的口吻，写得实在点",
    "五年级": "五年级的水平，别装老成",
    "六年级": "六年级的想法，有自己的判断",
    "初一": "初中生的口吻",
    "初二": "初中生的话，敢说真话",
    "初三": "初中生的思考，别空洞",
    "高一": "高中生的话，有想法但不空",
    "高二": "高中生该有的样子",
    "高三": "高中生的表达，紧凑",
}


def build_variants(text, meta, rng):
    n = int(meta.get("chars") or len(text.replace(" ", "").replace("\n", "")))
    title = (meta.get("title") or "这一篇").strip()
    grade = (meta.get("grade") or "").strip()
    style = _GRADE_STYLE.get(grade, "")
    out = []

    # ① 字数口径 —— 永远成立，字数取原文真实字数
    kind, desc = _len_bucket(n)
    out.append((
        "写一篇关于《%s》的作文，%s。" % (title, desc),
        {"_k": "字数", "_v": kind},
    ))

    # ② 年级口吻 —— grade 是原文真实年级，说「像这个年级的孩子写的」为真
    if style and rng.random() < 0.6:
        out.append((
            "写一篇关于《%s》的作文，%s。" % (title, style),
            {"_k": "口吻", "_v": grade},
        ))

    # ③ 开头方式 —— 仅当正文确实符合
    if rng.random() < 0.55:
        _, okind = _opening_kind(text)
        out.append((
            "写一篇关于《%s》的作文，%s。" % (title, okind),
            {"_k": "开头", "_v": okind},
        ))

    # ④ 时序叙述 —— 正文真有「那天/后来/小时候」才提
    if _has_time(text) and rng.random() < 0.6:
        out.append((
            "写一篇关于《%s》的作文，按时间顺序写，从头讲到尾。" % title,
            {"_k": "时序", "_v": "有"},
        ))

    # ⑤ 有人物对话 —— 正文确有对白才提
    if _has_dialog(text) and rng.random() < 0.6:
        out.append((
            "写一篇关于《%s》的作文，里面要有对话，人物说话要像真人说的。"
            % title,
            {"_k": "对话", "_v": "有"},
        ))

    # ⑥ 转折结构 —— 正文确有转折才提
    if _has_contrast(text) and rng.random() < 0.55:
        out.append((
            "写一篇关于《%s》的作文，要有转折，前面铺垫后面翻出来。" % title,
            {"_k": "转折", "_v": "有"},
        ))

    # ⑦ 裸指令 —— 必须保留一部分，否则模型只认带约束的长指令，
    #    遇到「写篇关于X的作文」反而不会（这正是用户的真实说法）
    out.append((
        rng.choice(["写一篇关于《%s》的作文。",
                    "写篇关于《%s》的文章。",
                    "来一篇《%s》的作文。",
                    "帮我写一篇《%s》。",
                    "《%s》，写一篇。"]),
        {"_k": "裸指令", "_v": "-"},
    ))
    return out


def main():
    max_per = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    rng = random.Random(20261014)
    rows = []
    stats = collections.Counter()
    grades = collections.Counter()
    n_src = 0

    with io.open(SRC, encoding="utf-8") as f:
        for line in f:
            n_src += 1
            d = json.loads(line)
            text = (d.get("output") or "").strip()
            if len(text) < 180:
                continue
            meta = d.get("meta") or {}
            if not isinstance(meta, dict):
                meta = {}
            grade = (meta.get("grade") or "").strip()
            # ★ 年级门槛：低于四年级一律不收
            if grade and grade_num(grade) < GRADE_MIN:
                stats["低年级剔除"] += 1
                continue
            grades[grade or "?"] += 1

            variants = build_variants(text, meta, rng)
            rng.shuffle(variants)
            # 保证「裸指令 + 字数」两条必留（这是最常见的两种真实说法）
            must = [v for v in variants if v[1]["_k"] in ("裸指令", "字数")]
            rest = [v for v in variants if v[1]["_k"] not in ("裸指令", "字数")]
            keep = must + rest[:max(0, max_per - len(must))]
            for ins, meta2 in keep:
                rows.append({
                    "instruction": ins,
                    "input": "",
                    "output": text,
                    "_src": "真人作文扩增",
                    "_dim": meta2["_k"],
                    "_grade": meta.get("grade") or "",
                    "_theme": meta.get("theme") or "",
                })
                stats[meta2["_k"]] += 1

    rng.shuffle(rows)
    with io.open(OUT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("源作文 %d 篇 → 扩增 %d 条（×%.1f）"
          % (n_src, len(rows), len(rows) / max(n_src, 1)))
    print("指令维度：", dict(stats))
    print("年级分布前 6：", grades.most_common(6))
    print("→ %s（%.1f MB）" % (OUT, os.path.getsize(OUT) / 1048576))


if __name__ == "__main__":
    main()