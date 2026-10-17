# -*- coding: utf-8 -*-
"""高考真题作文入库（2026-10-14）

  数据来源：MingRLY/College_Entrance_Exam_Chinese_Essays_and_Grades
            90 条 = 题目 + 满分作文 + 阅卷老师评语+ 分数

  为什么这 90 条比 6518 篇考场所文更值钱：
  ────────────────────────────────────────────────────────────────────
  ① **带题目**。6518篇考场作文只有成文，配不上「写作文」这个动作；
     这里每条都是「题目 → 成文」的完整配对 —— **这正是训练要的格式**。
  ② **带阅卷评语 + 分数**。评语是「人工审核过千万遍」的编辑意见
     （用户原话：「这种是人工已经审核过千百万遍的，肯定顶尖」）。
     可以直接变成「诊断维度」语料。
  ③ **题目本身就是指令**。「谈谈你对上述材料的思考和感悟」=
     用户真实会说的话。「材料作文」这种形式和用户的口语高度相似。

  同时产出三类样本（一份数据用三次）：
    A. 成文样本：题目 → 满分作文
    B. 评语样本：成文 → 阅卷诊断（治「听不懂批评」）
    C. 分数样本：成文 → 分数

  ★ 另一个数据集 com11000/c4_wenan（20575 条）**已排除**：
    实测首条是「婺城境内有沙畈、金兰、安地…」（政府公文/地理介绍），
    不是作文。**量大不等于有用**，混进去会把模型教成公文机器。
"""
import sys, io, os, json, re, collections

ROOT = r"D:\竹"
SRC = os.path.join(ROOT, "语料_raw", "gaokao", "train.jsonl")
OUT_A = os.path.join(ROOT, "训练语料_高考真题_成文.jsonl")
OUT_B = os.path.join(ROOT, "训练语料_高考真题_诊断.jsonl")


def split_item(s):
    """把「题目：…\\n作文：标题\\n\\n正文」拆成 (题目, 标题, 正文)。"""
    m = re.search(r"题目：(.*?)\n作文：(.*?)\n", s, re.S)
    if not m:
        # 退化：只有作文，没有题目
        m2 = re.search(r"作文：(.*?)\n", s, re.S)
        if not m2:
            return None
        title = m2.group(1).strip()
        body = s[m2.end():]
        return ("", title, body.strip())
    q = m.group(1).strip()
    rest = s[m.end():]
    head, _, body = rest.partition("\n")
    return (q, head.strip(), body.strip())


def clean_title(t):
    t = re.sub(r"^[　\s]+|[　\s]+$", "", t or "")
    return t


def main():
    rows = []
    with io.open(SRC, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    a_rows, b_rows = [], []
    stats = collections.Counter()
    for r in rows:
        item = r.get("input") or ""
        grade = r.get("output") or ""
        sp = split_item(item)
        if not sp:
            stats["解析失败"] += 1
            continue
        q, title, body = sp
        title = clean_title(title)
        if len(body) < 200:
            stats["正文过短"] += 1
            continue

        # A. 成文样本：题目 → 满分作文
        ins = q if q else "写一篇关于《%s》的作文。" % title
        a_rows.append({
            "instruction": ins,
            "input": "",
            "output": body,
            "_src": "高考真题",
            "_title": title,
        })
        stats["成文"] += 1

        # B/C. 诊断样本：成文 → 阅卷诊断（分数 + 评语）
        m = re.match(r"分数：(\d+)", grade)
        score = m.group(1) if m else None
        cmt = re.sub(r"^评语[：:]", "", grade).strip()
        if score:
            cmt = re.sub(r"^分数：\d+\s*", "", grade).strip()
        if cmt and len(cmt) > 40:
            b_rows.append({
                "instruction": "批改这篇作文，指出好在哪里、问题在哪，给出分数和理由。\n\n作文：%s" % body[:1200],
                "input": "",
                "output": ("分数：%s\n评语：%s" % (score, cmt)) if score else ("评语：%s" % cmt),
                "_src": "高考真题诊断",
                "_score": score,
            })
            stats["诊断"] += 1

    for p, rs in ((OUT_A, a_rows), (OUT_B, b_rows)):
        with io.open(p, "w", encoding="utf-8") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("原始 %d 条 → 成文 %d 条 / 诊断 %d 条" % (len(rows), len(a_rows), len(b_rows)))
    print("统计：", dict(stats))
    lens = sorted(len(r["output"]) for r in a_rows)
    if lens:
        print("作文长度：最短 %d / 中位 %d / 最长 %d"
              % (lens[0], lens[len(lens) // 2], lens[-1]))
    print("→ %s" % OUT_A)
    print("→ %s" % OUT_B)


if __name__ == "__main__":
    main()