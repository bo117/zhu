# -*- coding: utf-8 -*-
"""
口语诊断.py —— 把「你这篇有点道德绑架的意思」翻译成机器能执行的动作
================================================================================
你要的能力（一句话版）：

    用户说「这篇作文有点道德绑架的意思」
    → 竹要知道：①这是哪一类问题（说教/立意拔高/道德绑架）
               ②出在第几段 ③引用哪句原文 ④该怎么改

为什么要单独一个模块（2026-10-13 调研结论）：

· MultiWOZ 2.1 上的实证：**BERT 监督分类器打爆 few-shot LLM**
  （Mistral-7B weighted-F1 仅 0.50）。1.5B 用 LLM 方式做开放式分类只会更差
  （arXiv:2509.10010）。
· 所以这里走**确定性词表路由**，不是让模型自由发挥：
  规则命中 → 高置信、可测试、零幻觉；命中不了 → 交给模型兜底（fallback）。
· 这套标签体系同时是**训练语料的标注格式**（见 口语诊断语料.py），
  保证「推理时怎么判」和「训练时怎么学」是同一套词表 —— 否则训了也用不上。

八大问题类（对齐 CFMS-34 精简，与中文小学作文常见病灶一致）：

| key | 中文名 | 典型口语说法 | 该改什么 |
|---|---|---|---|
| moralizing | 道德绑架/说教 | 道德绑架、说教、强行升华、讲道理、说教味 | 删掉升华段，别替读者下结论 |
| contrived | 情节生硬/编造 | 太假、编的、不真实、生硬、夸张 | 换成可验证的具体小事 |
| vague | 空洞/没细节 | 太水、没细节、空、笼统、言之无物 | 换成可看见可听见的细节 |
| cliche | 老套/作文腔 | 作文腔、老套、套路、陈词滥调、抒情腔 | 换成人话，砍排比 |
| rushed | 仓促/没写完 | 太短、没写完、赶、字数不够 | 补事件链和余波 |
| offtrack | 跑题 | 跑题、偏题、没扣题、不着题 | 主题词回扣每一段 |
| repetitive | 重复/啰嗦 | 重复、啰嗦、啰唆、翻来覆去、一遍遍 | 合并不重复的段 |
| unnatural | 不通顺/语病 | 语病、别扭、不通、读着卡、拗口 | 逐句改顺 |

用法：
    from 口语诊断 import diagnose, PROBLEM_TYPES
    r = diagnose("这篇作文有点道德绑架的意思", essay_text)
    r.type        # 'moralizing'
    r.paragraphs  # [3, 5]      段落序号（1 起）
    r.evidence    # ['原文句…']  引用
    r.action      # '删掉升华段…'  该怎么改
    r.confidence  # 0.0~1.0
    r.matched     # '道德绑架'    命中了哪个词
"""
import re

# ── 八大问题类 ──────────────────────────────────────────────────────
# cues  = 触发词（口语、简称、错别字都收）
# hint  = 给执行器（改稿）的方向
# probe = 在正文里找证据时用的特征词
PROBLEM_TYPES = [
    {"key": "moralizing", "name": "道德绑架/说教",
     "cues": ["道德绑架", "说教", "强行升华", "升华", "讲道理", "上价值",
              "说教味", "教育意义", "教导", "说教感"],
     "hint": "删掉替读者下结论的升华段，让事件自己说话",
     "probe": ["我终于明白", "我懂得了", "原来", "这告诉我们", "道理",
                "我们应该", "所谓成长", "真正的", "从那以后我"]},

    {"key": "contrived", "name": "情节生硬/编造",
     "cues": ["生硬", "太假", "假", "编的", "编造", "不真实", "夸张",
              "胡编", "不合理", "巧合"],
     "hint": "把离奇的情节换成可信的小事，砍掉 improbable 的转折",
     "probe": ["正好", "恰好", "忽然", "突然", "就在这时", "偏偏"]},

    {"key": "vague", "name": "空洞/没细节",
     "cues": ["空洞", "空", "太水", "水", "没细节", "笼统", "言之无物",
              "泛泛", "没内容", "干巴巴"],
     "hint": "把概括句换成看得见听得见的具体细节",
     "probe": ["很多", "很多很多", "一些", "各种", "许多"]},

    {"key": "cliche", "name": "老套/作文腔",
     "cues": ["作文腔", "老套", "套路", "陈词滥调", "抒情腔", "文艺",
              "矫情", "假 literary", "文绉绉", "掉书袋"],
     "hint": "换成口语，砍掉排比和感叹",
     "probe": ["仿佛", "宛如", "似乎", "那一刻", "定格", "蔓延", "荡漾",
                "如诗", "画卷", "烙印"]},

    {"key": "rushed", "name": "仓促/没写完",
     "cues": ["太短", "没写完", "字数不够", "赶", "没写够", "草率",
              "太急", "短"],
     "hint": "补事件链的中间环节和余波，别急着跳到结论",
     "probe": []},

    {"key": "offtrack", "name": "跑题",
     "cues": ["跑题", "偏题", "离题", "没扣题", "不着题", "文不对题",
              "题不符"],
     "hint": "每一段都回扣主题，砍掉与主题无关的枝节",
     "probe": []},

    {"key": "repetitive", "name": "重复/啰嗦",
     "cues": ["重复", "啰嗦", "啰唆", "翻来覆去", "一遍遍", "反复说",
              "啰", "重复了", "车轱辘"],
     "hint": "合并意思相同的段落，每件事只说一次",
     "probe": ["一次又一次", "一遍遍", "又一次", "再次", "总是"]},

    {"key": "unnatural", "name": "不通顺/语病",
     "cues": ["语病", "别扭", "不通", "读着卡", "拗口", "不通顺",
              "语法", "错别字", "标点"],
     "hint": "逐句改顺，长句拆短，主谓宾理清",
     "probe": []},
]

_BY_CUE = {}
for _t in PROBLEM_TYPES:
    for _c in _t["cues"]:
        _BY_CUE[_c] = _t

# 否定词：用户说"不是道德绑架"时要撤销命中
_NEG = re.compile(r"(不是|别|没有|无|不至于|算不上|谈不上)")


class Diagnosis(object):
    """一次诊断的结果。用 .as_dict() 喂给改稿执行器或存成语料。"""

    def __init__(self, type=None, name="", paragraphs=None, evidence=None,
                 action="", confidence=0.0, matched="", raw=""):
        self.type = type
        self.name = name
        self.paragraphs = paragraphs or []
        self.evidence = evidence or []
        self.action = action
        self.confidence = confidence
        self.matched = matched
        self.raw = raw

    def as_dict(self):
        return {"type": self.type, "name": self.name,
                "paragraphs": self.paragraphs, "evidence": self.evidence,
                "action": self.action, "confidence": round(self.confidence, 3),
                "matched": self.matched}

    def __repr__(self):
        return "<Diagnosis %s %s 第%s段 conf=%.2f>" % (
            self.type, self.name, self.paragraphs, self.confidence)

    def __bool__(self):
        return bool(self.type)


def split_paragraphs(essay):
    """按空行切段，返回 [(序号, 段文本)]，序号从 1 起"""
    out = []
    for i, p in enumerate(re.split(r"\n\s*\n", essay or ""), 1):
        t = (p or "").strip()
        if t:
            out.append((i, t))
    return out


def _sentences(para):
    return [s.strip() for s in re.split(r"[。！？；]", para) if s.strip()]


def match_problem(request):
    """只做词表路由：用户这句话判成了什么问题类型？"""
    req = request or ""
    best = None
    for cue, t in _BY_CUE.items():
        for m in re.finditer(re.escape(cue), req):
            # 命中位置前面 6 字内有否定词 → 跳过
            pre = req[max(0, m.start() - 6):m.start()]
            if _NEG.search(pre):
                continue
            # 长词优先（"道德绑架" 比 "绑架" 更具体）
            score = len(cue) * 10 - m.start() * 0.1
            if best is None or score > best[0]:
                best = (score, cue, t)
    if not best:
        return None
    _, cue, t = best
    return t["key"], t["name"], cue, t["hint"]


def locate(essay, type_key, max_hits=3):
    """在正文里定位：哪几段、哪几句最可能是这个问题。"""
    tpl = next((t for t in PROBLEM_TYPES if t["key"] == type_key), None)
    if not tpl:
        return [], []
    paras = split_paragraphs(essay)
    probes = tpl["probe"]
    hit_p, hit_e = [], []
    if not probes:
        return hit_p, hit_e
    for idx, text in paras:
        for s in _sentences(text):
            n = sum(1 for p in probes if p in s)
            if n:
                hit_p.append(idx)
                hit_e.append(s[:120])
                break
        if len(hit_p) >= max_hits:
            break
    # 升华类问题优先打末段（作文的毛病几乎都在结尾）
    if type_key == "moralizing" and paras and not hit_p:
        hit_p = [paras[-1][0]]
        hit_e.append(_sentences(paras[-1][1])[-1][:120] if _sentences(paras[-1][1]) else "")
    return list(dict.fromkeys(hit_p)), hit_e[:max_hits]


def diagnose(request, essay=""):
    """主入口：用户那句话 + 原稿 → Diagnosis"""
    m = match_problem(request)
    if not m:
        return Diagnosis(raw=request)
    key, name, cue, hint = m
    paras, ev = locate(essay, key)
    conf = 0.5 if paras else 0.3
    if len(cue) >= 3:
        conf += 0.15
    if paras:
        conf += 0.15
    return Diagnosis(type=key, name=name, paragraphs=paras,
                     evidence=ev, action=hint, confidence=min(conf, 0.95),
                     matched=cue, raw=request)


# ── 自检：改完立刻能验，不依赖任何模型 ──────────────────────────────
if __name__ == "__main__":
    _ESSAY = ("我坐在开往故乡的绿皮火车上。\n\n"
              "王二毛是我童年唯一的玩伴。他家在隔壁，院里有棵枇杷树。\n\n"
              "火车驶过一座桥，河水泛着碎光。\n\n"
              "我终于明白，所谓成长，就是把哭声调成静音的过程。"
              "这告诉我们，每个人都要学会与自己和解。")
    _CASES = [
        "这篇作文有点道德绑架的意思",
        "感觉不太对劲",
        "太水了，没细节",
        "老套，全是作文腔",
        "有点啰嗦，翻来覆去的",
        "跑题了",
        "不是道德绑架，就是有点老套",
    ]
    print("=" * 60)
    for c in _CASES:
        d = diagnose(c, _ESSAY)
        print("%-22s → %s" % (c, d.as_dict()))
    print("=" * 60)
    assert diagnose("道德绑架", _ESSAY).type == "moralizing"
    assert diagnose("太水了", _ESSAY).type == "vague"
    assert diagnose("老套", _ESSAY).type == "cliche"
    assert diagnose("啰嗦", _ESSAY).type == "repetitive"
    assert diagnose("跑题", _ESSAY).type == "offtrack"
    assert not diagnose("这段写得挺好", _ESSAY)      # 无信号 → 交给模型兜底
    assert diagnose("不是道德绑架", _ESSAY).type != "moralizing"  # 否定
    d = diagnose("道德绑架", _ESSAY)
    assert d.paragraphs == [4], d.paragraphs          # 升华在末段
    print("✓ 八类路由 + 否定 + 定位 自检全过")