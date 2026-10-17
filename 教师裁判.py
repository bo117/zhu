# -*- coding: utf-8 -*-
"""
教师裁判.py —— 用 deepseek-v4-flash 当「真裁判」，给作文打质量分
========================================================================
【为什么要有它】
墨田 v1.4 的诚实结论：**生态的价值上限 = 裁判的上限**。竹现有的质检只有
overlap_check 一个二值闸门（旧版还是恒真的），择优选择.py 的规则裁判
四个维度里只有「新颖」一维真的携带信息，其余三个维度饱和、权重还是拍死的。

本模块把教师模型请来当裁判：给候选作文打五维分（扣题/原创/连贯/文采/细节），
输出可解析的 JSON。它既是「蒸馏质检」的筛子，也能在择优选择时替换掉规则裁判。

【省 token】
  · reasoning_effort="none" 关思考
  · judge_many 一次调用给多篇打分（批处理，1 次调用顶 K 次）
  · judge_one 单篇打分（更稳，解析失败率更低）

【用法】
  import 教师裁判 as judge
  d = judge.judge_one(essay, theme="成长")          # → {"扣题":8, ..., "总分":8.2}
  arr = judge.judge_many([(e1, "成长"), (e2, "梦想")])  # → [d1, d2]

依赖：无（标准库 urllib）。API key 从 配置.json 读，也可 judge.set_key() 覆盖。
"""
import json
import os
import re
import time
import urllib.request

BASE_URL = "https://token.sensenova.cn/v1"
MODEL = "deepseek-v4-flash"
DEFAULT_KEY = os.environ.get("SENSENOVA_API_KEY", "")   # 不写死：走环境变量或 配置.json

API_KEY = DEFAULT_KEY

# 五个打分维度（复读病专门设了「原创」一维，越高越不重复）
DIMS = ["扣题", "原创", "连贯", "文采", "细节"]

_JUDGE_SYSTEM = (
    "你是一位严格的语文阅卷老师，负责给中学生作文打分。"
    "每个维度打 0~10 的整数分，10 最好、0 最差。评分标准：\n"
    "- 扣题：是否紧扣给定主题，不跑题、不偏题\n"
    "- 原创：句子是否新鲜独特，还是套用烂大街模板句、自我重复（越原创越高）\n"
    "- 连贯：段落之间、句子之间的衔接是否自然流畅\n"
    "- 文采：语言是否有表现力、有画面感，而不是干巴巴\n"
    "- 细节：是否有具体的人、事、物撑起文章，而不是空喊口号、堆砌抒情\n\n"
    "只输出 JSON，不要任何解释或 Markdown 代码块。"
)


def set_key(key):
    global API_KEY
    API_KEY = key


def _load_key_from_config():
    global API_KEY
    here = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.join(here, "配置.json")
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, encoding="utf-8") as f:
                cfg = json.load(f)
            if cfg.get("api_key") and cfg.get("base_url", "").find("sensenova") >= 0:
                API_KEY = cfg["api_key"]
        except Exception:
            pass


def _chat(messages, temperature=0.2, max_tokens=800, retries=4, timeout=120):
    payload = {"model": MODEL, "messages": messages,
               "temperature": temperature, "max_tokens": max_tokens,
               "reasoning_effort": "none", "stream": False}
    data = json.dumps(payload).encode("utf-8")
    for attempt in range(retries):
        req = urllib.request.Request(
            BASE_URL + "/chat/completions", data=data,
            headers={"Authorization": "Bearer " + API_KEY,
                     "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                body = json.loads(r.read().decode("utf-8"))
            return (body["choices"][0]["message"].get("content") or "").strip()
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(min(60, 5 * (2 ** attempt) + 2))
                continue
            if e.code >= 500:
                time.sleep(3 * (attempt + 1))
                continue
            raise
        except Exception:
            time.sleep(2 * (attempt + 1))
            continue
    raise RuntimeError("教师裁判多次失败")


def _extract_json(text):
    """从模型输出里稳健地抠出 JSON 对象/数组（容忍前后夹带的话）"""
    text = text.strip()
    # 剥掉 markdown 代码块围栏
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if m:
        text = m.group(1).strip()
    # 找最外层 { 或 [
    for i, ch in enumerate(text):
        if ch in "{[":
            text = text[i:]
            break
    try:
        return json.loads(text)
    except Exception:
        # 截到最后一个 } / ]
        for closer in ("}", "]"):
            j = text.rfind(closer)
            if j > 0:
                try:
                    return json.loads(text[:j + 1])
                except Exception:
                    pass
    raise ValueError("无法解析裁判输出：%s" % text[:120])


def _to_scores(obj, total):
    d = {}
    for k in DIMS:
        try:
            d[k] = int(obj.get(k, 0))
        except Exception:
            d[k] = 0
    try:
        d["总分"] = float(obj.get("总分", total))
    except Exception:
        d["总分"] = round(sum(d[k] for k in DIMS) / len(DIMS), 2)
    return d


def judge_one(essay, theme, temperature=0.2):
    """单篇打分：1 次调用，返回 {扣题,原创,连贯,文采,细节,总分}"""
    _load_key_from_config()
    user = "主题：%s\n\n作文：\n%s\n\n请按评分标准打分。" % (theme, essay)
    out = _chat([{"role": "system", "content": _JUDGE_SYSTEM},
                 {"role": "user", "content": user}], temperature=temperature)
    obj = _extract_json(out)
    if isinstance(obj, list):
        obj = obj[0] if obj else {}
    total = sum(int(obj.get(k, 0)) for k in DIMS) / len(DIMS) if obj else 0
    return _to_scores(obj, total)


def judge_many(items, temperature=0.2):
    """批量打分：items = [(essay, theme), ...]，1 次调用返回分数列表。
    失败时退化为逐篇 judge_one（更稳但更费 token）。"""
    _load_key_from_config()
    if not items:
        return []
    lines = []
    for i, (essay, theme) in enumerate(items):
        lines.append("[%d] 主题：%s\n作文：%s" % (i + 1, theme, essay))
    user = ("下面有 %d 篇作文，请给每篇都打一套分。\n\n"
            % len(items)) + "\n\n".join(lines) + (
        "\n\n输出一个 JSON 数组，第 i 个元素是第 i 篇的分数，"
        "每个元素形如 {\"扣题\":..,\"原创\":..,\"连贯\":..,\"文采\":..,\"细节\":..,\"总分\":..}。")
    out = _chat([{"role": "system", "content": _JUDGE_SYSTEM},
                 {"role": "user", "content": user}], temperature=temperature)
    obj = _extract_json(out)
    if isinstance(obj, list) and len(obj) == len(items):
        res = []
        for o in obj:
            total = sum(int(o.get(k, 0)) for k in DIMS) / len(DIMS) if o else 0
            res.append(_to_scores(o, total))
        return res
    # 批量解析失败 → 逐篇兜底
    return [judge_one(e, t, temperature) for e, t in items]


if __name__ == "__main__":
    import sys
    demo = "外婆的院子里有一棵老枣树。每年秋天，她都要踩着凳子，\
用一根长竹竿把枣打下来，一颗一颗捡进搪瓷盆里，再挑最红的塞给我。\
她的手很粗，裂着口子，可塞枣的动作总是轻轻的。"
    if len(sys.argv) > 1:
        demo = sys.argv[1]
    _load_key_from_config()
    print(json.dumps(judge_one(demo, "亲情"), ensure_ascii=False, indent=2))
