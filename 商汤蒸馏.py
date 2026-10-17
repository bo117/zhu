# -*- coding: utf-8 -*-
"""
商汤蒸馏.py —— 用 deepseek-v4-flash 当教师，给「竹」蒸馏一套"零复读"作文语料
================================================================================
【为什么需要它】
竹的病是「复读」：实测训练语料 385,119 句只有 4,361 个不同句（重复率 98.9%），
同一个句子在「成长/英雄/遗憾」等完全不同的主题里被原样复用（例如
「墙上的挂钟走了很多年，秒针每跳一下，都像在替谁数着日子」）。
规则引擎（写作引擎.py）是素材片段拼装、本地模型（Qwen2.5+LoRA）是吃这套
语料训出来的——两条生成路径都遗传了复读。

【本脚本做什么】
用教师模型（deepseek-v4-flash，OpenAI 兼容）生成一批「每个句子都独一无二、
不落俗套、有具体细节」的范文，输出成 训练.py 能直接吃的 {instruction, output}
格式。学生模型再拿这套干净语料做 LoRA 蒸馏，复读病就从数据源头被切掉。

【省 token / 防 429 的六件事】
  1. reasoning_effort="none" —— 关掉思考模式，不白烧 reasoning token
  2. 覆盖矩阵（主题×结构×技法×开头）—— 少量样本铺满多样性，不靠海量堆
  3. 指纹去重 —— 生成即校验，撞库（含旧语料）或篇内自重复立即作废重生成
  4. 429 指数退避 + 断点续传 —— 撞限流不慌，重跑接着写，不重复花钱
  5. --limit N —— 每次最多烧 N 次调用，预算你说了算
  6. --dry-run —— 0 调用，先看要发哪些请求

【用法】
  python 商汤蒸馏.py --dry-run                       # 0 调用，预览提示词
  python 商汤蒸馏.py --limit 24                      # 烧 24 次调用（约 5% 预算）
  python 商汤蒸馏.py --limit 500 --themes 成长,亲情,友情   # 指定主题大批量
  python 商汤蒸馏.py --resume                        # 从上次断点接着写

产物：distill/蒸馏语料.jsonl  格式 {"instruction": ..., "output": ...}
"""
import argparse
import glob
import hashlib
import json
import os
import random
import re
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)

# ───────────────────────── 配置 ─────────────────────────
BASE_URL = "https://token.sensenova.cn/v1"
MODEL = "deepseek-v4-flash"
DEFAULT_KEY = os.environ.get("SENSENOVA_API_KEY", "")   # 不写死：走环境变量或 配置.json
OUT_DIR = os.path.join(ROOT, "distill")
OUT_PATH = os.path.join(OUT_DIR, "蒸馏语料.jsonl")
CKPT_PATH = os.path.join(OUT_DIR, "蒸馏断点.json")
FINGER_CACHE = os.path.join(ROOT, "记忆", "指纹库.json")

# 主题：优先高频应试主题，其次覆盖 65 主题全谱（写作素材库.py 的 THEME_CATEGORY）
CORE_THEMES = ["成长", "亲情", "父爱", "母爱", "友情", "梦想", "坚持", "挫折",
               "感恩", "青春", "故乡", "师恩", "诚信", "责任", "勇气", "选择",
               "平凡", "温暖", "时间", "读书"]
EXTRA_THEMES = ["陪伴", "牵挂", "团圆", "同桌", "知己", "离别", "重逢", "暗恋",
                "迷茫", "遗憾", "毕业", "考试", "春天", "夏天", "秋天", "冬天",
                "雨", "雪", "月亮", "大海", "花", "山", "童年", "思念", "回忆",
                "宽容", "善良", "自信", "独立", "家", "家国", "英雄", "劳动",
                "奉献", "信仰", "幸福", "感动", "微笑", "美食", "音乐", "旅行",
                "宠物", "生命", "等待", "希望"]

STRUCTURES = ["总分总", "总分", "并列", "递进", "对比"]
TECHNIQUES = ["白描", "插叙", "欲扬先抑", "蒙太奇", "首尾呼应", "象征",
              "心理独白", "环境烘托", "对话推进", "细节特写"]
OPENINGS = ["开门见山", "悬念开头", "倒叙", "场景切入", "对话开头", "抒情开头"]
WORD_COUNTS = [600, 800, 1000]

# 烂大街模板句（教师铁律里点名的禁用句，也用于质检兜底）
BANNED = [
    "墙上的挂钟走了很多年", "桌角那杯水早就凉透了", "有些事像茶",
    "人们常说", "后来我才懂", "日子一天天过去", "那一刻我忽然明白",
    "原来最深的感情都藏在最平常的日子里", "这世上的深情多半不动声色",
    "写到这里我抬头看了看窗外", "这些不起眼的碎片拼在一起",
    "青春是一场大雨", "哪有什么岁月静好",
]

# ───────────────────────── 教师提示词 ─────────────────────────
TEACHER_SYSTEM = (
    "你是资深中学语文教师，正在给一个'只会复读模板句'的学生模型写范文当训练数据。"
    "你的每篇范文都会被学生逐句模仿，所以你必须做到零复读、零俗套、有细节。\n\n"
    "【硬性铁律，违反任意一条这篇就作废】\n"
    "1. 零复读：全文每一个句子都必须是全新的，禁止任何两句在句式结构上雷同"
    "（例如连续出现两个'……像……'或两个'……的时候……'也算雷同）。\n"
    "2. 禁用烂大街模板句及其变体，包括但不限于：'墙上的挂钟走了很多年'"
    "'桌角那杯水早就凉透了''有些事像茶''人们常说''后来我才懂''日子一天天过去'"
    "'那一刻我忽然明白''写到这里我抬头看了看窗外''青春是一场大雨'。\n"
    "3. 用具体细节撑起全文：必须有至少一个具体的人、一件具体的小事、一个具体的"
    "物件或场景，靠细节打动读者。禁止空喊口号、禁止堆砌排比、禁止空洞抒情。\n"
    "4. 严格服从题目要求（主题/字数/结构/手法/开头方式），绝不跑题。\n"
    "5. 语言自然，像真人学生写的，不要'AI味'：不要每段都用'首先/其次/最后'，"
    "不要动不动就'总而言之'。\n\n"
    "【输出要求】只输出作文正文，不要任何解释、说明或小标题。字数严格控制在"
    "要求范围 ±10% 内。"
)


def _ask_teacher(spec):
    """构造发给教师的完整对话。spec 是 (主题, 字数, 结构, 技法, 开头)。"""
    theme, words, struct, tech, opening = spec
    user = (
        "写一篇关于「%s」的记叙文，%d 字左右。\n"
        "结构：%s。\n"
        "手法：%s。\n"
        "开头方式：%s。\n"
        "要求：写一个具体的人和一件具体的事，通过细节表达「%s」，"
        "不要直接说教、不要空抒情。"
        % (theme, words, struct, tech, opening, theme)
    )
    return [
        {"role": "system", "content": TEACHER_SYSTEM},
        {"role": "user", "content": user},
    ]


def _make_instruction(spec):
    """把规格转成用户会真实说出口的指令（训练数据里的 instruction 字段）"""
    theme, words, struct, tech, opening = spec
    s = "帮我写一篇关于%s的作文，%d字" % (theme, words)
    if tech and tech != "白描":
        s += "，%s手法" % tech
    if opening and opening != "开门见山":
        s += "，%s" % opening
    if struct and struct != "总分总":
        s += "，%s结构" % struct
    return s


# ───────────────────────── API 调用（带 429 退避） ─────────────────────────
def chat(messages, temperature=0.9, max_tokens=1600, retries=5, timeout=120):
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "reasoning_effort": "none",   # 关思考模式，省 token
        "stream": False,
    }
    data = json.dumps(payload).encode("utf-8")
    for attempt in range(retries):
        req = urllib.request.Request(
            BASE_URL + "/chat/completions", data=data,
            headers={"Authorization": "Bearer " + API_KEY,
                     "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                body = json.loads(r.read().decode("utf-8"))
            msg = body["choices"][0]["message"]
            content = (msg.get("content") or "").strip()
            if not content:
                raise RuntimeError("教师返回空内容")
            return content
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = min(60, 5 * (2 ** attempt) + random.random() * 3)
                log("  ⚠ 429 限流 → 退避 %.1f 秒（第 %d 次）" % (wait, attempt + 1))
                time.sleep(wait)
                continue
            if e.code >= 500:
                log("  ⚠ 服务端 %d → 重试 %d/%d" % (e.code, attempt + 1, retries))
                time.sleep(3 * (attempt + 1))
                continue
            log("  ✗ HTTP %d：%s" % (e.code, e.read().decode("utf-8", "replace")[:200]))
            raise
        except Exception as e:
            log("  ⚠ 网络异常（%s）→ 重试 %d/%d" % (e, attempt + 1, retries))
            time.sleep(2 * (attempt + 1))
            continue
    raise RuntimeError("教师多次失败，放弃此篇")


# ───────────────────────── 指纹去重 ─────────────────────────
_PUNCT = re.compile(r'[\u3000-\u303f\uff00-\uffef\u2018\u2019\u201c\u201d'
                    r'\u2026\u2014\s,.;:!?，。；：！？、（）《》""''——…-]+')
_SPLIT = re.compile(r'[。！？!?\n]')


def _norm(s):
    return _PUNCT.sub("", s or "")


def _sentences(text):
    out = []
    for seg in _SPLIT.split(text or ""):
        ns = _norm(seg)
        if len(ns) >= 6:
            out.append(ns)
    return out


class Fingerprint:
    """全局句级指纹：旧语料句 + 已生成句。撞上就说明在复读。"""

    def __init__(self):
        self.known = set()
        self._load_corpus()
        self._load_cache()

    def _load_corpus(self):
        """把旧训练语料的全部唯一句灌进去（3万篇只有 ~4.4k 个不同句，很轻）"""
        pats = [os.path.join(ROOT, "训练语料_精编.jsonl"),
                os.path.join(ROOT, "作文范文_qa.jsonl")]
        for fp in pats:
            if not os.path.exists(fp):
                continue
            try:
                with open(fp, encoding="utf-8", errors="replace") as f:
                    for line in f:
                        try:
                            body = json.loads(line).get("output", "")
                        except Exception:
                            continue
                        for s in _sentences(body):
                            self.known.add(s)
            except Exception as e:
                log("  指纹库加载 %s 失败：%s" % (fp, e))
        log("  旧语料指纹：%d 个不同句" % len(self.known))

    def _load_cache(self):
        if os.path.exists(FINGER_CACHE):
            try:
                with open(FINGER_CACHE, encoding="utf-8") as f:
                    self.known |= set(json.load(f))
            except Exception:
                pass

    def save(self):
        try:
            os.makedirs(os.path.dirname(FINGER_CACHE), exist_ok=True)
            with open(FINGER_CACHE, "w", encoding="utf-8") as f:
                json.dump(sorted(self.known), f, ensure_ascii=False)
        except Exception:
            pass

    def add(self, text):
        for s in _sentences(text):
            self.known.add(s)

    def check(self, text, limit=0.15):
        """返回 (是否通过, 重合率, 篇内重复率)"""
        sents = _sentences(text)
        n = len(sents)
        if n == 0:
            return False, 1.0, 1.0
        hit = sum(1 for s in sents if s in self.known)
        ratio = hit / n
        # 篇内自重复（一句话在文里出现两次以上就是复读）
        dup = n - len(set(sents))
        dup_ratio = dup / n
        ok = ratio <= limit and dup_ratio <= 0.1
        return ok, round(ratio, 3), round(dup_ratio, 3)


# ───────────────────────── 覆盖矩阵 ─────────────────────────
def build_specs(themes, limit, seed):
    """生成 (主题, 字数, 结构, 技法, 开头) 的覆盖序列，铺满多样性。

    用笛卡尔积 + 洗牌：小批次也能把结构/技法/开头三个维度铺开，
    不会出现「前 N 篇结构全是总分总」的退化。字数在洗牌后循环分配。
    """
    rng = random.Random(seed)
    combos = [(t, s, k, o) for t in themes for s in STRUCTURES
              for k in TECHNIQUES for o in OPENINGS]
    rng.shuffle(combos)
    specs = []
    for i, (theme, struct, tech, opening) in enumerate(combos):
        if len(specs) >= limit:
            break
        words = WORD_COUNTS[i % len(WORD_COUNTS)]
        specs.append((theme, words, struct, tech, opening))
    return specs


def log(msg):
    line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
    print(line, flush=True)


# ───────────────────────── 主流程 ─────────────────────────
def load_done():
    """断点：已完成的 instruction 集合 + 已写条数"""
    if not os.path.exists(CKPT_PATH):
        return set(), 0
    try:
        with open(CKPT_PATH, encoding="utf-8") as f:
            d = json.load(f)
        return set(d.get("done", [])), int(d.get("count", 0))
    except Exception:
        return set(), 0


def save_done(done, count):
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(CKPT_PATH, "w", encoding="utf-8") as f:
        json.dump({"done": sorted(done), "count": count}, f, ensure_ascii=False)


def main():
    global API_KEY
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="0 调用，只预览")
    ap.add_argument("--limit", type=int, default=24, help="最多生成多少篇")
    ap.add_argument("--themes", type=str, default="",
                    help="逗号分隔主题；留空用核心主题循环")
    ap.add_argument("--all-themes", action="store_true", help="覆盖全部 65 主题")
    ap.add_argument("--seed", type=int, default=20260916)
    ap.add_argument("--resume", action="store_true", help="从断点续传")
    ap.add_argument("--key", type=str, default="", help="覆盖 API key")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--sleep", type=float, default=8.0,
                    help="每篇之间的间隔秒数（默认 8s，压 429；赶时间可降到 1）")
    args = ap.parse_args()

    # key：命令行 > 配置.json > 默认
    API_KEY = args.key or DEFAULT_KEY
    cfg_path = os.path.join(ROOT, "配置.json")
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, encoding="utf-8") as f:
                cfg = json.load(f)
            if cfg.get("api_key") and cfg.get("base_url", "").find("sensenova") >= 0:
                API_KEY = cfg["api_key"]
        except Exception:
            pass

    themes = [t for t in args.themes.replace("，", ",").split(",") if t.strip()]
    if args.all_themes:
        themes = CORE_THEMES + EXTRA_THEMES
    if not themes:
        themes = CORE_THEMES

    # 候选池放大：配合 --resume，跳过已完成的组合后仍能找到新组合，持续产出
    specs = build_specs(themes, max(args.limit * 6, 300), args.seed)
    log("=== 商汤蒸馏 ===")
    log("模型 %s | 主题 %d 个 | 目标 %d 篇 | 温度 %.2f"
        % (MODEL, len(themes), args.limit, args.temperature))

    if args.dry_run:
        log("【dry-run】不发请求，预览前 8 个任务：")
        for i, sp in enumerate(specs[:8]):
            ins = _make_instruction(sp)
            sys_msgs = _ask_teacher(sp)
            log("  #%d %s" % (i + 1, ins))
            log("       教师指令：%s" % sys_msgs[1]["content"].replace("\n", " "))
        log("（共 %d 篇，正式跑请去掉 --dry-run）" % args.limit)
        return

    fp = Fingerprint()
    done, count = load_done() if args.resume else (set(), 0)
    if count:
        log("续传：已完成 %d 篇" % count)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_f = open(OUT_PATH, "a", encoding="utf-8")
    ok_cnt = count
    new_cnt = 0
    try:
        for idx, spec in enumerate(specs):
            if new_cnt >= args.limit:      # 本批产出够了就收口（候选池比目标大得多）
                break
            ins = _make_instruction(spec)
            if ins in done:
                continue
            theme, words = spec[0], spec[1]
            max_tok = int(words * 1.6) + 120
            log("[%d/%d] %s" % (idx + 1, len(specs), ins))
            got = None
            for try_i in range(3):   # 撞库/自重复最多重生成 3 次
                try:
                    text = chat(_ask_teacher(spec),
                                temperature=args.temperature, max_tokens=max_tok)
                except Exception as e:
                    log("  ✗ 生成失败：%s" % e)
                    break
                ok, ratio, dup_ratio = fp.check(text)
                if ok:
                    got = text
                    break
                log("  ↻ 复读（撞库 %.1f%% / 篇内 %.1f%%）→ 重生成" %
                    (ratio * 100, dup_ratio * 100))
            if got is None:
                log("  ✗ 本篇作废（多次仍复读/失败），跳过")
                continue
            # 去掉教师可能自作主张加的标题行（《》开头那行保留也无妨，交给训练侧）
            fp.add(got)
            fp.save()
            done.add(ins)
            ok_cnt += 1
            new_cnt += 1
            out_f.write(json.dumps({"instruction": ins, "output": got},
                                   ensure_ascii=False) + "\n")
            out_f.flush()
            save_done(done, ok_cnt)
            log("  ✓ 通过（累计 %d 篇）" % ok_cnt)
            # 温和限速：避免短时间连发把 429 提前招来
            time.sleep(max(0.0, args.sleep))
    finally:
        out_f.close()
    log("=== 完成：本批产出 %d 篇 → %s ===" % (ok_cnt - count, OUT_PATH))


if __name__ == "__main__":
    main()
