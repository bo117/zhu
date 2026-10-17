# -*- coding: utf-8 -*-
"""
竹 · 写作助手 2.0 —— 会记事、会安慰人、会上网查资料并自己整合的聊天伙伴
==================================================================
没有预置答案题库。它的回复由四件套拼出来（仿脑分工）：
  1) 海马体：长期记忆 —— 你说过的话按关键词存进本地 TXT，
     几十句之后再提起"酸苹果"，它立刻能翻出来接上话
  2) 杏仁核：情绪雷达 —— 听得懂难过/烦/累/孤独/开心，
     用人的方式回应你、安慰你（这是情感反射弧，不是查表答案）
  3) 前额叶：联网整合 —— 不懂的问题自己去搜多篇文章，
     把句子打分、去重、揉成一段通顺中文，逐字流式吐给你
  4) 讲话习惯：口癖 + 联想词，它的"性格"部分

所有思考数据和思考过程，都以 UTF-8 中文 TXT 存到本地 记忆/ 文件夹：
  记忆/chat_log.txt   每一句对话
  记忆/facts.txt      长期记忆（时间 | 关键词 | 情绪 | 原话）
  记忆/thinking.txt   每一次思考过程

用法：
  python 竹.py            直接聊天（联网默认开启，流式输出）
  python 竹.py --demo     跑演示（含"酸苹果"记忆联想演示）
  python 竹.py --offline  断网模式（纯本地：训练库+记忆+情绪，照样能答）
  python train_gen2.py       启动"训练 2 代"：爬公开训练集喂给竹，不爬重复
  聊天时连按 3 次「1」= 清空聊天/记忆/思考三本账（训练库是知识，保留）；
  左上角实时显示 token 用量和每秒吐字速度

可选升级：编辑 brain_config.json 填入 DeepSeek 等 OpenAI 兼容 API Key，
整合质量和共情能力会大幅提升；不填 Key 也能跑（本地抽取式整合）。

理论依据（不是瞎编）：
  - 情绪快速通道：LeDoux 杏仁核快慢通路（先共情，再思考）
  - 记忆索引：海马体的线索回忆（关键词即提取线索）
  - 多文档整合：TF 加权句子打分 + 去重（简化版 MMR/TextRank）
  - 流式输出：LLM 的 SSE 流式协议（DeepSeek/OpenAI 兼容）
"""

import hashlib
import html as _html
import json
import os
import random
import re
import sys
import threading
import time
import urllib.parse
import urllib.request
import webbrowser

# 神经网络大脑（双塔语义模型）：练好就自动启用，没 numpy / 没练好都不碍事
try:
    import 神经网络 as neural_brain
except Exception:
    neural_brain = None

# 写作引擎（写作模式总管：65主题/12题材/5结局/6开头/5结构/6名家/
# 3档推理强度/长篇续写/局部修改——写作模式一律走它）
try:
    import 写作引擎 as write_brain
except Exception:
    write_brain = None

# ----------------------------------------------------------------------
# 配置：配置.json（版本/模型/服务器地址，写作模式的中枢设置）
# ----------------------------------------------------------------------
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "配置.json")
DEFAULT_CONFIG = {
    "_说明": "版本：免费=纯本地小模型；Plus/Pro=填 API Key 走大模型（Pro 推理更强、更耗 token）。"
             "server_url 填云服务器地址后，API Key 需与服务器匹配才能用高阶版。",
    "edition": "免费",
    "api_key": "",
    "base_url": "https://api.deepseek.com",
    "model": "deepseek-chat",
    "server_url": "",
    "stream_speed": 0.02,   # 逐字输出的间隔秒数（像人打字）
    # ── 桌面文档导出（可选，默认关闭）──────────────────────────
    # enabled  : 是否自动在桌面创建 docx。默认 False = 一律直接在对话里输出正文，
    #            不再强制「先落 docx 才能看到长文」。想自动落盘就设 true。
    # threshold: enabled=true 时，达到多少字才落桌面
    # 用户临时想存：直接说「存到桌面 / 导出文档」即可（无需改配置）
    # 设置：编辑 配置.json，或跑 `python 设置桌面导出.py`
    "desktop_export": {
        "enabled": False,
        "threshold": 1000,
    },
}


def desktop_export_cfg():
    """读桌面文档导出设置，返回 (是否开启, 字数阈值)。
    默认 (True, 1000) —— 即「大于等于 1000 字才在桌面创建」。
    带 1 秒缓存，改完 配置.json 无需重启。"""
    global _DEX_CACHE
    now = time.time()
    if _DEX_CACHE and now - _DEX_CACHE[0] < 1.0:
        return _DEX_CACHE[1], _DEX_CACHE[2]
    d = load_config().get("desktop_export")
    if not isinstance(d, dict):
        d = {}
    enabled = bool(d.get("enabled", False))       # 默认关闭：直接输出，不强制落盘
    try:
        threshold = int(d.get("threshold", 1000))
    except Exception:
        threshold = 1000
    threshold = max(0, threshold)
    _DEX_CACHE = (now, enabled, threshold)
    return enabled, threshold


_DEX_CACHE = None

# 用户显式要文档时（无论配置开关），照样落一份：说「存到桌面 / 导出文档」即可
_SAVE_INTENT_RE = re.compile(
    r"存到桌面|存桌面|导出(成|为)?(文档|word|docx|文件)?|保存到(桌面|文档)|"
    r"生成(一份)?(文档|word|docx)|要(一份)?(文档|word|docx)|下载(成)?(文档|word)")


def _save_intent(msg):
    """用户是不是明确要求把内容存成文档"""
    return bool(_SAVE_INTENT_RE.search(msg or ""))


def load_config():
    cfg = dict(DEFAULT_CONFIG)
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg.update(json.load(f))
    except FileNotFoundError:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_CONFIG, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
    return cfg


def save_config(cfg):
    """界面改版本/模型/服务器后写回 配置.json"""
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False


# ----------------------------------------------------------------------
# 账户与 API Key（本地账本 账户.json；每账户限 20 把 Key）
# server_url 配置后，verify 会先去服务器验 Key，服务器没通就退回本地校验
# ----------------------------------------------------------------------
ACCOUNTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "账户.json")
MAX_KEYS_PER_ACCOUNT = 20


def load_accounts():
    try:
        with open(ACCOUNTS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"accounts": {}}


def save_accounts(data):
    try:
        with open(ACCOUNTS_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False


def _pwd_hash(salt, pwd):
    return hashlib.sha256((salt + ":" + pwd).encode("utf-8")).hexdigest()


def account_register(user, pwd):
    user = (user or "").strip()
    if not user or not pwd:
        return False, "用户名和密码都得有。"
    if not re.fullmatch(r"[\w\u4e00-\u9fff\-]{2,24}", user):
        return False, "用户名 2~24 位，只许中英文、数字、下划线。"
    if len(pwd) < 6:
        return False, "密码至少 6 位。"
    data = load_accounts()
    if user in data["accounts"]:
        return False, "这个用户名已经有人用了。"
    salt = os.urandom(8).hex()
    data["accounts"][user] = {
        "salt": salt,
        "pwd": _pwd_hash(salt, pwd),
        "created": time.strftime("%Y-%m-%d %H:%M"),
        "keys": [],
    }
    save_accounts(data)
    return True, "账户「%s」注册成功，进来就能生成 API Key。" % user


def account_login(user, pwd):
    data = load_accounts()
    acc = data["accounts"].get((user or "").strip())
    if not acc:
        return False, "没有这个账户，先注册一个。"
    if acc["pwd"] != _pwd_hash(acc.get("salt", ""), pwd or ""):
        return False, "密码不对，再想想。"
    return True, "欢迎回来，%s。" % user


def _get_account(user):
    data = load_accounts()
    return data, data["accounts"].get((user or "").strip())


def key_list(user):
    _, acc = _get_account(user)
    if not acc:
        return None, "先登录。"
    rows = [{"key": k["key"][:12] + "…" + k["key"][-6:],
             "full": k["key"], "created": k["created"],
             "plan": k.get("plan", "plus"), "status": k.get("status", "active")}
            for k in acc.get("keys", [])]
    return rows, "共 %d/%d 把" % (len(rows), MAX_KEYS_PER_ACCOUNT)


def key_create(user, plan="plus"):
    data, acc = _get_account(user)
    if not acc:
        return None, "先登录再生成 Key。"
    active = [k for k in acc.get("keys", []) if k.get("status", "active") == "active"]
    if len(active) >= MAX_KEYS_PER_ACCOUNT:
        return None, "每个账户最多 %d 把 Key，先吊销几把再来。" % MAX_KEYS_PER_ACCOUNT
    key = "zhu-" + os.urandom(16).hex()
    acc.setdefault("keys", []).append({
        "key": key, "created": time.strftime("%Y-%m-%d %H:%M"),
        "plan": plan if plan in ("plus", "pro") else "plus",
        "status": "active"})
    save_accounts(data)
    return key, "新 Key 生成（第 %d/%d 把）" % (len(acc["keys"]),
                                               MAX_KEYS_PER_ACCOUNT)


def key_revoke(user, key):
    data, acc = _get_account(user)
    if not acc:
        return False, "先登录。"
    for k in acc.get("keys", []):
        if k["key"] == key:
            k["status"] = "revoked"
            save_accounts(data)
            return True, "已吊销。"
    return False, "没找到这把 Key。"


def key_verify(key):
    """验 Key：填了 server_url 先问服务器（后期对接用），没填/没通
    就查本地账本。返回 (ok, 说明)"""
    key = (key or "").strip()
    if not key:
        return False, "Key 是空的。"
    cfg = load_config()
    server = (cfg.get("server_url") or "").strip()
    if server:
        try:
            req = urllib.request.Request(
                server.rstrip("/") + "/verify",
                data=json.dumps({"key": key}).encode("utf-8"),
                headers={"Content-Type": "application/json"}, timeout=5)
            with urllib.request.urlopen(req, timeout=6) as resp:
                obj = json.loads(resp.read().decode("utf-8"))
                return bool(obj.get("ok")), obj.get("msg", "服务器校验完成")
        except Exception as e:
            return False, "服务器没验证过（%s）——Key 必须与服务器匹配。" % e
    data = load_accounts()
    for acc in data["accounts"].values():
        for k in acc.get("keys", []):
            if k["key"] == key and k.get("status", "active") == "active":
                return True, "本地校验通过（%s 版）" % k.get("plan", "plus")
    return False, "这把 Key 不在账上。"



import 阿呆 as aidan
from 阿呆 import is_knowledge_question, load_history

# ── 作文三雷达（写作/改文/点评） ──────────────────────────
# 写作：写一篇关于XX的作文/帮我写600字作文
# ⚠️ 旧版太窄：`[一?篇]?.{0,6}?` 要求「一篇」和「作文」之间最多 6 个字，
# 用户说「写一篇关于坚持的1000字作文」时中间有 9 个字 → 不匹配 → 掉进闲聊兜底，
# 表现就是"答非所问"。这里放宽到 30 字，并补上「以X为题」「1000字作文」等常见说法。
ESSAY_WRITE_RE = re.compile(
    r"(写|帮我写|给我写|替我写|来)[一]?篇?[^。！？\n]{0,30}?(作文|文章|记叙文|议论文|散文)"
    r"|关于[^。！？\n]{1,24}?的?(作文|文章)"
    r"|以[^。！？\n]{1,20}?为(题|话题)"
    r"|[0-9一二三四五六七八九十百千]{1,6}\s*字[^。！？\n]{0,8}?(作文|文章)")
# 改文：把父爱改成母爱 / 重写 / 不满意再来一篇 / 帮我改这篇稿子
#      （改文必须在写作之后；改稿走"局部修改"，不整篇重生）
ESSAY_EDIT_RE = re.compile(
    r"(把|将).{1,10}(改|换)成|改成|重新写|再写一(篇|遍)|重写|不满意.{0,6}"
    r"(再来|重写)|不行.{0,4}(重写|换)|修改(一下)?(这篇|这篇作文)?"
    r"|帮[我]?改(一?下|一改)?(这篇|这篇作文|这篇稿子|我的作文)"
    r"|(这篇|这篇作文|稿子).{0,4}(改一?下|修改|润色)")
# 点评：点评/评价一下这篇作文 + 用户贴正文
ESSAY_REVIEW_RE = re.compile(
    r"(点评|评价|批改|打分|帮我看看|改一改|提提意见).{0,8}"
    r"(作文|文章|这篇|写的|我的)|我的作文|这是我写的")
# 小说雷达：开新书（题材词/小说字样）
NOVEL_RE = re.compile(
    r"写(一?本|一?部|一?篇)?.{0,8}?小说|长篇.{0,4}(写|小说)|开(一?本|部)新书?"
    r"|给我(来|写)(一?章|一章小说)|写.{0,6}(二次元|都市爽文|都市|末日|末世|玄幻|修仙"
    r"|仙侠|科幻|星际|悬疑|推理|武侠|江湖|无限流|穿越|重生|甜宠|言情|恐怖|灵异)"
    r".{0,6}(小说|文|故事)?")
def _is_edit_feedback(msg, state=None):
    if write_brain is None:
        return False
    return write_brain.is_structured_revision(msg, state or {})
# 整本雷达：一次写完整本小说（作者/简介/目录/正文）
NOVEL_FULL_RE = re.compile(r"整本|全本|完整的小?说|一口气写完|从目录到正文")
# 续写雷达：小说工作台开着时，"续写/下一章/接着写"
CHAPTER_CONT_RE = re.compile(
    r"^(续写|下一章|接着写|继续写|再写一章|写下一章|往后写|更新一章?)(一?章)?.{0,10}$"
    r"|续写.{0,8}(一章|下一章|往下|正文)|下一章(呢|啊|呗|吧)?")




UI_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "写作台V3.html")
WEB_PORT = 8765
CHAT_LOCK = threading.Lock()



# ----------------------------------------------------------------------
# 竹（主）：写作模式的本体。聊天/搜索/记忆委托给 阿呆（聊天脑），
# 写作/改稿/续写/点评走写作引擎 + 工作台。
# ----------------------------------------------------------------------
class Companion:
    def __init__(self, offline=False, stream_delay=0.02, quiet=False):
        # 聊天脑：阿呆（记忆、情绪、联网搜索、寒暄、知识问答）
        self.aidan = aidan.Companion(offline=offline,
                                     stream_delay=stream_delay,
                                     quiet=quiet)
        self.cfg = load_config()
        # 写作工作台：记住上一篇作文/小说（多轮修改不迷路），重启后接着改
        self.essay_state = None
        if write_brain is not None:
            self.essay_state = write_brain.load_state()

    # ---- 委托属性：接口层照旧能用 buddy.mem / buddy.total_tokens ----
    @property
    def mem(self):
        return self.aidan.mem

    @property
    def total_tokens(self):
        return self.aidan.total_tokens

    @property
    def llm(self):
        return self.aidan.llm

    @property
    def edition(self):
        return self.aidan.edition

    def add_tokens(self, n):
        self.aidan.add_tokens(n)

    def stream(self, text):
        self.aidan.stream(text)

    def apply_edition(self):
        self.aidan.apply_edition()

    # ---------- 主入口：写作归主，聊天归阿呆 ----------
    def reply(self, msg, retry=False, last_reply=""):
        steps = []
        msg = (msg or "").strip()
        if write_brain is not None:
            if self.essay_state:
                context = dict(self.essay_state)
                if context.get("mode") == "novel" and context.get("chapters"):
                    context["text"] = context["chapters"][-1]["text"]
                plan = write_brain.revision_plan(context, msg)
                if plan["intent"] == "revise":
                    steps.append("主控：上下文修改计划 → 工作台")
                    return self._essay_edit(msg, steps, plan), steps
                if plan["intent"] == "clarify":
                    return plan["question"], steps
                if plan["intent"] == "review":
                    return self._essay_review(msg, steps), steps
                if plan["intent"] == "chat" and plan.get("source") == "semantic" and not (
                        context.get("mode") == "novel" and CHAPTER_CONT_RE.search(msg)):
                    return self.aidan.reply(msg, retry=retry, last_reply=last_reply)
            # ① 小说：开新书 > 续写下一章
            if NOVEL_RE.search(msg) and NOVEL_FULL_RE.search(msg):
                steps.append("主控：整本小说请求 → 一次成书")
                return self._novel_fullbook(msg, steps), steps
            if NOVEL_RE.search(msg):
                steps.append("主控：小说请求 → 写作引擎开书")
                return self._novel_write(msg, steps), steps
            if self.essay_state and self.essay_state.get("mode") == "novel" \
                    and CHAPTER_CONT_RE.search(msg):
                steps.append("主控：长篇续写 → 推进下一章")
                return self._novel_continue(steps), steps
            # ② 作文：写作 > 改稿 > 点评
            if ESSAY_WRITE_RE.search(msg):
                steps.append("主控：写作请求 → 写作引擎动笔")
                return self._essay_write(msg, steps), steps
            if ESSAY_REVIEW_RE.search(msg) and not ESSAY_WRITE_RE.search(msg):
                steps.append("主控：点评请求 → 四维体检")
                return self._essay_review(msg, steps), steps
        # ②.5 工作台有稿时，短句改稿反馈归主（"不要是雨天""背景要晴天"这类）
        # ③ 其余一律交给阿呆（聊天脑：记忆/情绪/搜索/寒暄/知识）
        return self.aidan.reply(msg, retry=retry, last_reply=last_reply)

    # ---------- 写作四件套：作文 / 小说续写 / 局部修改 / 点评 ----------
    def _brief_from_params(self, body):
        """界面结构化参数 → 写作单；缺的字段用默认值补齐"""
        mode = body.get("mode") or "essay"
        words = body.get("words") or 600
        try:
            words = max(200, min(400000, int(words)))
        except Exception:
            words = 600
        return {
            "mode": "novel" if mode in ("novel", "小说") else "essay",
            "theme": body.get("theme") or None,
            "genre": body.get("genre") or None,
            "ending": body.get("ending") or None,
            "opening": body.get("opening") or "开门见山",
            "technique": body.get("technique") or None,
            "structure": body.get("structure") or None,
            "master": body.get("master") or None,
            "depth": body.get("depth") or "标准",
            "style": body.get("style") or "记叙文",
            "words": words,
            "arc": "先虐后甜" if body.get("arc") else None,
            "direction": body.get("direction") or None,
            "characters": body.get("characters") or None,
        }

    def _brief_merge(self, body):
        """界面结构化参数 + 「其他要求」(msg) → 合成写作单。
        面板上明确选了的优先；msg 只补空位 + 解析可执行的额外要求"""
        brief = self._brief_from_params(body)
        msg = (body.get("msg") or "").strip()
        if msg and write_brain is not None:
            parsed = write_brain.parse_brief(msg)
            for k in ("theme", "genre", "ending", "technique",
                      "structure", "master"):
                if not brief.get(k) and parsed.get(k):
                    brief[k] = parsed[k]
            if brief.get("opening") in (None, "开门见山") \
                    and parsed.get("opening") not in (None, "开门见山"):
                brief["opening"] = parsed["opening"]
            if parsed.get("mode") == "novel":
                brief["mode"] = "novel"
            if not body.get("words"):
                brief["words"] = parsed["words"]
            if body.get("direction"):
                brief["direction"] = body["direction"]
            elif parsed.get("direction"):
                brief["direction"] = parsed["direction"]
            brief["extra"] = write_brain.parse_extra(msg)
        return brief

    def _write_from_brief(self, brief, steps):
        """按写作单动笔：小说开新书；作文出全篇。写完更新工作台"""
        if brief["mode"] == "novel":
            st = write_brain.start_novel(brief)
            self.essay_state = st
            write_brain.save_state(st)
            nch = len(st.get("chapters") or [])
            steps.append("写作工作台：开新书%s（%s · %s结局 · %d章）"
                         % (st["title"], st["genre"], st["ending"], nch))
            char_head = ""
            if st.get("characters"):
                char_head = write_brain.characters_block(st["characters"]) + "\n\n"
            ol = st.get("outline") or []
            ol_txt = ""
            if ol:
                ol_txt = "【分章大纲】\n" + "\n".join(
                    "  第%s章 %s" % (c.get("no"), c.get("title")) for c in ol) + "\n\n"
            return ("%s（%s · 结局走向：%s）\n\n%s%s%s\n\n"
                    "（说「续写」就接着往下写下一章，可以一直写下去。）"
                    % (st["title"], st["genre"], st["ending"],
                       char_head, ol_txt, st.get("text") or ""))
        theme = brief["theme"]
        if not theme:
            return ("写哪方面的？给我个主题（比如「父爱」「青春」「暗恋」），"
                    "或者在左侧写作模式里选好参数再点「开始写作」。")
        # ★ API 主笔（本地跑，无需 GPU）：配置了 key 就先用大模型写全篇，
        #   失败/无 key 自动回退下面的原引擎路径。
        #   回退一行：写作引擎.USE_API_WRITER = False
        try:
            _api = write_brain.api_write_essay(brief)
        except Exception:
            _api = None
        if _api:
            title, text = _api
            extras = [x for x in (brief["structure"], brief["technique"],
                                  "仿%s" % brief["master"] if brief["master"] else "",
                                  "%s开头" % brief["opening"]
                                  if brief["opening"] != "开门见山" else "",
                                  "%s结局" % brief["ending"] if brief["ending"] else "",
                                  brief["depth"] + "推理"
                                  if brief["depth"] != "标准" else "")
                      if x]
            st = {"mode": "essay", "theme": theme, "title": title, "text": text,
                  "turn": 0, "style": brief["style"], "words": brief["words"],
                  "ending": brief["ending"], "opening": brief["opening"],
                  "technique": brief["technique"], "structure": brief["structure"],
                  "master": brief["master"], "depth": brief["depth"],
                  "arc": brief["arc"], "direction": brief.get("direction"),
                   "by": "api", "request": brief.get("request"),
                   "extra": brief.get("extra"), "object": brief.get("object")}
            self.essay_state = st
            write_brain.save_state(st)
            steps.append("写作工作台：《%s》%d 字（API 主笔）" % (title, len(text)))
            return "《%s》%s\n\n%s" % (title,
                                      "（%s）" % "/".join(extras) if extras else "",
                                      text)
        # 桌面文档导出（改为「可选」，默认不再强制）：
        #  · 默认：直接把正文输出到对话，不需要先落 docx；
        #  · 配置 desktop_export.enabled=true 时，按字数阈值自动落盘；
        #  · 用户说「存到桌面 / 导出文档」时，无论配置一律落盘。
        _dex_on, _dex_min = desktop_export_cfg()
        _explicit = bool(brief.get("_save_doc"))
        if ((_explicit or _dex_on)
                and (_explicit or brief["words"] >= _dex_min)
                and hasattr(write_brain, "write_longform_doc")):
            path, written, no, preview = write_brain.write_longform_doc(
                brief, max(int(brief["words"] or 0), 1200))
            bk = path.split("《")[-1].split("》")[0]
            self.essay_state = {"mode": "longform", "theme": theme,
                                "title": "《" + bk + "》", "text": preview,
                                "words": written, "turn": 0, "path": path,
                                "style": "长文"}
            write_brain.save_state(self.essay_state)
            steps.append("长文创作：%d 字 / %d 节 → 已存桌面文档" % (written, no))
            return ("长文已完成（正文已在下方，同时存了一份桌面文档）：\n%s\n\n"
                    "共 %d 字 / %d 节，文风统一、内容不重复。\n\n正文：\n%s"
                    % (path, written, no, preview))
        title, text = write_brain.write_essay2(brief)
        st = {"mode": "essay", "theme": theme, "title": title, "text": text,
              "turn": 0, "style": brief["style"], "words": brief["words"],
              "ending": brief["ending"], "opening": brief["opening"],
              "technique": brief["technique"], "structure": brief["structure"],
              "master": brief["master"], "depth": brief["depth"],
              "arc": brief["arc"], "direction": brief.get("direction"),
              "request": brief.get("request"), "extra": brief.get("extra"),
              "object": brief.get("object"), "by": "material_draft"}
        self.essay_state = st
        write_brain.save_state(st)
        extras = [x for x in (brief["structure"], brief["technique"],
                              "仿%s" % brief["master"] if brief["master"] else "",
                              "%s开头" % brief["opening"]
                              if brief["opening"] != "开门见山" else "",
                              "%s结局" % brief["ending"] if brief["ending"] else "",
                              brief["depth"] + "推理" if brief["depth"] != "标准" else "")
                  if x]
        steps.append("写作工作台：《%s》%d 字（%s%s）"
                     % (title, len(text), brief["style"],
                        " · " + "/".join(extras) if extras else ""))
        return "模型未能完成可用的全文，以下为素材草稿：\n\n《%s》%s\n\n%s" % (title,
                                   "（%s）" % "/".join(extras) if extras else "",
                                   text)

    def _essay_write(self, msg, steps):
        """写作文：解析主题/字数/开头/结局/手法 → 写作引擎生成"""
        brief = write_brain.parse_brief(msg)
        brief["_save_doc"] = _save_intent(msg)
        if brief["mode"] == "novel":
            return self._write_from_brief(brief, steps)
        if brief["theme"] is None and brief["genre"] is None:
            return ("写什么主题的作文？告诉我题目，"
                    "比如'写一篇关于父爱的作文'，我马上动笔。")
        if brief["theme"] is None and brief["genre"]:
            brief["theme"] = "成长"
        return self._write_from_brief(brief, steps)

    def _novel_write(self, msg, steps):
        """开小说：题材/结局/书名/第一章，一步到位"""
        brief = write_brain.parse_brief(msg)
        brief["_save_doc"] = _save_intent(msg)
        return self._write_from_brief(brief, steps)

    def _novel_fullbook(self, msg, steps):
        """整本小说：作者信息/简介/目录/正文一次生成"""
        brief = write_brain.parse_brief(msg)
        if not brief["genre"]:
            brief["genre"] = "青春校园"
        st = write_brain.write_full_novel(brief)
        self.essay_state = st
        write_brain.save_state(st)
        steps.append("整本小说：《%s》作者%s，%d 章，%d 字"
                     % (st["title"], st["author"],
                        len(st["chapters"]), len(st["text"])))
        return ("全书完成——%s\n\n作者：%s\n\n%s\n\n（想单独扩写某一章，"
                "说「扩写第X章」；想调结局说「结局改成XX」。）"
                % (st["title"], st["author"], st["text"]))

    def _novel_continue(self, steps):
        """长篇续写：按题材节拍推进一章，终章按结局收笔"""
        if not self.essay_state or self.essay_state.get("mode") != "novel":
            return "还没开书呢。说「写一本末日小说」这种，我先给你起第一章。"
        st, ch = write_brain.continue_novel(self.essay_state)
        self.essay_state = st
        write_brain.save_state(st)
        total = len(st["chapters"])
        tail = ""
        if st.get("last_beat") == "结局":
            tail = "（全书完。想接着看可以喊「番外」，或者换个题材再开一本。）"
        elif st.get("last_beat") == "番外":
            tail = "（番外也写完了。开新书吧，题材你挑。）"
        _how = st.get("last_beat") or ("API 主笔" if st.get("by") == "api"
                                       else "续写")
        steps.append("写作工作台：《%s》第 %d 章「%s」（%s），共 %d 章"
                     % (st["title"].strip("《》"), ch["no"], ch["title"],
                        _how, total))
        return "%s\n\n第%s章 %s\n\n%s\n\n%s" % (
            st["title"], write_brain._cn(ch["no"]), ch["title"],
            ch["text"], tail)

    def _essay_edit(self, msg, steps, plan=None):
        """用当前稿和修改记录执行计划，校验失败时保持工作台不变。"""
        if not self.essay_state:
            return "请先提供或生成一篇稿子。"
        state = self.essay_state
        context = dict(state)
        if state.get("mode") == "novel" and state.get("chapters"):
            context["text"] = state["chapters"][-1]["text"]
        result = write_brain.execute_revision(context, msg, plan)
        if not result["ok"]:
            steps.append("写作工作台：未提交修改")
            return result["message"]
        updated = result["state"]
        if updated.get("mode") == "novel" and updated.get("chapters"):
            updated["chapters"][-1]["text"] = result["text"]
            updated["text"] = "\n\n".join(
                "第%s章 %s\n%s" % (write_brain._cn(chapter["no"]), chapter["title"], chapter["text"])
                for chapter in updated["chapters"])
        self.essay_state = updated
        write_brain.save_state(updated)
        steps.append("写作工作台：第%d轮修改（%s）" %
                     (updated["turn"], "；".join(result["changes"])))
        return updated["text"]

    def _essay_review(self, msg, steps):
        """点评作文：用户贴正文（消息够长且不像指令）→ 四维体检；
        没贴正文但有工作台存货 → 点评存货"""
        text = None
        # 消息本身是长正文（不是指令开头的短句）
        if len(msg) > 120 and not ESSAY_WRITE_RE.search(msg):
            text = msg
        elif self.essay_state:
            text = self.essay_state["text"]
        if text is None:
            return ("想让我点评的话，把作文正文发给我，"
                    "我给你按结构、语言、情感、立意四个维度打分。")
        review = write_brain.review(text)
        steps.append("写作工作台：点评完成")
        return review

    def _essay_import(self, msg, steps):
        """改作文：用户喂来一篇现成的稿子 → 接进工作台，然后按反馈改"""
        st = write_brain.import_essay(msg)
        if st is None:
            return "稿子太短了，多给几段我才有得改。"
        self.essay_state = st
        write_brain.save_state(st)
        steps.append("写作工作台：接稿《%s》（%d 字，主题识别：%s）"
                     % (st["title"], st["words"], st["theme"]))
        return ("稿子接住了——《%s》（%d 字，看着像写「%s」的）。\n\n"
                "哪里不满意直接说：改词、换开头、调结局、加细节、"
                "精简都行，我只动你说的地方。"
                % (st["title"], st["words"], st["theme"]))



def brain_turn(buddy, msg):
    """一轮对话：写作请求主自己答；聊天转发给阿呆（账本/缓存/重试判定都在阿呆侧）

    ⚠️ 这里以前是直接 `aidan.brain_turn(buddy.aidan, msg)` ——
    等于**把写作请求也一起丢给了阿呆**，而阿呆只管聊天（它自己文档里就写着
    「写作时走写作引擎，不经过阿呆」）。
    后果：用户说「写一篇关于坚持的1000字作文」→ 进了聊天模块 → 回一句
    「你之前提过…」 → 用户看到的就是"换个说法它就变傻了"。
    **模型从头到尾没出场。**（注：ESSAY_WRITE_RE 在 竹.py 和 阿呆.py 里各有一份，
    但只有 竹.py 这份真正被使用；阿呆.py 那份是没人调用的死副本。）

    修法：先让「主」的雷达看一眼，是写作就主自己答；不是才转给阿呆。
    """
    try:
        if ESSAY_WRITE_RE.search(msg) or NOVEL_RE.search(msg):
            return buddy.reply(msg)
    except Exception:
        pass                      # 雷达出错就照老路走，不能把对话卡死
    return aidan.brain_turn(buddy.aidan, msg)

def run_web(buddy, offline=False):
    """本地小网页（Python 自带 http.server，零依赖）：
    127.0.0.1:8765，只在本机听招呼，不出外网。"""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    buddy.delay = 0.0   # 网页模式不靠 sleep 卡速度，靠前端打字机动画

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):   # 终端不打一串串访问日志，保持清爽
            pass

        def _json(self, obj, code=200):
            data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _text(self, txt, ctype, code=200):
            data = txt.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "%s; charset=utf-8" % ctype)
            if ctype.startswith("text/html"):
                self.send_header("Cache-Control", "no-store")   # 界面永不过期
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/", "/index.html", "/ui"):
                try:
                    with open(UI_PATH, "r", encoding="utf-8") as f:
                        self._text(f.read(), "text/html")
                except FileNotFoundError:
                    self._text("<h3 style='font-family:monospace'>写作台.html 不见了，"
                               "检查它是不是跟 竹.py 放在同一个文件夹。</h3>",
                               "text/html", 500)
            elif path.startswith("/api/history"):
                self._json({"rows": load_history()})
            elif path.startswith("/api/usage"):
                self._json({"tokens": buddy.total_tokens,
                            "edition": buddy.edition})
            elif path.startswith("/api/state"):
                st = buddy.essay_state
                self._json({"state": st if st else None})
            elif path.startswith("/api/options"):
                if write_brain is not None:
                    self._json(write_brain.options())
                else:
                    self._json({"err": "写作引擎没装上"}, 500)
            else:
                self._json({"err": "not found"}, 404)

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            raw_body = self.rfile.read(n)
            try:
                body = json.loads(raw_body.decode("utf-8") or "{}")
            except Exception:
                body = {}
            path = self.path.split("?")[0]
            if path.startswith("/api/chat"):
                msg = (body.get("msg") or "").strip()
                if not msg:
                    self._json({"err": "空话我可接不住"}, 400)
                    return
                with CHAT_LOCK:               # 一问一答排队，账本不打架
                    reply, steps = brain_turn(buddy, msg)
                self._json({"reply": reply, "steps": steps,
                            "tokens": buddy.total_tokens})
            elif path == "/api/write":
                # 写作模式主接口（一次成篇版）：结构化参数 + 其他要求合并
                with CHAT_LOCK:
                    if write_brain is None:
                        self._json({"err": "写作引擎没装上"}, 500)
                        return
                    brief = buddy._brief_merge(body)
                    steps = []
                    reply = buddy._write_from_brief(brief, steps)
                    buddy.add_tokens(len(reply))
                    buddy.mem.log_chat("你", body.get("msg") or "【写作模式】")
                    buddy.mem.log_chat("竹", reply)
                self._json({"reply": reply, "steps": steps,
                            "state": buddy.essay_state,
                            "tokens": buddy.total_tokens})
            elif path.startswith("/api/write_stream"):
                # 流式写作：NDJSON 逐段下发 {"delta": "..."}，末行 {"done":...}
                # Plus/Pro 配了 Key → 大模型真流式；免费版 → 本地引擎逐段吐
                if write_brain is None:
                    self._json({"err": "写作引擎没装上"}, 500)
                    return
                self.send_response(200)
                self.send_header("Content-Type",
                                 "application/x-ndjson; charset=utf-8")
                self.end_headers()

                def emit(obj):
                    self.wfile.write((json.dumps(obj, ensure_ascii=False)
                                      + "\n").encode("utf-8"))
                    self.wfile.flush()

                with CHAT_LOCK:
                    steps = []
                    action = body.get("action") or "write"
                    use_llm = buddy.llm.available() \
                        and (buddy.edition in ("Plus", "Pro")
                             or getattr(buddy.llm, "is_local", False)) \
                        and action == "write"
                    full = None
                    if use_llm:
                        brief = buddy._brief_merge(body)
                        if not brief["theme"]:
                            emit({"err": "先选个主题（或在其他要求里写清楚），"
                                         "大模型才知道往哪儿下笔"})
                            return
                        system = ("你是写作助手「竹」，中文写作功底扎实。严格服从用户的"
                                  "文体、主题、字数、开头方式、结尾走向、结构、模仿名家"
                                  "和其他要求；正文之外不要任何解释。")
                        ask = ("写一篇关于「%s」的%s，%d字左右。" %
                               (brief["theme"], brief["style"], brief["words"]))
                        if brief["mode"] == "novel" and brief["genre"]:
                            ask = "开一本%s小说的第一章，%d字左右。" % (
                                brief["genre"], brief["words"])
                        if brief["opening"] and brief["opening"] != "开门见山":
                            ask += "开头用「%s」。" % brief["opening"]
                        if brief["ending"]:
                            ask += "结尾走向「%s」。" % brief["ending"]
                        if brief["structure"]:
                            ask += "结构用「%s」。" % brief["structure"]
                        if brief["master"]:
                            ask += "模仿「%s」的笔意（仿其神不抄其文）。" % brief["master"]
                        if brief["technique"]:
                            ask += "表现手法用「%s」。" % brief["technique"]
                        ex = brief.get("extra") or {}
                        if ex.get("raw"):
                            ask += "其他要求：%s" % ex["raw"]
                        chunks = []

                        def on_delta(d):
                            chunks.append(d)
                            try:
                                emit({"delta": d})
                            except Exception:
                                pass

                        try:
                            buddy.llm.chat_stream(system, ask, on_delta)
                            full = "".join(chunks).strip()
                            mt = re.search(r"《(.+?)》", full)
                            st = {"mode": "essay",
                                  "theme": brief["theme"] or "随笔",
                                  "title": mt.group(1) if mt else "无题",
                                  "text": full, "turn": 0,
                                  "style": brief["style"],
                                  "words": brief["words"],
                                  "ending": brief["ending"],
                                  "depth": brief["depth"]}
                            buddy.essay_state = st
                            write_brain.save_state(st)
                            steps.append("大模型流式写作：《%s》%d 字"
                                         % (st["title"], len(full)))
                        except Exception as e:
                            emit({"delta": "\n\n（大模型没接上：%s，改用本地引擎）\n\n" % e})
                            full = None
                    if full is None:
                        if action == "fullbook":
                            brief = buddy._brief_merge(body)
                            if not brief["genre"]:
                                brief["genre"] = "青春校园"
                            st = write_brain.write_full_novel(brief)
                            buddy.essay_state = st
                            write_brain.save_state(st)
                            full = st["text"]
                            steps.append("整本小说：《%s》%d 章 %d 字"
                                         % (st["title"], len(st["chapters"]),
                                            len(full)))
                        elif action == "continue":
                            if not buddy.essay_state \
                                    or buddy.essay_state.get("mode") != "novel":
                                emit({"err": "先开一本小说再续写"})
                                return
                            st, ch = write_brain.continue_novel(buddy.essay_state)
                            buddy.essay_state = st
                            write_brain.save_state(st)
                            full = "%s\n\n第%s章 %s\n\n%s" % (
                                st["title"], write_brain._cn(ch["no"]),
                                ch["title"], ch["text"])
                            steps.append("长篇续写：《%s》第 %d 章"
                                         % (st["title"], ch["no"]))
                        else:
                            brief = buddy._brief_merge(body)
                            full = buddy._write_from_brief(brief, steps)
                    # 逐段下发（本地引擎是秒出的，稍微加一点节奏感）
                    piece = 160
                    sent = 0
                    while sent < len(full):
                        emit({"delta": full[sent:sent + piece]})
                        sent += piece
                        time.sleep(0.04)
                    buddy.add_tokens(len(full))
                    buddy.mem.log_chat("你", body.get("msg")
                                       or ("【续写】" if action == "continue"
                                           else "【写作模式】"))
                    buddy.mem.log_chat("竹", full)
                    emit({"done": True, "steps": steps,
                          "state": buddy.essay_state,
                          "tokens": buddy.total_tokens})
            elif path.startswith("/api/import_file"):
                # 文件导稿：.txt / .md / .docx 直接进工作台，不用粘贴
                name = urllib.parse.parse_qs(
                    urllib.parse.urlparse(self.path).query).get(
                    "name", [""])[0]
                try:
                    import io as _io
                    import zipfile
                    raw = raw_body
                    text = ""
                    if name.lower().endswith(".docx"):
                        z = zipfile.ZipFile(_io.BytesIO(raw))
                        xml = z.read("word/document.xml").decode("utf-8",
                                                                 "ignore")
                        xml = xml.replace("</w:p>", "\n").replace("<w:br/>",
                                                                  "\n")
                        text = _html.unescape(re.sub(r"<[^>]+>", "", xml))
                    else:
                        for enc in ("utf-8", "gb18030", "utf-16"):
                            try:
                                text = raw.decode(enc)
                                break
                            except Exception:
                                continue
                except Exception as e:
                    self._json({"err": "文件读不出来（%s）" % e}, 400)
                    return
                text = text.replace("\r\n", "\n").strip()
                if not text:
                    self._json({"err": "文件里没有文字"}, 400)
                    return
                with CHAT_LOCK:
                    steps = []
                    reply = buddy._essay_import(text, steps)
                    buddy.add_tokens(len(reply))
                    buddy.mem.log_chat("你", "【导入文件】%s" % (name or "正文"))
                    buddy.mem.log_chat("竹", reply)
                self._json({"reply": reply, "steps": steps,
                            "state": buddy.essay_state,
                            "tokens": buddy.total_tokens})
            elif path.startswith("/api/continue"):
                # 长篇续写：推进下一章
                with CHAT_LOCK:
                    steps = []
                    if not buddy.essay_state \
                            or buddy.essay_state.get("mode") != "novel":
                        self._json({"err": "先开一本小说再续写"}, 400)
                        return
                    reply = buddy._novel_continue(steps)
                    buddy.add_tokens(len(reply))
                    buddy.mem.log_chat("你", "续写下一章")
                    buddy.mem.log_chat("竹", reply)
                self._json({"reply": reply, "steps": steps,
                            "state": buddy.essay_state,
                            "tokens": buddy.total_tokens})
            elif path.startswith("/api/revise"):
                # 局部修改：上一篇 + 反馈 → 精准动刀
                fb = (body.get("feedback") or body.get("msg") or "").strip()
                if not fb:
                    self._json({"err": "说说哪里不满意"}, 400)
                    return
                with CHAT_LOCK:
                    steps = []
                    if not buddy.essay_state:
                        self._json({"err": "工作台还没有稿子，先写一篇或导入一篇"},
                                   400)
                        return
                    reply = buddy._essay_edit(fb, steps)
                    buddy.add_tokens(len(reply))
                    buddy.mem.log_chat("你", "【改稿】" + fb)
                    buddy.mem.log_chat("竹", reply)
                self._json({"reply": reply, "steps": steps,
                            "state": buddy.essay_state,
                            "tokens": buddy.total_tokens})
            elif path.startswith("/api/import"):
                # 改作文：把现成的稿子接进工作台
                text = (body.get("text") or "").strip()
                if not text:
                    self._json({"err": "稿子是空的"}, 400)
                    return
                with CHAT_LOCK:
                    steps = []
                    reply = buddy._essay_import(text, steps)
                    buddy.add_tokens(len(reply))
                    buddy.mem.log_chat("你", "【导稿】%s…" % text[:40])
                    buddy.mem.log_chat("竹", reply)
                self._json({"reply": reply, "steps": steps,
                            "state": buddy.essay_state,
                            "tokens": buddy.total_tokens})
            elif path.startswith("/api/review"):
                with CHAT_LOCK:
                    steps = []
                    text = (body.get("text") or "").strip()
                    if text:
                        reply = write_brain.review(text)
                    else:
                        reply = buddy._essay_review("", steps)
                    buddy.add_tokens(len(reply))
                    buddy.mem.log_chat("你", "点评这篇作文")
                    buddy.mem.log_chat("竹", reply)
                self._json({"reply": reply, "steps": steps,
                            "tokens": buddy.total_tokens})
            elif path.startswith("/api/config"):
                # 版本 / 模型 / 服务器地址 / API Key
                cfg = buddy.cfg
                for k in ("edition", "model", "server_url", "api_key",
                          "base_url"):
                    if k in body:
                        cfg[k] = (body[k] or "").strip()
                save_config(cfg)
                buddy.aidan.apply_edition()   # 版本/模型门禁统一在阿呆侧重载
                ok, msg = key_verify(cfg.get("api_key"))
                self._json({"ok": True, "edition": buddy.edition,
                            "key_ok": ok, "key_msg": msg,
                            "msg": "设置已保存。版本：%s" % buddy.edition})
            elif path.startswith("/api/account/register"):
                ok, msg = account_register(body.get("user"),
                                           body.get("pwd"))
                self._json({"ok": ok, "msg": msg})
            elif path.startswith("/api/account/login"):
                ok, msg = account_login(body.get("user"), body.get("pwd"))
                if ok:
                    buddy.current_user = (body.get("user") or "").strip()
                self._json({"ok": ok, "msg": msg,
                            "user": getattr(buddy, "current_user", None)})
            elif path.startswith("/api/keys/list"):
                rows, msg = key_list(getattr(buddy, "current_user", None))
                self._json({"ok": rows is not None, "rows": rows or [],
                            "msg": msg, "max": MAX_KEYS_PER_ACCOUNT})
            elif path.startswith("/api/keys/create"):
                key, msg = key_create(getattr(buddy, "current_user", None),
                                      body.get("plan") or "plus")
                self._json({"ok": key is not None, "key": key, "msg": msg,
                            "max": MAX_KEYS_PER_ACCOUNT})
            elif path.startswith("/api/keys/revoke"):
                ok, msg = key_revoke(getattr(buddy, "current_user", None),
                                     body.get("key") or "")
                self._json({"ok": ok, "msg": msg})
            elif path.startswith("/api/direction"):
                # 作文方向：add（出题 + 贴范文）/ del
                if write_brain is None:
                    self._json({"err": "写作引擎没装上"}, 500)
                    return
                act = body.get("action")
                if act == "add":
                    ok, msg = write_brain.direction_add(body.get("name"),
                                                        body.get("samples"))
                elif act == "del":
                    ok, msg = write_brain.direction_del(body.get("name"))
                else:
                    ok, msg = False, "action 要是 add 或 del"
                self._json({"ok": ok, "msg": msg,
                            "directions": write_brain.direction_options()["directions"]})
            elif path.startswith("/api/verify"):
                ok, msg = key_verify(body.get("key"))
                self._json({"ok": ok, "msg": msg})
            elif path.startswith("/api/clear"):
                with CHAT_LOCK:
                    buddy.aidan.mem.clear_all()
                    buddy.aidan.total_tokens = 0
                    aidan.save_total_tokens(0)
                    aidan.ANSWER_CACHE.clear()
                self._json({"ok": True, "msg": "聊天记录、长期记忆、思考过程全清了"})
            else:
                self._json({"err": "not found"}, 404)

    srv, port = None, WEB_PORT
    for cand in range(WEB_PORT, WEB_PORT + 10):
        try:
            srv = ThreadingHTTPServer((os.environ.get("ZHU_BIND", "127.0.0.1"), cand), Handler)
            port = cand
            break
        except OSError:
            print("  端口 %d 被占（多半已经开着一个竹），换下一个…" % cand)
    if srv is None:
        print("  8765~8774 全被占用了。把旧的竹窗口关掉，再启动一次。")
        return
    url = "http://127.0.0.1:%d" % port
    print("=" * 58)
    print("  竹 · 写作助手 2.0 网页界面已启动（本地小服务器，只在本机转，不经过外网）")
    print("  地址：%s" % url)
    print("  浏览器没自动弹开的话，就把上面这行地址粘进浏览器。")
    print("  关掉这个窗口 = 竹下班；想回老式黑窗口聊天加 --cli")
    print("=" * 58)
    try:
        webbrowser.open(url)
    except Exception:
        pass
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n竹: 关店收工。今天的对话都存进 记忆/ 了，下次接着聊。")


# ----------------------------------------------------------------------
# 主程序
# ----------------------------------------------------------------------
def enable_vt():
    """给 Windows 控制台开 ANSI 魔法（光标能跳回左上角）；
    开不了就返回 False，状态行改成普通打印"""
    if os.name != "nt":
        return True
    try:
        import ctypes
        h = ctypes.windll.kernel32.GetStdHandle(-11)
        mode = ctypes.c_uint32()
        if not ctypes.windll.kernel32.GetConsoleMode(h, ctypes.byref(mode)):
            return False
        if not (mode.value & 0x0004):
            ctypes.windll.kernel32.SetConsoleMode(h, mode.value | 0x0004)
        return True
    except Exception:
        return False


def main():
    args = sys.argv[1:]
    offline = "--offline" in args
    demo = "--demo" in args
    think = "--think" in args   # 训练模式：把思考过程打在屏幕上
    cli = "--cli" in args       # 老式黑窗口聊天；不带就是网页界面

    # 默认走网页界面（还是双击/运行这个 Python 文件启动，浏览器自动弹开）
    if not (cli or demo):
        buddy = Companion(offline=offline, stream_delay=0.0)
        run_web(buddy, offline=offline)
        return

    buddy = Companion(offline=offline,
                      stream_delay=0.0 if demo else 0.02)
    if demo:
        buddy.delay = 0.0

    VT = enable_vt()

    def show_status(total, speed="-"):
        """左上角状态行：token 用量 + 上一条吐字速度（每次答完自动刷新）"""
        txt = ("  token:%d ｜ 上次:%s" % (total, speed)).ljust(36)
        if VT:
            sys.stdout.write("\x1b7\x1b[H" + txt + "\x1b8")
            sys.stdout.flush()
        else:
            print(txt)

    show_status(buddy.total_tokens)
    print("=" * 58)
    print("  竹 · 写作助手 2.0 —— 会记事、会安慰人、会上网查资料整合的伙伴")
    print("  记忆/思考全部以 TXT 中文存储到 记忆/ 文件夹")
    print("  (%s%s | 退出输入 quit / exit | 连按 3 次 1 清空记忆)" %
          ("断网模式" if offline else "联网整合默认开启",
           " | 思考过程可见" if think else ""))
    print("=" * 58)

    def respond(msg):
        # 走和网页同一条"一轮对话"核心路：retry 判定、缓存、记账都一致
        reply, steps = brain_turn(buddy, msg)
        if think and steps:
            print("  [思考] " + "；".join(steps))
        print("竹: ", end="", flush=True)
        t0 = time.time()
        buddy.stream(reply)
        dt = time.time() - t0
        show_status(buddy.total_tokens,
                    ("%.0f字/秒" % (len(reply) / dt)) if dt > 0 else "-")
        print("-" * 58)

    if demo:
        print("\n[演示模式] 看它怎么记事、安慰人、接旧账：\n")
        # 清掉上次演示留下的记忆，保证演示效果干净（正式聊天不受影响）
        for name in ("chat_log.txt", "facts.txt", "thinking.txt"):
            p = os.path.join(MEM_DIR, name)
            if os.path.exists(p):
                os.remove(p)
        script = [
            "你叫什么名字？",
            "我今天吃的那个苹果好酸。",
            "烦死了，今天上班被领导骂了一顿。",
            "今天我还因为这个酸苹果专门去刷了个牙。",
            "Python 的创始人是谁，能给我讲讲他的个人简介吗？",
        ]
        for line in script:
            print("你: %s" % line)
            respond(line)
            time.sleep(0.2)
        print("演示结束。直接运行  python 竹.py  就能跟它聊天。")
        return

    print("\n开始聊天吧。你说过的每句话它都会记进 TXT。\n")
    # 连按 3 次「1」= 清空全部记忆（这个暗号不经过"大脑"，直接执行）
    ones = 0
    while True:
        try:
            line = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n竹: 诶？你走了？我把今天的对话都存好了，下次接着聊。")
            break
        if not line:
            continue
        if line.lower() in ("quit", "exit", "退出"):
            print("竹: 哦。那拜拜——放心，聊天记录都存进 记忆/ 了。")
            break
        if line != "1":
            ones = 0
            respond(line)
            continue
        ones += 1
        buddy.mem.log_chat("你", "1")
        if ones == 1:
            r = "嗯？一个 1。连按 3 次 1 我会把三本账全清空，你这是在试我吗？"
        elif ones == 2:
            r = "第 2 个 1……我可警告你，再来一个，聊天记录、长期记忆、思考过程就全没了。"
        else:
            buddy.mem.clear_all()
            r = "好，收到 3 个 1。记事本全部清空，我现在跟刚出厂一样干净。"
            ones = 0
        print("竹: ", end="", flush=True)
        t0 = time.time()
        buddy.stream(r)
        dt = time.time() - t0
        buddy.add_tokens(len(r))
        show_status(buddy.total_tokens,
                    ("%.0f字/秒" % (len(r) / dt)) if dt > 0 else "-")
        buddy.mem.log_chat("竹", r)
        print("-" * 58)


if __name__ == "__main__":
    main()
