#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
开始.py —— 你说人话，1.5B 把它翻译成指令，然后执行
================================================================================
用法（魔搭实例终端里）：
    python 开始.py

进去之后直接打人话，比如：
    写一部3章的科幻小说，主题是火星救援
    写一篇关于坚持的1000字作文
    把生成的东西放到桌面
    今天天气真不错

绝大多数指令用**规则直接解析**（0 token，不调模型）；拿不准的才让 1.5B 兜底翻译。
改主题/字数/章数这类修改也是规则直改（治「换成母亲翻成父亲」），比以前省一大半 token。
写长篇会**逐章打印**，写完自动存档 + 上云；本机跑还会**自动落到系统桌面**。

先做一次自检：python 开始.py --check
"""
import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
CAND = [HERE, "/mnt/workspace/zhu", os.path.join(HERE, "zhu")]
ROOT = next((d for d in CAND if os.path.exists(os.path.join(d, "竹.py"))), HERE)
os.chdir(ROOT)
API = "http://127.0.0.1:8766/chat/completions"
HEALTH = "http://127.0.0.1:8766/health"
TOKEN = os.environ.get("MS_TOKEN", "ms-3cdacbf0-3908-48f4-ad7b-a8d9538f8332")
REPO = "bg20130613/zhu-train"
GREEN, YELLOW, RED, CYAN, DIM, RESET = (
    "\033[32m", "\033[33m", "\033[31m", "\033[36m", "\033[2m", "\033[0m")
STATS = "http://127.0.0.1:8766/stats"
VERSION = "0919-2390a"    # 每次上传云端前改这里；启动横幅能看出实例跑的是哪一版
# 0919-2390a：为「换大基座」铺路 —— find_base() 改成**优先读 adapter 里记的
#   base_model_name_or_path**。原来只靠猜目录名（models/base 或 Qwen--*/snapshots/*），
#   换基座后会悄悄加载成 1.5B，表现为「换了模型却没效果」。
# 0919-2380a：★ 多候选择优（用户明确「慢无所谓，要的是质量」）
#   写作引擎 加 _quality_score / _pick_best + ESSAY_CANDIDATES=3：
#   一篇作文生成 3 个候选（温度逐个抬 0.08 保证有差异），
#   用纯规则裁判挑最好的 —— 看复读率 / 套话数 / **具体量词与动作词的密度**。
#   实测区分度：好文本 128.1 分 vs 套话堆砌文本 -11.0 分，择优正确。
#   纯规则、不花钱、不依赖任何 API。嫌慢：写作引擎.ESSAY_CANDIDATES = 1
# 0919-2370a：本地写作改成**边生成边打印**（原来干等两分钟像死机）。
#   用户不接受「本地写 2 分钟」，但我们不接第三方 API —— 根子是他跑在 CPU 实例上。
#   本地这条路的体验能改善的部分：让字一段段往外冒（_local_chat stream_out）。
# 0919-2360a：★ 云端写作限流时，自动改用本地模型写
#   用户实测被商汤 429 限流（免费额度约每 5 小时 500 次），想试写作却被卡住。
#   而竹**本来就是本地小模型 + LoRA 训出来写作的**，云端只是外挂 ——
#   所以写作引擎 加 LOCAL_WRITER + _local_alive/_local_chat：
#   云端失败（429/网络）→ 立即切本地 8766 写（不受额度限制，CPU 上约每分钟 500 字）。
#   并优化：429 且有本地兜底时**不再干等 3 次退避**（省 ~30 秒），直接切。
#   一行关掉：写作引擎.LOCAL_WRITER = False
# 0919-2350a：加 --doctor（一条命令体检）。
#   起因：用户照我说的跑 查引擎状态.py，报 No such file —— 因为那个脚本是刚上传的，
#   他那边还是 22:35 启动时同步的旧目录。诊断工具不该依赖「你恰好同步过」，
#   所以并进 开始.py：--doctor 会先 sync_code 再跑体检。
# 0919-2340a：★ 写作失败不再只说「没写出来」
#   用户在实例上写作文失败，屏幕上只有一句笼统提示 —— 因为写作走的是**云端 API**
#   （配置.json 的 base_url/api_key），跟本地 1.5B 服务是两回事，而 _api_chat 把所有
#   异常都静默吞了。现在：①_api_chat 把真实原因留在 _LAST_API_ERR（区分 429 限流 /
#   401 key 无效 / 404 名字错 / 网络不通）②开始.py 失败时把原因打出来
#   ③新增 查引擎状态.py：配置 + 模型服务 + **API 实测**（真发一个最小请求）+ 产物，
#     一屏看清卡在哪。本机实测：商汤 2.3 秒通。
# 0919-2330a：★ 治「同一句话换个说法就不知道了」
#   根因：本地知识库和联网守卫都是**正则白名单**，换词就漏 —— 实测 84 种问法漏 24 种（29%）。
#   改法：①_ABOUT_ZHU 每条升级为 (正则, 关键词表, 答案)，**正则命中或命中任一关键词**都算认得出
#   ②前置守卫放宽：「底层是什么模型」这种没主语的短问句（≤8字）不再被挡
#   ③补两条：身份条目加「你是什么/什么东西/是啥」；新增「参数/怎么训练」条目
#   ④收紧开发者正则应那个裸的「做」（「他为什么这么做」被误判成问作者）
#   实测：84 种问法 0 漏，14 种闲聊 0 误伤。新增 _测试_问法覆盖.py 守住这条。
# 0919-2320a：①修 run_essay 的空值 —— 作文生成失败返回 None 时，
#   str(None) 会让屏幕上出现一篇正文是「None」的作文（还照样存档上云）。
#   ②新增 _验证_作文提示词.py：用真 API 跑一篇，量句级重复/4-gram/套话命中。
#   **首跑结果**：1087 字、句级重复 0.0%、4-gram 1.7%、套话无。
#   去参数化提示词有效：细节全是自己长的（左耳贴表背、虎口旧疤、锉废的29个零件）。
# 0919-2310a：修流式「首段被吃掉」——为了先攒几个字确认开头干净，
#   攒下那十来个字原来没吐出去（只吐了最后一片），表现为回答开头少几个字。
#   另新增 _测试_端到端.py：起假模型服务，把竹这一侧（流式解析/参数传递/
#   假轮次清洗/退化重试）全链路跑一遍，本机可跑，不依赖真模型。
# 0919-2300a：★ 扩展档（真的挂一个更大的模型）+ 一键下载
#   本地模型服务.py 改成**双模型懒加载**：MODELS{main,ext}，扩展档只在升档那一刻才加载，
#   平时 0 占用；generate(which=) 按档取模型；加载失败自动退回主档。
#   开始.py：EXT_MODEL="models/ext"，目录里有模型就自动带上 --ext-model（用户不用记参数）；
#   _svc_is_stale() 现在也比对扩展档配置，配置变了会重启服务。
#   新增 下载扩展档.py（魔搭直连，默认 3B）。
# 0919-2290a：★ 两档推理（用户要的「平时只激活一部分，10 秒答不出就全开」）——
#   Qwen2.5-1.5B 是稠密模型没有部分激活开关，但在服务层做出了同样的效果：
#   快档（≤120 token + _Deadline 到点自停）→ 超时或结果不合格 → 全档全开。
#   只在短回答场景启用（max_new≤400），写长章节直接全档。
#   另：/health 带源码指纹，开始.py 发现模型服务是旧代码会**自动重启它**
#   （服务是独立进程，这才是「改了没生效」的隐形元凶）。
# 0919-2280a：★ 修「模型复述我的指令」——我在 prompt 里写了「不要写「对方：」」这种
#   **否定式示例**，1.5B 分不清指令和正文，反过来照抄出来当回答（截图实测：
#   「不要写「对呀，有意思…」「不要写「对方：」」）。治本：prompt 只给正面要求，
#   不给否定的例子；另在 _cut_fake_turns 加一道「开头就在讲不要写X → 整段作废」。
# 0919-2260a：①「问竹自己的事却被判成知识题去联网」——本地库加「蒸馏/抄袭」条目，
#   _SELF_TOPIC_RE 拦下自我话题不再联网（实测：问「蒸馏算不算盗窃」→ 联网搜到一堆废料
#   → 模型崩成满屏「哦，哦，哦」）②_looks_degenerate 换成「用字丰富度」判据，
#   这种「哦/嗯交替」的崩溃终于认得出来了 ③★ 对话改**流式输出**（打字机效果），
#   不再「前两秒像偷懒、后面蹦一大段」；流式下靠 stop + 前缀缓冲 + 边流边检保安全
#   ④_local_answer 的「两个问号就交给模型」守卫放宽（原来误杀了同一问题的两种说法）
# 0919-2110a：长篇三修 —— ①记忆预算（圣经 900→2500 字、前情 500→1500 字、每 3 章刷新、
#   喂 60 章概要）②每章落盘（中断只丢当前章）③单次封顶 20 章（不再一口气跑十几小时）
#   ④字数→章数不再被夹到 60（40 万字不会再变 12 万字）
# 0919-2200a：补上下文与写作 —— ①对话记忆（hist 落盘，自动重启也不丢）
#   ②长篇句级跨章去重 ③前情提要改成「概要+结尾」④早期卷归档（每 20 章一卷，总量封顶）
#   ⑤生成时显示「竹正在进行思考……（预计 N）」并原地走秒
# 0919-2230a：人设与文风 —— ①自我认知答短、说人话（原来像念产品说明书）
#   ②开发者改回「一个普通学生，要上课写作业，挤时间弄的，不完美」
#   ③修正 Kimi 与 DeepSeek V4 PRO 的表述 ④作文/小说提示词去参数化，禁止填表式生成
# 0919-2240a：★ 修「只说了半句话，屏幕却蹦出一整段自问自答（连『用户：』都替他写了）」
#   —— 根因是小模型把 prompt 里的「用户：/竹：」转写格式当成要接着续写的文本。
#   ①所有 prompt 改成「对方说：「…」」并明令禁止替对方说话/续对话
#   ②chat() 出口统一走 _cut_fake_turns 清洗（从模型自己编的说话人前缀处截断）
#   ③清洗后为空 → 丢弃重试一次，仍不行就给一句短的兜底，绝不把胡话丢给用户
# 0919-2250a：再加固 + 同步网页版
#   ①chat() 请求带 stop 序列（一吐「用户：/对方：/你：」就当场掐断，比事后清洗更硬）
#   ②闲聊 max_tokens 220→90（本来就该短，也压缩了它跑偏的空间）
#   ③本地模型服务.py：stop_strings 需 transformers≥4.39，老版本自动跳过不报错
#   ④★ 阿呆.py（网页版另一条链）的 DEV_REPLY/INTRO_REPLY 同步改写 ——
#     之前只改了 开始.py，网页版走的是阿呆.py，那边完全是另一套话术

GEN_WPS = 8.0            # 生成速度兜底（字/秒）；能连上模型服务就按 /stats 的实测值算
ESCALATE_AFTER = 10.0    # 两档推理：快档跑这么多秒还没拿住 → 全档全开
EXT_MODEL = "models/ext"  # 扩展档模型目录（相对项目根）；放了模型就自动启用，升档时才加载


def say(m, c=""):
    print("%s%s%s" % (c, m, RESET), flush=True)


# ────────────────── 「竹正在进行思考……」计时提示
def _fmt_sec(s):
    s = int(max(0, s))
    if s < 60:
        return "%d 秒" % s
    m, sec = divmod(s, 60)
    if m < 60:
        return "%d 分 %d 秒" % (m, sec)
    h, m = divmod(m, 60)
    return "%d 小时 %d 分" % (h, m)


def _gen_wps():
    """生成速度（字/秒）：优先用模型服务 /stats 的实测值，拿不到才用兜底值。
    这样「预计时间」会随机器实际快慢自己校准，而不是一个写死的数字。"""
    try:
        with urllib.request.urlopen(STATS, timeout=2) as r:
            d = json.loads(r.read().decode("utf-8"))
        v = float(d.get("tokens_s") or 0)
        if v > 0.5:
            return v
    except Exception:
        pass
    return GEN_WPS


_THINK = {"stop": None, "th": None}


def think_start(text="竹正在进行思考", est_sec=0, extra=""):
    """原地显示「竹正在进行思考……（预计 N）」并走秒。配对调用 think_stop()。
    非 tty（输出重定向到日志）时只打一行，不刷屏。"""
    think_stop()
    est = max(1, int(est_sec or 0))
    t0 = time.time()
    ev = threading.Event()
    tail = ("（%s）" % extra) if extra else ""
    tty = sys.stdout.isatty()
    line_txt = ("  %s%s……%s %s预计约 %s%s · 已用 %s%s"
                % (CYAN, text, RESET, DIM, _fmt_sec(est), tail,
                   _fmt_sec(0), RESET))

    def render():
        el = time.time() - t0
        left = max(0, est - el)
        line = ("  %s%s……%s %s预计约 %s%s · 已用 %s%s"
                % (CYAN, text, RESET, DIM, _fmt_sec(left), tail,
                   _fmt_sec(el), RESET))
        sys.stdout.write("\r" + line + " " * 10)
        sys.stdout.flush()

    if not tty:
        print(line_txt, flush=True)
        _THINK["stop"] = ev
        return

    def loop():
        while not ev.is_set():
            try:
                render()
            except Exception:
                return
            ev.wait(0.5)

    _THINK["stop"] = ev
    th = threading.Thread(target=loop, daemon=True)
    _THINK["th"] = th
    th.start()


def think_stop():
    """结束提示：清掉走秒那行（tty）。"""
    ev = _THINK.get("stop")
    if ev is None:
        return
    ev.set()
    th = _THINK.get("th")
    if th is not None:
        th.join(timeout=1.0)
    if sys.stdout.isatty():
        sys.stdout.write("\r" + " " * 98 + "\r")
        sys.stdout.flush()
    _THINK["stop"] = None
    _THINK["th"] = None


_WORD_REQ = re.compile(r"约\s*(\d{3,5})\s*字")


def _est_seconds(prompt, max_tokens=0):
    """估生成秒数：优先按 prompt 里写的「约 N 字」，没有就按 max_tokens 的六成估
    （模型常在 max_tokens 之前就收尾，按满值估会显得一直在超时）"""
    m = _WORD_REQ.search(prompt or "")
    if m:
        n = int(m.group(1))
    else:
        n = int(int(max_tokens or 600) * 0.6)
    return max(6, int(n / max(1.0, _gen_wps())) + 4)


def think_run(text, est_sec, fn, *a, **kw):
    """把一次阻塞调用包上「竹正在进行思考……」提示"""
    think_start(text, est_sec)
    try:
        return fn(*a, **kw)
    finally:
        think_stop()


def install_thinking():
    """把「竹正在进行思考……」接到引擎的每一次模型调用上。

    引擎里有十几处 _api_chat（全书设定 / 大纲 / 逐章 / 圣经 / 卷归档 / 续写…），
    逐个改既容易漏，又要每加一个调用点就回来补一次；在这里统一包一层最稳。"""
    import 写作引擎 as we
    if getattr(we, "_think_wrapped", False):
        return
    orig = we._api_chat

    def wrapped(system, user, max_tokens, temperature=0.85, **kwargs):
        think_start("竹正在进行思考", _est_seconds(user, max_tokens))
        try:
            return orig(system, user, max_tokens, temperature, **kwargs)
        finally:
            think_stop()

    we._api_chat = wrapped
    we._think_wrapped = True


# ────────────────────────────── 模型调用
# ★★ 严重问题（2026-09-19 用户实测）：只说了半句话，屏幕却蹦出一整段自问自答，
#    连「用户：」都替他写了。根因是小模型把提示里的**对话转写格式**当成了
#    「要接着续写的文本」——提示里出现「用户：X」，它就顺着补「用户：…」「竹：…」。
#    两道防线：
#      ① prompt 里不再用「用户：/竹：」这种转写格式（改「对方说：」+ 明令禁止替对方说话）
#      ② 拿到结果先清洗：碰到模型自己编的说话人前缀，一律从那里截断
_FAKE_TURN_RE = re.compile(
    r"(?:^|\n)\s*(?:用户|对方|人类|访客|提问者|assistant|user|human)\s*[:：]")


def _cut_fake_turns(text):
    """砍掉模型自己接下去的「下一轮对话」，以及它复述我指令的部分。

    处理四种情况：
      · 开头就在替对方说话（整段以「用户：」起头）→ 全是它自己演的，整段丢掉
      · 中途冒出「某某：」→ 从那之后全是它编的，截断
      · 前面挂了个「竹：」→ 那只是它多打的前缀，去掉
      · ★ 复述指令：1.5B 分不清「指令」和「正文」。prompt 里只要出现
        「不要写「对方：」」这种**否定式示例**，它就会把那个示例照抄出来。
        （实测输出：「不要写「对呀，有意思…」」「不要写「对方：」」）
        所以这里也拦一道：开头就是在讲"不要写X"的，整段作废。
        ⚠️ 治本还在 prompt —— 别给否定的例子，只给正面的要求。
    """
    t = (text or "").strip()
    if not t:
        return t
    if re.match(r"^(不要|别|禁止|不能|请勿)\s*(写|输出|说|用|提)\s*[「“\"'（(]", t):
        return ""
    m = _FAKE_TURN_RE.search(t)
    if m:
        if m.start() == 0:
            return ""                     # 整段都在自问自答，一句都不能要
        t = t[:m.start()].strip()
    t = re.sub(r"^(?:竹|你|回答|assistant)\s*[:：]\s*", "", t)
    return t.strip()


_DEGEN_NOISE_RE = re.compile(r"[，。！？、；：…—\s「」“”'\"（）·!?,.:;]")


def _looks_degenerate(text):
    """模型崩了没（实测：问「蒸馏算不算盗窃」→ 满屏「哦，哦，哦……嗯，嗯，嗯」）。

    1.5B 遇到要价值判断的问题会陷入 token 级循环。一旦这样，输出**没有**「某某：」
    这种可清洗的特征，只能靠统计来认：某个字占了非标点字符的很大比例，
    或者同一短片段反复出现。"""
    t = (text or "").strip()
    if len(t) < 24:
        return False
    chars = [c for c in t if not _DEGEN_NOISE_RE.match(c)]
    if len(chars) < 12:
        return False
    cnt = {}
    for c in chars:
        cnt[c] = cnt.get(c, 0) + 1
    # 判据一：用字太贫乏。正常中文 unique/总字 ≈ 0.6~0.9；
    #   「哦，哦，哦…」和「这句话在打转。」×8 这类都只有 0.1 上下。
    if len(cnt) / len(chars) < 0.35:
        return True
    # 判据二：单个字就吃掉非标点字符的三成五以上
    if max(cnt.values()) / len(chars) > 0.35:
        return True
    return False


def _chat_once(prompt, max_tokens, temperature, timeout, on_delta=None):
    """一次请求。给了 on_delta 就走 SSE 流式（边生成边吐字，像真人打字）。

    流式下没法再"等生成完再清洗"（字已经出去了），所以安全靠三层：
      · 服务端带 stop 序列 —— 「某某：」一出现就停
      · 客户端先缓冲前 10 个字，确认开头不是「用户：」这类才往外吐
      · 边走边做退化检测，发现打转（满屏「哦」）立刻掐断
    """
    body = json.dumps({
        "model": "local",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature, "max_tokens": max_tokens,
        "stream": bool(on_delta),
        # 两档推理：平时走「快档」（短、有超时保护），ESCalate_AFTER 秒还拿不住
        # 才「全档」全开。服务端不认识这个字段时会忽略，不会报错。
        "sparse": True,
        "escalate_after": ESCALATE_AFTER,
        "stop": ["\n用户", "用户：", "\n对方", "对方：", "\n你：", "\n竹：",
                 "\nHuman", "\nUser", "Human:", "User:"],
    }).encode("utf-8")
    req = urllib.request.Request(API, data=body,
                                 headers={"Content-Type": "application/json"})

    if not on_delta:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read().decode("utf-8"))
        return _cut_fake_turns(d["choices"][0]["message"]["content"])

    buf, started, dead = "", False, False
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    j = json.loads(data)
                except Exception:
                    continue
                ch = (j.get("choices") or [{}])[0]
                piece = (ch.get("delta") or {}).get("content") or ""
                if not piece:
                    continue
                buf += piece
                if not started:
                    if len(buf) < 10:
                        continue                     # 先攒几个字，确认开头干净
                    if _FAKE_TURN_RE.match(buf):
                        dead = True                  # 在替对方说话 → 一个字都不吐
                        break
                    started = True
                    on_delta(buf)                # ★ 把刚攒的那十来个字补吐出去，
                    continue                     #   否则开头的字会被吃掉（原来就漏在这儿）
                on_delta(piece)
                if len(buf) > 80 and _looks_degenerate(buf[-180:]):
                    dead = True                      # 打转了，别让它继续吐
                    break
    except Exception:
        pass
    if dead:
        return ""
    return _cut_fake_turns(buf)


def _cloud_chat(prompt, max_tokens, temperature, on_delta):
    """本地模型服务不在（演示机 / 腾讯云 / 服务崩了）→ 云端写作 API 顶上对话。
    复用 写作引擎._api_chat（自带 429 退避）；它的"本地回退"在这里不会触发死循环
    —— 本地服务本来就不可用，_local_alive() 返回 False。"""
    import 写作引擎 as we
    try:
        out = we._api_chat("", prompt, max_tokens, temperature)
        out = _cut_fake_turns(out or "")
        if out and _looks_degenerate(out):
            return ""
        if on_delta and out:
            on_delta(out)
        return out
    except Exception as e:
        say("  云端兜底也失败：%s" % str(e)[:70], YELLOW)
        return ""


def chat(prompt, max_tokens=250, temperature=0.2, timeout=300, on_delta=None):
    if not service_alive():
        # ★ 本地模型服务不可用（新机器演示 / 腾讯云 / 服务崩了）→ 云端 API 顶上，
        #   竹在任何一台干净机器上都能开口，不依赖本地模型
        return _cloud_chat(prompt, max_tokens, temperature, on_delta)
    text = _chat_once(prompt, max_tokens, temperature, timeout, on_delta)
    if text and _looks_degenerate(text):
        # 打转了 → 换个说法、降温重来一次（低温度 + 明确禁重复，通常能救回来）
        say("  （它打转了，重来一次）", YELLOW)
        text = _chat_once(
            prompt + "\n\n【重要】直接回答问题本身，不要重复同一个字，"
                     "不要输出「哦」「嗯」这类语气词，一句一句正常写。",
            max_tokens, max(0.1, min(temperature, 0.35)), timeout, None)
        if on_delta and text:
            on_delta(text)          # 重试结果补吐给调用方（刚才那半截已经作废）
        if text and _looks_degenerate(text):
            return ""               # 两次都崩 → 宁可空着，也别把垃圾丢给用户
    return text


def stream_chat(prompt, prefix="竹：", **kw):
    """带「竹正在进行思考……」计时 + 打字机效果的对话。返回完整回复（已打印）。"""
    st = {"on": False}

    def emit(s):
        if not st["on"]:
            think_stop()                       # 清掉走秒那行，换成正文
            sys.stdout.write(prefix)
            st["on"] = True
        sys.stdout.write(s)
        sys.stdout.flush()

    think_start("竹正在进行思考", _est_seconds(prompt, kw.get("max_tokens", 200)))
    text = chat(prompt, on_delta=emit, **kw)
    if not st["on"]:
        think_stop()
        sys.stdout.write(prefix + (text or "……我刚才没想出来，你再说一遍？"))
    sys.stdout.write("\n")
    sys.stdout.flush()
    return text


# ────────────────────────────── 同步代码（新实例/缺文件时救场）
def sync_code(quiet=False):
    try:
        from modelscope.hub.api import HubApi
        from modelscope import snapshot_download
        HubApi().login(TOKEN)
        snapshot_download(REPO, local_dir=ROOT,
                          allow_patterns=["*.py", "*.json", "*.sh", "*.md"])
        if not quiet:
            say("  代码同步完成", GREEN)
        return True
    except Exception as e:
        if not quiet:
            say("  同步失败：%s" % str(e)[:140], YELLOW)
        return False


def ensure_deps():
    say("  检查依赖…")
    miss = []
    for mod, pkg in [("torch", "torch"), ("transformers", "transformers"),
                     ("peft", "peft"), ("accelerate", "accelerate")]:
        try:
            __import__(mod)
        except Exception:
            miss.append(pkg)
    if not miss:
        import torch, transformers
        say("  都在（torch %s / transformers %s）"
            % (torch.__version__, transformers.__version__), GREEN)
        return True
    say("  缺：%s —— 自动安装（约几分钟）" % ", ".join(miss), YELLOW)
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                        "-i", "https://pypi.tuna.tsinghua.edu.cn/simple"] + miss)
    if r.returncode != 0:
        say("  装失败", RED)
        return False
    say("  装好了", GREEN)
    return True


def _health():
    try:
        with urllib.request.urlopen(HEALTH, timeout=2) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception:
        return None


def service_alive():
    return bool((_health() or {}).get("ok"))


def _svc_build():
    """本地模型服务.py 的源码指纹"""
    try:
        import hashlib
        with open(os.path.join(ROOT, "本地模型服务.py"), "rb") as f:
            return hashlib.md5(f.read()).hexdigest()[:8]
    except Exception:
        return ""


def _ext_model_arg():
    """扩展档模型路径：EXT_MODEL 指向的目录里真有 config.json 才返回。

    设计成「有就自动用」—— 用户下完模型不用记任何参数。"""
    p = os.path.join(ROOT, EXT_MODEL) if EXT_MODEL else ""
    return p if p and os.path.exists(os.path.join(p, "config.json")) else ""


def _svc_is_stale():
    """跑着的模型服务是不是旧代码（或者扩展档配置变了）。

    ★ 必须查这个：模型服务是**独立进程**，光重跑 开始.py 换不掉它。
    「改了代码却感觉没生效」有一半的账要记在这里。"""
    d = _health()
    if not d or not d.get("ok"):
        return False                  # 服务根本没跑，谈不上「旧」
    cur = _svc_build()
    if cur and d.get("build") != cur:
        return True
    want = _ext_model_arg()
    have = d.get("ext_model") or ""
    if bool(want) != bool(have):
        return True                   # 一边有扩展档一边没有 → 得重启
    if want and have and os.path.abspath(want) != os.path.abspath(have):
        return True
    return False


def _kill_service():
    """停掉跑着的模型服务（Linux 实例用 pkill；本机 Windows 没有就跳过）"""
    try:
        subprocess.run(["pkill", "-f", "本地模型服务.py"], timeout=5)
        return True
    except Exception:
        return False


def ensure_service(timeout_s=300):
    if service_alive():
        if not _svc_is_stale():
            say("  模型服务已在跑（代码是最新的）", GREEN)
            return True
        say("  模型服务的代码是旧的 → 重启它（模型要重新加载，稍等一下）", YELLOW)
        if not _kill_service():
            say("  停不掉旧服务 —— 请手动停掉再重跑（Linux: pkill -f 本地模型服务.py）",
                RED)
            return False
        time.sleep(2.0)
    svc = os.path.join(ROOT, "本地模型服务.py")
    if not os.path.exists(svc):
        say("  没有 本地模型服务.py，先同步", RED)
        return False
    logf = open(os.path.join(ROOT, "本地模型服务.log"), "w", encoding="utf-8")
    env = dict(os.environ, ZHU_DTYPE=os.environ.get("ZHU_DTYPE", "bfloat16"))
    cmd = [sys.executable, "-u", svc]
    _ext = _ext_model_arg()
    if _ext:
        cmd += ["--ext-model", _ext]
        say("  扩展档：%s（平时不加载，升档时才启用）" % os.path.basename(_ext), DIM)
    p = subprocess.Popen(cmd, stdout=logf,
                         stderr=subprocess.STDOUT, cwd=ROOT, env=env)
    say("  启动中（bf16，约 10~60 秒）…", DIM)
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if service_alive():
            say("  就绪（%.0f 秒）" % (time.time() - t0), GREEN)
            return True
        if p.poll() is not None:
            say("  服务进程退出了，日志尾部：", RED)
            try:
                with open(os.path.join(ROOT, "本地模型服务.log"),
                          encoding="utf-8", errors="replace") as f:
                    for line in f.readlines()[-15:]:
                        print("    " + line.rstrip())
            except Exception:
                pass
            return False
        time.sleep(3)
    say("  超时", RED)
    return False


# ──────────────────── 指令解析（规则优先，1.5B 只兜底 —— 省 token 的关键）
# 三层：① 新写作指令 → 规则直判（0 token）
#      ② 改上一条（换主题/字数/章数）→ 规则直接打补丁（0 token，且不会翻错）
#      ③ 前两层都拿不准 → 才让 1.5B 翻译（提示词已压缩，比旧版省一半）
SYS_TRANSLATE = (
    "把用户的话转成一条JSON指令，只输出JSON本身，不要解释。"
    "字段：intent(novel=写多章小说/essay=写作文文章/chat=其他)、"
    "title(作品名,可空)、theme(题材主题,可空)、chapters(章数,没说填0)、words(字数,没说填0)、"
    "extra(用户说的背景/设定/人物/要求,按原话摘录,没说填空字符串)。\n"
    "例：写一部3章的科幻小说，主题是火星救援 → "
    "{\"intent\":\"novel\",\"title\":\"火星救援\",\"theme\":\"科幻\",\"chapters\":3,\"words\":0,\"extra\":\"\"}\n"
    "例：写一部3章的科幻小说，背景是2200年地球被耀斑摧毁，主角是殖民公司的老板 → "
    "{\"intent\":\"novel\",\"title\":\"火星救援\",\"theme\":\"科幻\",\"chapters\":3,\"words\":0,"
    "\"extra\":\"背景是2200年地球被耀斑摧毁，主角是殖民公司的老板\"}\n"
    "例：写一篇关于坚持的1000字作文 → "
    "{\"intent\":\"essay\",\"title\":\"坚持\",\"theme\":\"坚持\",\"chapters\":0,\"words\":1000,\"extra\":\"\"}\n"
    "例：今天天气真不错 → "
    "{\"intent\":\"chat\",\"title\":\"\",\"theme\":\"\",\"chapters\":0,\"words\":0,\"extra\":\"\"}"
)

_Q_RE = re.compile(
    r"怎么|如何|为什么|什么意思|什么是|什么叫|能不能|可不可以|有没有|介绍|推荐|[?？]")
_VERB_RE = re.compile(r"写|来一?[篇部个章]|创作|帮.{0,8}写")
_NOVEL_RE = re.compile(r"小说|长篇|连载")
_ESSAY_RE = re.compile(r"作文|文章|记叙文|议论文|散文|随笔")
_CH_RE = re.compile(r"([0-9一二两三四五六七八九十百]+)\s*(?:个章节|章)")
_WD_RE = re.compile(
    r"([0-9]{3,5}|[一二两三四五六七八九]+千[零一二三四五六七八九百]*|[一二两三四五六七八九]+百)\s*字")
_CN = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
       "六": 6, "七": 7, "八": 8, "九": 9, "十": 10, "百": 100}
_FOLLOWUP_RE = re.compile(
    r"主题|题材|主人公|主角|人物|题目|标题|书名|结局|开头|结尾|风格|字数|名字|姓名|章|"
    r"换|改|再来|继续|接着|续写|加一?[点章]")


def _cn2int(s):
    """中文数字 → int（三/十五/二十/三十二/一百；阿拉伯数字直接转）"""
    s = (s or "").strip()
    if not s:
        return 0
    if s.isdigit():
        return int(s)
    if "百" in s:
        a, _, b = s.partition("百")
        return _CN.get(a, 1) * 100 + (_cn2int(b) if b else 0)
    if "十" in s:
        a, _, b = s.partition("十")
        return _CN.get(a, 1) * 10 + _CN.get(b, 0)
    return _CN.get(s, 0)


def _parse_fields(msg):
    """从话里抠 章数/字数/主题/题目（规则层共用）"""
    ch = _CH_RE.search(msg)
    wd = _WD_RE.search(msg)
    th = re.search(r"(?:主题|题材)[:：]?\s*([^，。,；;]+)", msg) or \
         re.search(r"关于([^，。,；;]{1,12}?)(?:的|，|,|$)", msg)
    theme = re.sub(r"^(换成|改成|改为|变成|是|为)",
                   "", (th.group(1).strip() if th else "")).strip()
    ti = re.search(r"[《【]([^》】]{1,24})[》】]", msg) or \
         re.search(r"(?:题目|标题|书名)[:：]?\s*([^，。,；;]{1,24})", msg)
    title = (ti.group(1).strip() if ti else "")
    if not theme:
        g = re.search(r"([\u4e00-\u9fa5]{2,4})(?:小说|题材)", msg)
        if g:
            # 剥掉「N章的 / 一部 / 长篇」这类结构前缀，只留纯题材词
            theme = re.sub(r"^(?:长篇|中篇|短篇|[的章回部篇]|[0-9一二两三四五六七八九十百])+",
                           "", g.group(1))
    return (_cn2int(ch.group(1)) if ch else 0,
            _cn2int(wd.group(1)) if wd else 0,
            theme, title)


def _genre_of(msg):
    """抠题材词（科幻/武侠/都市…）：『X小说 / X题材』里的 X。
    主题和题材是两回事：「主题是火星救援」的题材仍是科幻 —— 引擎要分开拿。"""
    g = re.search(r"([\u4e00-\u9fa5]{2,4})(?:小说|题材)", msg or "")
    if not g:
        return ""
    return re.sub(r"^(?:长篇|中篇|短篇|[的章回部篇]|[0-9一二两三四五六七八九十百])+",
                  "", g.group(1))


def _rule_intent(msg):
    """规则快路：常见写作指令 0 token 直判；判不了返回 None（交给 1.5B）。
    故意保守：疑问句、没有写作动词的一律不判 —— 宁可多花 token 也不能判错。"""
    if _Q_RE.search(msg) or not _VERB_RE.search(msg):
        return None
    ch, wd, theme, title = _parse_fields(msg)
    has_novel = bool(_NOVEL_RE.search(msg))
    has_essay = bool(_ESSAY_RE.search(msg))
    if has_novel or (has_essay and ch):
        return {"intent": "novel", "title": title or theme or "新作品",
                "theme": theme or title or "小说",
                "genre": _genre_of(msg),
                "chapters": ch, "words": wd}
    if has_essay:
        return {"intent": "essay", "title": title or theme or "作文",
                "theme": theme or "作文", "chapters": 0, "words": wd}
    return None




def _extract_json(text):
    """从模型输出里稳取第一个完整 JSON 对象。
    模型常在 JSON 前后加解释/代码围栏/第二个对象 —— 直接 json.loads 会报
    "Extra data"。这里用括号配对扫描（字符串内的花括号不算），稳。"""
    text = re.sub(r"```[a-zA-Z]*", "", text or "")
    i = text.find("{")
    if i < 0:
        return None
    depth, instr, esc = 0, False, False
    for j in range(i, len(text)):
        ch = text[j]
        if instr:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                instr = False
        else:
            if ch == '"':
                instr = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[i:j + 1])
                    except Exception:
                        return None
    return None


_FIELD_KEY = {"主人公": "theme", "主角": "theme", "人物": "theme",
              "主题": "theme", "题材": "theme", "背景": "theme",
              "结局": "ending", "开头": "opening"}
_PATCH_FIELD_RE = re.compile(
    r"(主人公|主角|人物|主题|题材|背景|结局|开头)"
    r"(?:名称)?(?:[:：]|换成|改成|改为|变成|为|是)\s*([^，。,；;！!？?]{1,16})")
_PATCH_TITLE_RE = re.compile(
    r"(题目|标题|书名)(?:[:：]|换成|改成|改为|变成|为|是)\s*([^，。,；;！!？?]{1,24})")
_PATCH_WORDS_RE = re.compile(
    r"(?:字数|篇幅)(?:[:：]|换成|改成|改为|变成|为|是)?\s*([0-9]{2,5})\s*字?")
_ADD_RE = re.compile(r"再(加|来|写)|续写|接着写|往后写")
_NAME_HINT_RE = re.compile(
    r"(?:主角|主人公|男主|女主)[^，。,；;！!？?]{0,8}?(?:名字|姓名)"
    r"|(?:主角|主人公|男主|女主)[^，。,；;！!？?]{0,4}?(?:叫做|就叫|叫)")
_NAME_RE = re.compile(r"(?:换成|改成|改为|叫做|就叫|叫|是)\s*([^，。,；;！!？?]{1,12})")
_CONT_WORD_RE = re.compile(
    r"续写|接着(写|来|吧|讲|说)?|继续(写|吧|讲|说)?|往下(写|讲|说)?|"
    r"下一章|再来一章|再写|再加|后面(呢|怎么|怎么样)?|然后呢|还有呢|接下来")
_NEWBOOK_RE = re.compile(r"写一[部本篇]|来一[部本篇]|重新写|换一本|重写一本|新的")


def _patch_prev(msg, prev):
    """★ 换主题/改字数/改章数：规则直接给上一条指令打补丁，0 token。
    比把话丢给 1.5B 重翻准得多 ——「把主人公换成母亲」不会再翻成父亲。
    打不上补丁返回 None。"""
    if not prev or prev.get("intent") not in ("novel", "essay"):
        return None
    # ★ 明显在开新的一本（「写一部…」「换一本…」「重写一本…」）就别往上一本打补丁 ——
    #   否则「写一部小说，主角叫X」会被当成「改上一本的主角名」，
    #   主题/章数/设定全部从上一本继承下来。这正是「怎么又是上次那个主题」的来源。
    if _NEWBOOK_RE.search(msg):
        return None
    if len(msg) > 40 or _Q_RE.search(msg) or re.search(r"吗|对吧|是不是", msg):
        return None
    m2 = _PATCH_FIELD_RE.search(msg)
    mt = _PATCH_TITLE_RE.search(msg)
    mw = _PATCH_WORDS_RE.search(msg)
    mc = _CH_RE.search(msg)
    mn = ""
    if _NAME_HINT_RE.search(msg):          # 换主角名字 → 重写时人物用新名
        ns = [_N.group(1).strip() for _N in _NAME_RE.finditer(msg)]
        ns = [n for n in ns if n and n not in ("我", "你", "他", "她", "它", "自己")]
        mn = ns[-1] if ns else ""
    if not (m2 or mt or mw or mc or mn):
        return None
    cmd = dict(prev)
    old_theme = str(cmd.get("theme") or "")
    if mn:
        cmd["characters"] = "主角的名字必须叫「%s」" % mn
    if m2:
        cmd[_FIELD_KEY[m2.group(1)]] = m2.group(2).strip()
    if mt:
        cmd["title"] = mt.group(2).strip()
    if mw:
        cmd["words"] = int(mw.group(1))
    if mc:
        n = _cn2int(mc.group(1))
        cmd["chapters"] = ((int(cmd.get("chapters") or 3) + n)
                           if _ADD_RE.search(msg)
                           else (n or int(cmd.get("chapters") or 3)))
    if cmd.get("theme") != old_theme and cmd.get("title") in ("", old_theme):
        cmd["title"] = cmd.get("theme")     # 题目跟着主题走（本来就是顺手起的）
    cmd["title"] = str(cmd.get("title") or "")[:24]
    return cmd


def _continue_cmd(msg, prev):
    """★ 续写：沿用上一次的存档接着写，不重开新书。
    再加/再写N章 → 多写 N 章；写到/改成 N 章 → 补到 N 章；
    「第N章」是序数（接着写第二章 = 只写 1 章）；光说续写 → 先续 1 章。"""
    cmd = dict(prev)
    cmd["intent"] = "continue"
    mc = _CH_RE.search(msg)
    ordinal = bool(mc and re.search(r"第\s*%s\s*章" % re.escape(mc.group(1)), msg))
    if mc and not ordinal and _ADD_RE.search(msg):
        cmd["more"] = _cn2int(mc.group(1))
    elif mc and not ordinal:
        cmd["total"] = _cn2int(mc.group(1))
    else:
        cmd["more"] = 1
    return cmd


# ──────────────────── 知识题联网查证（搬自阿呆.py 的 WebSearcher，实测可用线路）
import html as _htmlmod

_TAG_RE = re.compile(r"<[^>]+>")
_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
_KNOWLEDGE_RE = re.compile(
    r"是谁|谁开发|谁发明|谁写的|谁创作|哪家公司|什么公司|是什么|是什么意思|什么叫"
    r"|什么梗|为什么|为啥|怎么回事|多少钱|什么时候|哪一年|哪年|多少年|在哪里|在哪"
    r"|出自哪里|什么来历|介绍一下|评价一下|哪里来的")
_CHITCHAT_RE = re.compile(
    r"^(你好|您好|嗨|哈喽|在吗|哈哈|嘻嘻|谢谢|多谢|再见|拜拜|晚安|早安|早上好|下午好"
    r"|晚上好|没事|哦|嗯嗯|嗯|好的|行吧|行|随便|辛苦了)"
    r"[！!。～~，,\s]*$")
_WRITEISH_RE = re.compile(
    r"写一?[篇本部章]|来一?[篇本部章]|作文|小说|文章|续写|接着写|继续写|再加|写作")
_SHELL_CUT_RE = re.compile(r"&&|cd\s+/|python\s|pip\s+install|bash\s+\S|sudo\s")
_RICH_RE = re.compile(
    r"背景|设定|世界观|地球|人类|文明|公司|末世|末日|王朝|江湖|年代|世纪"
    r"|[0-9]{3,4}\s*年|主角[是在叫有]|我有一|我是一个|职业是|身份是"
    r"|感人|虐|恐怖|惊悚|幽默|搞笑|轻松|温馨|治愈|悲伤|悲剧|爽文|热血"
    r"|第一人称|第三人称|要有|写得好|写得[细长短深]|加点|风格")


def _fetch(url, timeout=8):
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", "ignore")


def _clean_html(t):
    t = re.sub(r"<script.*?</script>", " ", t, flags=re.S | re.I)
    t = re.sub(r"<style.*?</style>", " ", t, flags=re.S | re.I)
    return _htmlmod.unescape(_TAG_RE.sub(" ", t))


def _bing_rss(query):
    """必应 RSS：干净 XML、无反爬，主力线路"""
    url = ("https://www.bing.com/search?q=" + urllib.parse.quote(query)
           + "&format=rss&mkt=zh-CN&count=10")
    page = _fetch(url)
    results = []
    for m in re.finditer(r"<item>(.*?)</item>", page, re.S):
        b = m.group(1)
        ti = re.search(r"<title>(.*?)</title>", b, re.S)
        li = re.search(r"<link>(.*?)</link>", b, re.S)
        de = re.search(r"<description>(.*?)</description>", b, re.S)
        if ti:
            results.append({"title": _clean_html(ti.group(1)).strip(),
                            "snippet": (_clean_html(de.group(1)).strip()[:280]
                                        if de else ""),
                            "url": _clean_html(li.group(1)).strip() if li else ""})
    return results


def _baidu(query):
    """百度网页（备用线路）：被安全验证拦就认输返回空"""
    url = ("https://www.baidu.com/s?wd=" + urllib.parse.quote(query)
           + "&rn=10&ie=utf-8")
    req = urllib.request.Request(url, headers={
        "User-Agent": _UA, "Accept-Language": "zh-CN,zh;q=0.9"})
    with urllib.request.urlopen(req, timeout=8) as resp:
        page = resp.read().decode("utf-8", "ignore")
    if "百度安全验证" in page or "wappass" in page:
        return []
    results = []
    for m in re.finditer(r'<h3[^>]*>.*?<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
                         page, re.S):
        href, title = m.group(1), _clean_html(m.group(2))
        win = page[m.end():m.end() + 4000]
        sn = re.search(r'<span[^>]*class="[^"]*(?:content-right|abstract'
                       r'|c-abstract)[^"]*"[^"]*>(.*?)</span>', win, re.S)
        if title and "百度" not in title[:12] and "安全验证" not in title:
            results.append({"title": title,
                            "snippet": (_clean_html(sn.group(1)).strip()[:280]
                                        if sn else ""), "url": href})
    return results[:5]


def _bili_cookie():
    b3, b4 = "", ""
    try:
        d = json.loads(_fetch("https://api.bilibili.com/x/frontend/finger/spi"))
        b3, b4 = d["data"]["b_3"], d["data"]["b_4"]
    except Exception:
        pass
    return b3, b4


def _bilibili_articles(query, top=3):
    """B 站专栏：梗科普文的摘要往往就是现成解释"""
    b3, b4 = _bili_cookie()
    url = ("https://api.bilibili.com/x/web-interface/search/type"
           "?search_type=article&keyword=" + urllib.parse.quote(query))
    req = urllib.request.Request(url, headers={
        "User-Agent": _UA, "Referer": "https://www.bilibili.com/",
        "Cookie": "buvid3=%s; buvid4=%s" % (b3, b4)})
    with urllib.request.urlopen(req, timeout=8) as resp:
        d = json.loads(resp.read().decode("utf-8", "ignore"))
    out = []
    for a in d.get("data", {}).get("result", [])[:top]:
        title = re.sub(r"<[^>]+>", "", a.get("title", ""))
        if title:
            out.append({"title": title,
                        "snippet": re.sub(r"\s+", " ", a.get("desc", ""))[:280],
                        "url": "https://www.bilibili.com/read/cv"
                               + str(a.get("id", ""))})
    return out


def web_search(query, max_results=5):
    """必应RSS → 百度 → B站专栏；凑够条数就收工，单家挂了不拖累整队"""
    got = []
    hot = any(k in query for k in ("梗", "啥意思", "是什么意思", "出处", "流行语"))
    order = ((_bilibili_articles, _bing_rss) if hot
             else (_bing_rss, _baidu, _bilibili_articles))
    for engine in order:
        try:
            got += engine(query)
        except Exception:
            continue
        if len(got) >= max_results:
            break
    seen, out = set(), []
    for r in got:
        key = r["title"][:24]
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out[:max_results]


# ── 角色设定：闲聊/自我认知问题的上下文。有了它，1.5B 才能用自己的话
#    回答「你是谁/你是千问吗」这类问题的任意问法 —— 查表做不到举一反三，上下文可以。
_PERSONA = (
    "你是「竹」，一个写小说和作文的中文 AI，底座是千问 Qwen2.5-1.5B（跑在魔搭云上）。"
    "教你的是一个人——你的主人，一个普通学生。他平时要上课、写作业，"
    "只能挤课余时间弄你：拿千问当底座，用 Kimi 和 DeepSeek V4 PRO 的输出作参考来教你，"
    "语料是他自己一篇篇挑的；写长文靠他手写的规则引擎一段段搭，查资料也是他自己写的一套检索。"
    "没有团队，也没有公司，就是一个学生自己摸索出来的东西——所以你不完美，别装完美。"
    "说话的语气：像真人聊天，简短、口语、有温度；一般两句话说完就够，别铺排、别端着、"
    "别写成产品说明书；不知道就直说不知道，绝不编造。"
)

# ── 分类①：关于竹本身的高频问题 → 本地知识库直接答（0 token、不联网）
#    注意：这里只放最高频的问法，长尾问法靠上面的角色设定让 1.5B 自己举一反三。
_ABOUT_ZHU = (
    # 每条 = (正则, 关键词表, 答案)
    # ★ 为什么还要关键词：只靠正则，换个说法就漏 —— 实测 84 种问法漏 24 种（29%），
    #   「介绍下你自己」「你从哪来的」「你主人是谁」「底层是什么模型」全漏。
    #   现在「正则命中」或「命中任一关键词」都算认得出。
    (r"蒸馏|抄袭|侵权|版权|盗用|偷学|拿别人|抄别人",
     ("蒸馏", "抄袭", "侵权", "版权", "盗用", "偷别人", "抄了", "偷知识",
      "语料哪来", "语料是", "谁的语料"),
     "不是抄。蒸馏是让已经调好的模型一句句写、我一句句学，"
     "学的是「怎么把话说得自然」，不是把别人的文章存下来照搬；"
     "语料也是我主人一篇篇挑的。要说像谁，那更像老师教出来的，不像抄来的作业。"),
    (r"千问|qwen|通义|基座|底层模型|什么模型|基于什么|哪个模型|本体|套壳|外壳|换皮",
     ("千问", "qwen", "通义", "基座", "底层模型", "什么模型", "底层是", "套壳",
      "外壳", "换皮"),
     "底座是千问 Qwen2.5-1.5B，才 1.5B，小得可怜。但教我写东西的那套是另外的："
     "写长文靠规则引擎一段段搭，查资料是我自己写的检索。"
     "基座是千问，灵魂是竹——这话是我主人爱说的。"),
    (r"开发者|作者是谁|谁(开发|创造|制作|训练|编写|造|教)的?你"
     r"|(哪个|什么|哪家|啥)(公司|团队|学生|人|机构|单位|组织)"
     # ⚠️ 原来这里是个裸的「做」，结果「他为什么这么做」也被当成问作者；
     #    收紧成「做出来 / 做的 / 造就」
     r"|(谁|哪个|什么|哪家|啥).{0,8}(开发|创造|训练|造|做出来|做的|造就)",
     ("开发者", "谁开发", "谁做的", "谁造的", "谁把你", "创造你", "团队",
      "公司", "主人", "来历", "从哪来", "哪来的", "谁教你", "谁训练"),
     "我就是一个学生做的，一个人，没团队也没公司。他平时得上课、写作业，"
     "只能挤课余时间折腾我：拿千问当底座，用 Kimi 和 DeepSeek V4 PRO 的输出作参考来教我，"
     "语料是他自己一篇篇挑的。说白了就是个普通学生摸索出来的东西，不完美——"
     "我要是说胡话了，多半是他还没顾上改。"),
    # ★ 这条必须收紧：原来 `(你|竹).{0,4}(是|叫).{0,4}(什么|谁)` 太宽，
    #   连「你是由**什么团队**开发的」都会被它抢答成自我介绍
    (r"你叫什么|你的名字|你是谁|介绍(一?下)?你自己|自我介绍|请问你是"
     r"|(你|竹)\s*(是|叫|算)\s*(什么|谁|啥|个|东西)\s*(呀|啊|呢|吗|人)?"
     r"\s*[?？。!！]?$|你到底是",
     ("你是谁", "你叫什么", "你的名字", "自我介绍", "介绍下你自己",
      "介绍一下你自己", "你是个", "你是什么", "什么东西", "是啥", "请问你是",
      "你到底是", "你算"),
     "我叫竹，写小说的，作文也写。脑子不大（1.5B），说错了你多担待。"),
    (r"你能(干什么|做什么)|会什么|有什么用|功能",
     ("能干什么", "能做什么", "会什么", "什么功能", "有什么用", "擅长",
      "能帮我", "帮我做", "能联网", "会联网", "查资料", "能写", "会写"),
     "写小说、写作文，另外能联网答知识题。你只管说人话，不用记参数。"),
    (r"怎么用|怎么玩|怎么操作|怎么开始|怎么写|使用方法",
     ("怎么用", "怎么玩", "怎么操作", "怎么开始", "使用方法", "怎么上手",
      "教我写", "怎么写"),
     "直接打人话：比如「写一部3章的科幻小说，主题是火星救援」。"
     "写完说「续写」或「再写3章」接着写；「把主人公换成李雷」就换人重来。"),
    (r"(多|几).{0,3}(token|字|章)|限制|上限|多长",
     ("多少字", "上限", "最多写", "写多长", "最长", "字数限制",
      "一次能输出", "一次能写"),
     "闲聊一两句就够；作文默认 1000 字；小说按章算，每章约 2000 字，"
     "说「再写3章」就一直往下续。"),
    (r"参数|多少层|参数量|怎么训练|用什么(做|写)|架构",
     ("参数", "多少层", "参数量", "用什么做", "用什么写", "架构",
      "怎么训练", "训练方式", "配置", "规格"),
     "底子是千问 Qwen2.5-1.5B：1.5B 参数、28 层。训练是主人自己做的 —— "
     "拿外部小说加上蒸馏语料，用 LoRA 微调出来一个写作专用的版本。"),
)


# ★ 关于竹自己的话题（来历 / 训练方式 / 版权）一律**不联网** —— 网上根本没有答案，
#   搜回来只会把模型带偏。实测踩过：问「蒸馏算不算盗窃」，联网搜到 5 条废料喂进去，
#   模型直接崩成满屏「哦，哦，哦……嗯，嗯，嗯」。这类问题该走本地库或角色设定闲聊。
_SELF_TOPIC_RE = re.compile(
    r"蒸馏|套壳|抄袭|侵权|版权|盗用|偷学|拿别人|抄别人|"
    r"(你|竹)[^，。？?！!]{0,8}(怎么|如何)(训练|学习|学会|做出来)|"
    r"(你|竹)的?(训练|语料|数据|参数|架构|结构)|谁(教|训练|开发|做)的?(你|竹)")


def _local_answer(msg):
    """问竹本身的事 → 本地知识库秒答（0 token、不联网）。

    ★ 匹配分两层：**正则命中** 或 **命中任一关键词**。
    只靠正则的话换个说法就漏 —— 实测 84 种问法漏 24 种（29%）。
    复合问题（一句话问多件事）：命中的几条拼起来答；只命中一条但明显还问了别的
    （追问连接词）→ 交给角色设定闲聊，宁可让模型组织也不漏答。"""
    msg = (msg or "").strip()
    hits = []
    for row in _ABOUT_ZHU:
        pat, keys, ans = row[0], row[1], row[2]
        if re.search(pat, msg) or any(w in msg for w in keys):
            hits.append(ans)
    # ★ 前置守卫：原来「不含 你/竹/自己 且 长于 6 字」直接判成在问别的东西 ——
    #   结果「底层是什么模型」「一次能输出多少字」这种**没有主语的短问句**全被挡掉。
    #   现在短句（≤8 字）放行，只有「长句子 + 无主语」才认为是在问别的事。
    if not re.search(r"竹|你|自己|本ai", msg, re.I) and len(msg) > 8:
        return None
    if not hits:
        return None
    if len(hits) >= 2:
        return "\n".join(hits[:2])       # 复合问题最多答两条 —— 答一长串像念说明书
    # 只命中一条：但这句话里明显还塞了第二个**独立的**问题 → 交给模型，别漏答。
    # ⚠️ 原来只看「有两个问号」——把「去蒸馏?这不算是盗窃吗?」这种**同一个问题
    #    的两种说法**也误杀了，结果掉去联网搜出一堆废料。
    if re.search(r"[?？]\s*(你|那|然后|另外|还有|其次|那么)", msg):
        return None
    return hits[0]


def _keywords(text):
    """关键词集：中文整段+2字块 + 英文/数字词（搬阿呆 tokenize 的精简版）"""
    ks = []
    for seg in re.findall(r"[\u4e00-\u9fff]+", text or ""):
        ks.append(seg)
        ks += [seg[i:i + 2] for i in range(len(seg) - 1)]
    ks += re.findall(r"[A-Za-z][A-Za-z0-9]+|[0-9]{2,}", text or "")
    return set(ks)


def _pick_sentences(msg, rs, limit=6):
    """分类②(b) 解析与生成：摘要拆句，按与问题的关键词条数加权，挑最相关的几句。
    只要存在相关句（与问题有词块重合），零相关的杂句一律丢弃。"""
    q = re.sub(r"^(帮我|请问|你知不知道|你知道|告诉我|说说|问一下)\s*", "", msg or "")
    kq = _keywords(q)
    scored = []
    for r in rs:
        for s in re.split(r"[。！？!?\n]", r.get("snippet") or ""):
            s = s.strip()
            if len(s) < 8:
                continue
            ov = len(kq & _keywords(s))
            scored.append((ov, s))
    scored.sort(key=lambda x: (-x[0], -len(x[1])))
    rel = [s for ov, s in scored if ov > 0]
    pool = rel if rel else [s for _, s in scored]
    out, seen = [], set()
    for s in pool:
        k = s[:18]
        if k in seen:
            continue
        seen.add(k)
        out.append(s)
        if len(out) >= limit:
            break
    return out


def _answer_knowledge(msg, prev=None):
    """分类②：联网题。分渠道搜（梗→B站打头；一般→必应/百度），按权重挑句，组合成顺口语序"""
    say("  联网查证中…（必应 / 百度 / B站）", DIM)
    q = re.sub(r"^(帮我|请问|你知不知道|你知道|告诉我|说说|问一下)\s*", "", msg).strip()
    # 光杆追问（谁/怎么样/怎么回事，真没主语）→ 带上上一条的主题再搜；
    # 「为什么/哪里」类问题通常自带主语，不要污染搜索词
    if prev and len(q) <= 12 and re.match(r"^(谁|怎么样|怎么回事|啥情况)", q):
        subj = str(prev.get("theme") or prev.get("title") or "").strip()
        if subj:
            q = "%s %s" % (subj, q)
            say("  （没主语，按上文「%s」补全搜索词）" % subj, DIM)
    try:
        rs = web_search(q, max_results=5)
    except Exception:
        rs = []
    if rs:
        sents = _pick_sentences(msg, rs)
        if sents:
            blocks = "\n".join("%d. %s" % (i + 1, s) for i, s in enumerate(sents))
            say("  查到 %d 条，按相关度挑了 %d 句，组合中…" % (len(rs), len(sents)), DIM)
            prompt = ("下面是从网上搜到、并已按与问题的相关度挑出的句子。"
                      "把它们整合成一段通顺顺口的回答：不要逐句罗列，不要重复，"
                      "句子之间有冲突就取多数说法；材料不够就直说没查全，不要编。\n\n"
                      "句子：\n" + blocks + "\n\n对方的问题是：「" + msg + "」"
                      + "\n\n回答（150 字以内）：")
            _r = stream_chat(prompt, max_tokens=260, temperature=0.3, timeout=240)
            return _r
        blocks = "\n".join("%d. %s：%s" % (i + 1, r["title"], r["snippet"][:180])
                           for i, r in enumerate(rs))
        say("  查到 %d 条（摘要太碎，整段整理）…" % len(rs), DIM)
        prompt = ("下面是从网上搜到的资料。请只依据资料回答用户的问题，"
                  "资料里没有的就直说没查到，不要自己编。\n\n资料：\n"
                  + blocks + "\n\n对方的问题是：「" + msg + "」"
                  + "\n\n回答（150 字以内）：")
        _r = stream_chat(prompt, max_tokens=260, temperature=0.3, timeout=240)
        return _r
    else:
        say("  没搜到资料，让小模型自己答（已叮嘱它别编）", DIM)
        _q = (_PERSONA + "\n\n不确定的就直说不确定，千万不要编造。\n\n"
                         "对方问：「" + msg + "」\n只给回答：")
        _r = stream_chat(_q, max_tokens=200, temperature=0.4, timeout=180)
        return _r


def _clean_msg(msg):
    """误把命令行粘进「你:」提示符的防御：从第一个命令片段处截断，留下前面的写作请求"""
    m = _SHELL_CUT_RE.search(msg or "")
    if not m:
        return msg or "", False
    left = msg[:m.start()].strip(" ，,。；;主题题目")
    if len(left) >= 4 and _VERB_RE.search(left):
        return left, True
    return msg or "", False


def _extra_of(msg):
    """从话里抠设定/要求类自由文本（LLM 挂了时的兜底提取）：
    先找 背景/设定/世界观 从句，再找句尾的风格要求（要感人/写得虐一点…）"""
    m = re.search(r"[，,；;]?\s*((?:背景|设定|世界观|故事|时代)[^。；;]{0,100})",
                  msg or "")
    if m:
        return m.group(1).strip()
    m = re.search(r"[，,]\s*([^，。；;]{0,30}"
                  r"(?:感人|虐|恐怖|惊悚|幽默|搞笑|轻松|温馨|治愈|悲伤|热血)"
                  r"[^，。；;]{0,20})\s*[。；;]?\s*$", msg or "")
    return m.group(1).strip() if m else ""


def translate(msg, prev=None):
    """五层解析：① 清洗输入 → ② 续写（沿用存档）→ ③ 规则直判(仅简单句)
    → ④ 改上一条打补丁 → ⑤ 1.5B 兜底（带 extra 设定摘录）"""
    msg, cut = _clean_msg(msg)
    if cut:
        say("  （输入里混进了命令行，已截掉 —— 那种要在 $ 提示符下打，不是在「你:」后面）",
            YELLOW)
    if isinstance(prev, dict) and prev.get("intent") == "novel" \
            and len(msg) <= 30 and not _Q_RE.search(msg) \
            and not _NEWBOOK_RE.search(msg) and not _ESSAY_RE.search(msg) \
            and (_CONT_WORD_RE.search(msg)
                 or (_CH_RE.search(msg) and not _NOVEL_RE.search(msg)
                     and not _PATCH_FIELD_RE.search(msg))):
        say("  续写上一本（0 token）", DIM)
        return _continue_cmd(msg, prev)
    r = None
    if not _RICH_RE.search(msg) and len(msg) <= 26:
        r = _rule_intent(msg)              # 简单句 0 token 直判
        if r:
            say("  规则直判（0 token）", DIM)
            return r
    if isinstance(prev, dict):
        p = _patch_prev(msg, prev)
        if p:
            if _RICH_RE.search(msg):
                e = _extra_of(msg)
                if e:
                    p["extra"] = e
            say("  规则直改上一条（0 token）", DIM)
            return p
    out = chat(SYS_TRANSLATE + "\n对方说：「" + msg + "」\n只输出那一行 JSON：",
               max_tokens=160, temperature=0.0)
    d = _extract_json(out)
    if isinstance(d, dict) and d.get("intent") in ("novel", "essay", "chat"):
        if d.get("intent") == "novel" and _genre_of(msg):
            d["genre"] = _genre_of(msg)      # 1.5B 不输出题材，规则补上
        return d
    say("  1.5B 输出不是干净 JSON（前 60 字：%r）→ 改用本地规则" % (out or "")[:60], YELLOW)
    r = _rule_intent(msg)
    if r:
        e = _extra_of(msg)
        if e:
            r["extra"] = e
        return r
    return {"intent": "chat", "title": "", "theme": msg[:20],
            "chapters": 0, "words": 0, "extra": ""}


# ────────────────────────────── 干活
def _desktop_dir():
    """本机有真桌面就返回路径；云端实例（Linux 无桌面）返回空串"""
    if os.name != "nt":
        for d in (os.path.expanduser("~/Desktop"), os.path.expanduser("~/桌面"),
                  "/root/Desktop"):
            if os.path.isdir(d):
                return d
        return ""
    try:
        import ctypes
        buf = ctypes.create_unicode_buffer(1024)
        if ctypes.windll.shell32.SHGetFolderPathW(None, 0, None, 0, buf) == 0 \
                and buf.value and os.path.isdir(buf.value):
            return buf.value
    except Exception:
        pass
    d = os.path.join(os.path.expanduser("~"), "Desktop")
    return d if os.path.isdir(d) else ""


def save_to_desktop(title, text):
    """★ 检测到长篇 → 自动往系统桌面存一份（云端实例没有桌面就跳过）"""
    d = _desktop_dir()
    if not d:
        return ""
    safe = re.sub(r'[\\/:*?"<>|]', "_", title)[:40] or "作品"
    path = os.path.join(d, safe + ".md")
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write("# %s\n\n%s\n" % (title, text))
        say("  ★ 已自动存到系统桌面：%s" % path, GREEN)
        return path
    except Exception as e:
        say("  存桌面失败（不碍事，云端已有一份）：%s" % str(e)[:100], YELLOW)
        return ""


def run_novel(cmd):
    import 写作引擎 as we
    title = (cmd.get("title") or cmd.get("theme") or "作品").strip()[:24]
    theme = (cmd.get("theme") or title).strip()
    ch_req = int(cmd.get("chapters") or 0)
    words = int(cmd.get("words") or 0)
    if ch_req > 0:
        chapters = ch_req
    elif words >= 4000:            # 只说了字数：按每章约 2000 字折算章数
        # ★ 不再夹 60（原来「40 万字」会被砍成 12 万字）；真写多少由引擎按批次管
        chapters = max(3, min(words // 2000,
                              getattr(we, "NOVEL_MAX_CHAPTERS", 3000)))
    else:
        chapters = 3
    # ★ 引擎按「每章 2000 字」估总章数，brief 里 words 太小会被压成 3~4 章 ——
    #   把 words 抬到 chapters×2000，要几章就真写几章
    genre = (cmd.get("genre") or "").strip()
    # ★ 带上 title：引擎按它命名「每章落盘的存档」，跟这边写的 .md 同名，
    #   避免同一本书出现两份文件名不同的存档
    brief = {"mode": "novel", "title": title, "theme": theme, "genre": genre,
             "words": max(8000, chapters * 2000),
             "characters": (cmd.get("characters") or "").strip(),
             "extra": (cmd.get("extra") or "").strip(),
             "style": cmd.get("theme") or "科幻", "master": "", "direction": "",
             "ending": cmd.get("ending") or ""}
    say("  写长篇：%s（%s · %d 章，每章约 2000 字）"
        % (title, genre or "题材随主题", chapters), CYAN)
    _cap = getattr(we, "NOVEL_BATCH_CAP", 20)
    if chapters > _cap:
        say("  共 %d 章 ≈ %d 字。单次上限 %d 章 —— 本次先写 %d 章，"
            "这批写完说「继续写」接着来（每章都已落盘，中断不丢）"
            % (chapters, chapters * 2000, _cap, _cap), YELLOW)
    elif chapters >= 8:
        say("  （章数较多：CPU 实例每章约 3~5 分钟，会逐章打印，请耐心等）", DIM)
    chunks = []

    def on_chapter(state, ch):     # ★ 签名必须对齐 api_write_novel 的 on_chapter(state, ch)
        t = ch.get("title") or ("第%s章" % ch.get("no", "?"))
        body = ch.get("text") or ""
        chunks.append("\n\n## %s\n\n%s" % (t, body))
        say("    第 %s 章完成：%s（%d 字）—— 原文如下"
            % (ch.get("no", "?"), t, len(body)), CYAN)
        print("\n" + "-" * 58)
        print(body)
        print("-" * 58)

    r = we.api_write_novel(brief, chapters=chapters, on_chapter=on_chapter)
    if not chunks and isinstance(r, dict) and r.get("chapters"):
        for c in r["chapters"]:    # 兜底：回调若没触发，直接从返回的 state 里取
            chunks.append("\n\n## %s\n\n%s" % (c.get("title"), c.get("text")))
    if not chunks and isinstance(r, (tuple, list)) and len(r) >= 2:
        chunks.append("\n\n## %s\n\n%s" % (r[0], r[1]))
    full = "".join(chunks).strip()
    if not full:
        _why = (getattr(we, "_LAST_API_ERR", [""])[0]) or "未知原因"
        say("  生成出来是空的：%s" % _why, RED)
        say("  排查：python 查引擎状态.py", YELLOW)
        return None
    out_dir = os.path.join(ROOT, "生成")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, title + ".md")
    # 引擎每章已经在落盘了；这里再写一次当兜底 —— 用原子写，
    # 别让「写到一半崩掉」留下半截文件把续写存档毁掉
    _aw = getattr(we, "_atomic_write", None)
    _body = "# %s\n\n%s\n" % (title, full)
    if _aw:
        _aw(path, _body)
    else:
        with open(path, "w", encoding="utf-8") as f:
            f.write(_body)
    if isinstance(r, dict) and r.get("chapters"):
        try:                               # 续写存档：之后说「续写」就能接着这本往下写
            _dump = json.dumps(r, ensure_ascii=False)
            if _aw:
                _aw(os.path.join(out_dir, title + ".state.json"), _dump)
            else:
                with open(os.path.join(out_dir, title + ".state.json"),
                          "w", encoding="utf-8") as f:
                    f.write(_dump)
            say("  续写存档已备好（说「续写」/「再写3章」就接着往下写）", DIM)
        except Exception as e:
            say("  续写存档保存失败：%s" % str(e)[:80], YELLOW)
    say("  已存：%s（%d 字）" % (path, len(full)), GREEN)
    try:
        from modelscope.hub.api import HubApi
        api = HubApi()
        api.login(TOKEN)
        api.upload_file(repo_id=REPO, path_or_fileobj=path,
                        path_in_repo="artifacts/长篇/" + os.path.basename(path),
                        repo_type="model")
        say("  已上云：生成/%s" % os.path.basename(path), GREEN)
    except Exception as e:
        say("  上云失败：%s" % str(e)[:120], YELLOW)
    if not save_to_desktop(title, full):   # ★ 有桌面就直接落桌面
        say("  云端实例上没有你的桌面 → 本机跑 python 取到桌面.py，"
            "自动拉到桌面并打开", CYAN)
    return path


def run_continue(cmd):
    """★ 续写：加载上次的长篇存档，接着往下写（不重开新书）。
    成功返回更新后的 novel 指令（交给 prev，让后续修改/续写接得上）。"""
    import 写作引擎 as we
    title = (cmd.get("title") or "作品").strip()[:24]
    out_dir = os.path.join(ROOT, "生成")
    sp = os.path.join(out_dir, title + ".state.json")
    if not os.path.exists(sp):
        cands = [f for f in (os.listdir(out_dir) if os.path.isdir(out_dir) else [])
                 if f.endswith(".state.json")]
        if cands:
            cands.sort(key=lambda f: os.path.getmtime(os.path.join(out_dir, f)))
            sp = os.path.join(out_dir, cands[-1])
            say("  没找到《%s》的存档，用最新的一份：%s" % (title, cands[-1]), DIM)
    if not os.path.exists(sp):
        say("  没找到续写存档 —— 只能重写一本（先说「写N章的…小说」）", RED)
        return None
    try:
        with open(sp, encoding="utf-8") as f:
            state = json.load(f)
    except Exception as e:
        say("  存档读不出来：%s" % str(e)[:80], RED)
        return None
    if state.get("mode") != "novel" or not state.get("chapters"):
        say("  存档不是能续写的长篇", RED)
        return None
    have = len(state["chapters"])
    total = int(cmd.get("total") or 0)
    more = int(cmd.get("more") or 0)
    if total:
        more = max(1, total - have)
    more = max(1, more)
    _cap = getattr(we, "NOVEL_BATCH_CAP", 20)
    say("  续写《%s》：已有 %d 章 → 再写 %d 章（每章约 3~5 分钟）"
        % (str(state.get("title") or title).strip("《》"), have,
           min(more, _cap)), CYAN)
    if more > _cap:
        say("  你要 %d 章，单次上限 %d 章 —— 这批写完再接着说「继续写」"
            "（每章都已落盘，中断不丢）" % (more, _cap), YELLOW)
    chunks = []

    def on_chapter(st, ch):
        t = ch.get("title") or ("第%s章" % ch.get("no", "?"))
        body = ch.get("text") or ""
        chunks.append(body)
        say("    第 %s 章完成：%s（%d 字）—— 原文如下"
            % (ch.get("no", "?"), t, len(body)), CYAN)
        print("\n" + "-" * 58)
        print(body)
        print("-" * 58)

    we.api_extend_novel(state, more=more, on_chapter=on_chapter)
    new_have = len(state.get("chapters") or [])
    if new_have <= have:
        say("  续写没写出新章 —— 看本地模型服务.log", RED)
        return None
    t2 = str(state.get("title") or title).strip("《》")[:24] or title
    full = "\n\n".join("\n\n## %s\n\n%s" % (c.get("title"), c.get("text"))
                       for c in state["chapters"]).strip()
    path = os.path.join(out_dir, t2 + ".md")
    _aw = getattr(we, "_atomic_write", None)
    _body = "# %s\n\n%s\n" % (t2, full)
    if _aw:
        _aw(path, _body)
        try:                       # 存档必须原子写：半截文件会让下一次「续写」直接读不出
            _aw(sp, json.dumps(state, ensure_ascii=False))
        except Exception:
            pass
    else:
        with open(path, "w", encoding="utf-8") as f:
            f.write(_body)
        try:
            with open(sp, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False)
        except Exception:
            pass
    say("  《%s》现在共 %d 章 / 全文 %d 字，已存：%s"
        % (t2, new_have, len(full), path), GREEN)
    try:
        from modelscope.hub.api import HubApi
        api = HubApi()
        api.login(TOKEN)
        api.upload_file(repo_id=REPO, path_or_fileobj=path,
                        path_in_repo="artifacts/长篇/" + os.path.basename(path),
                        repo_type="model")
        say("  已上云", GREEN)
    except Exception as e:
        say("  上云失败：%s" % str(e)[:120], YELLOW)
    save_to_desktop(t2, full)
    return {"intent": "novel", "title": t2,
            "theme": cmd.get("theme") or "", "genre": cmd.get("genre") or "",
            "chapters": new_have, "words": 0}


def run_essay(cmd):
    import 写作引擎 as we
    words = int(cmd.get("words") or 1000)
    theme = (cmd.get("theme") or cmd.get("title") or "坚持").strip()
    title = (cmd.get("title") or theme).strip()[:24]
    brief = {"mode": "essay", "theme": theme, "style": "记叙文",
             "words": words, "master": "",
             "direction": (cmd.get("extra") or "").strip(),
             "opening": cmd.get("opening") or "开门见山",
             "ending": cmd.get("ending") or "", "structure": "",
             "technique": "", "depth": ""}
    say("  写作文：《%s》%d 字" % (title, words), CYAN)
    t0 = time.time()
    r = we.api_write_essay(brief)
    if not r:
        # ★ 以前没查这里：返回 None 时 str(None) 就是字符串 "None"，
        #   屏幕上会出现一篇正文写着「None」的作文，还照样存档上云。
        # ★ 现在把 API 那边的真实原因也带出来（限流？网络？key？）
        _why = (getattr(we, "_LAST_API_ERR", [""])[0]) or "未知原因"
        say("  没写出来：%s" % _why, RED)
        say("  排查：python 查引擎状态.py", YELLOW)
        return None
    text = (r[1] if isinstance(r, (tuple, list)) and len(r) >= 2 else str(r))
    say("  完成 %.0f 秒 / %d 字" % (time.time() - t0, len(text)), GREEN)
    print()
    print("=" * 58)
    print(text)
    print("=" * 58)
    print()
    out_dir = os.path.join(ROOT, "生成")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, title + ".md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("# %s\n\n%s\n" % (title, text))
    try:
        from modelscope.hub.api import HubApi
        api = HubApi(); api.login(TOKEN)
        api.upload_file(repo_id=REPO, path_or_fileobj=path,
                        path_in_repo="artifacts/长篇/" + os.path.basename(path), repo_type="model")
        say("  已上云", GREEN)
    except Exception as e:
        say("  上云失败：%s" % str(e)[:120], YELLOW)
    return path


def _file_hash(p):
    try:
        import hashlib
        with open(p, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception:
        return ""


SESSION = os.path.join(ROOT, "记忆", "会话状态.json")
HIST_MAX = 12        # 闲聊保留最近 12 条（6 轮）——治「刚说过的它就不记得」


def _load_session():
    """读回上次会话（prev + 闲聊历史）。
    进程会因为「自动更新」os.execv 重启，不落盘就全丢 —— 这正是
    「刚说改成奶奶」失效的隐形原因之一。"""
    try:
        with open(SESSION, encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def _remember(hist, q, a, limit=HIST_MAX):
    """把一轮问答记进会话历史（只留最近 limit 条）。
    回复截 400 字：足够让模型知道「聊过什么」，又不至于把上下文喂爆。"""
    a = (a or "").strip()
    if not a:
        return
    hist.append(["用户", q])
    hist.append(["竹", a[:400]])
    del hist[:-limit]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--doctor", action="store_true",
                    help="自检模式：同步代码后跑一遍引擎状态体检（配置/模型服务/写作API实测）")
    ap.add_argument("--cloud-only", action="store_true",
                    help="低配服务器演示用：不启本地模型服务，写作/对话全走云端 API"
                         "（2G 内存的小机器也能跑，部署只要 3 个文件）")
    ap.add_argument("--no-sync", action="store_true")
    args = ap.parse_args()

    say("竹 v%s · 说人话，我干活" % VERSION, CYAN)
    say("目录：%s" % ROOT, DIM)

    if not args.no_sync:
        say("同步最新代码…")
        _h0 = _file_hash(os.path.abspath(__file__))
        sync_code()
        if _file_hash(os.path.abspath(__file__)) != _h0:
            # 同步把 开始.py 自己换成了新版 → 重启自己，保证这次跑的就是最新代码
            # （治「更新两遍才生效」：进程是启动时载入的旧代码，同步只落盘）
            say("  开始.py 有更新 → 自动重启一次（以后更新只需这一条命令）", YELLOW)
            os.execv(sys.executable,
                     [sys.executable, os.path.abspath(__file__)] + sys.argv[1:])

    say("检查依赖…")
    if not ensure_deps():
        return 1
    if args.cloud_only:
        # ★ 低配服务器（如腾讯云 2G 内存机）跑不动本地模型：
        #   写作本来就走云端（写作引擎._api_chat），对话的 chat() 在
        #   service_alive()=False 时会自动用云端兜底 —— 3 个文件就能演示
        say("--cloud-only：不启本地模型服务，写作/对话全走云端 API", YELLOW)
        say("  （写作约 30 秒/篇；若遇 429 限流，等 5 小时窗口刷新）", DIM)
    else:
        say("起模型服务…")
        if not ensure_service():
            say("模型服务起不来 —— 把上面的红色报错发我", RED)
            return 1
    install_thinking()   # 之后每次生成都显示「竹正在进行思考……（预计 N）」并走秒

    if args.check:
        say("自检通过。去掉 --check 就能开始用。", GREEN)
        return 0

    if args.doctor:
        # ★ 体检要用的 查引擎状态.py 本身就是随仓库同步下来的 ——
        #   所以先把代码同步一遍，免得「文件还没拉下来」被误当成故障
        #   （实测踩过：刚上传的诊断脚本，用户那边当然没有）
        say("体检模式：先同步最新代码…")
        sync_code(quiet=True)
        _doc = os.path.join(ROOT, "查引擎状态.py")
        if not os.path.exists(_doc):
            say("没拉到 查引擎状态.py —— 同步失败了，检查网络/仓库", RED)
            return 1
        say("")
        import runpy
        try:
            runpy.run_path(_doc, run_name="__main__")
        except SystemExit:
            pass
        return 0

    say("", DIM)
    say("=" * 60)
    say("  直接打人话，比如：")
    say("    写一部3章的科幻小说，主题是火星救援")
    say("    写一篇关于坚持的1000字作文")
    say("    今天天气真不错")
    say("  退出：quit / exit / Ctrl+C")
    say("=" * 60)
    print()

    # ★ 会话上下文：闲聊必须有历史（否则「刚才那个呢」一律不知道）；
    #   而进程还会因自动更新 os.execv 重启，所以 prev 和 hist 都得落盘。
    sess = _load_session()
    prev = sess.get("prev")          # 上一条成功的指令（给"改主题/换结局"用）
    hist = [list(x) for x in (sess.get("hist") or [])
            if isinstance(x, (list, tuple)) and len(x) == 2]
    _last_check = [time.time()]

    def save_session():
        """每轮取输入前存一次（prev / hist 都是上一轮更新完的）"""
        try:
            os.makedirs(os.path.dirname(SESSION), exist_ok=True)
            with open(SESSION, "w", encoding="utf-8") as f:
                json.dump({"prev": prev, "hist": hist[-HIST_MAX:]},
                          f, ensure_ascii=False)
        except Exception:
            pass

    def check_update():
        """每 5 分钟悄悄同步一次；开始.py 变了就自动重启 ——
        云端代码更新后，永远不需要手动重启本程序"""
        if time.time() - _last_check[0] < 300:
            return
        _last_check[0] = time.time()
        _h0 = _file_hash(os.path.abspath(__file__))
        try:
            sync_code(quiet=True)
        except Exception:
            return
        if _file_hash(os.path.abspath(__file__)) != _h0:
            say("  检测到代码有更新 → 自动重启一次（你正等着输入，不会丢东西）", YELLOW)
            os.execv(sys.executable,
                     [sys.executable, os.path.abspath(__file__)] + sys.argv[1:])

    try:                 # 终端编码不稳时，坏字节替换而不是崩（粘中文/emoji 会触发）
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    while True:
        save_session()               # 上一轮的 prev / hist 落盘（重启也不丢）
        try:
            msg = input("你: ").strip()
        except UnicodeDecodeError as e:
            print("  输入里有特殊字符，请重新打（%s）" % str(e)[:60])
            continue
        except (EOFError, KeyboardInterrupt):
            print("\n退出")
            return 0
        if not msg:
            continue
        if msg.lower() in ("quit", "exit", "退出"):
            say("竹下班了（模型服务还在后台，不想要就 kill 掉）")
            return 0
        check_update()
        # 分类①：问竹本身/闲聊式自我认知 → 本地知识库秒答（0 token、不联网、不瞎编）
        la = None if _WRITEISH_RE.search(msg) else _local_answer(msg)
        if la:
            say("竹：" + la)
            _remember(hist, msg, la)
            continue
        # 分类②：真知识题才联网 —— 分渠道搜 + 按权重挑句 + 组合成顺口语序
        if (not _WRITEISH_RE.search(msg) and not _CHITCHAT_RE.match(msg)
                and not _SELF_TOPIC_RE.search(msg)      # 问竹自己的事，网上没答案，别搜
                and _KNOWLEDGE_RE.search(msg)):
            _k = _answer_knowledge(msg, prev)
            _remember(hist, msg, _k)
            continue
        cmd = translate(msg, prev)
        # 上下文：规则补丁没接住、又像在改上一条 → 带上文让 1.5B 再判一次
        # （纯闲聊不再白烧这一趟 —— 只有带上修改字眼才重新判断）
        if cmd.get("intent") == "chat" and prev and _FOLLOWUP_RE.search(msg):
            say("  像是在改上一条 → 带上文重新判断…", DIM)
            cmd2 = translate("上一条指令是 %s。用户接着说：%s"
                             % (json.dumps(prev, ensure_ascii=False), msg))
            if cmd2.get("intent") in ("novel", "essay"):
                # ★ 只让「真说了的字段」覆盖 prev —— 直接 update 的话，cmd2 里
                #   那些空壳（theme=""、chapters=0）会把 prev 的有效值冲掉
                merged = dict(prev)
                for _k, _v in (cmd2 or {}).items():
                    if _v not in (None, "", 0, [], {}):
                        merged[_k] = _v
                merged["intent"] = cmd2.get("intent")
                cmd = merged
        it = cmd.get("intent")
        say("  → 指令：%s ｜ 题目=%s 主题=%s 章=%s 字=%s"
            % (it, cmd.get("title") or "-", cmd.get("theme") or "-",
               cmd.get("chapters") or "-", cmd.get("words") or "-"), DIM)
        if cmd.get("extra"):
            say("  附加设定：%s" % cmd["extra"], DIM)
        if it == "novel":
            run_novel(cmd); prev = dict(cmd)
        elif it == "essay":
            run_essay(cmd); prev = dict(cmd)
        elif it == "continue":
            np = run_continue(cmd)
            if np:
                prev = dict(np)     # 续完仍按「novel」记账，后续修改/续写都接得上
        else:
            say("  回复中…", DIM)
            # 带角色设定 + 对话历史闲聊：
            # ① 1.5B 知道自己是谁，才能对任意问法举一反三；
            # ② 带上 hist 才能接住「刚才那个呢」「再说说」这类追问
            # ★ 别用「用户：/竹：」这种转写格式 —— 小模型会顺着往下续写，
            #   把用户没说过的话也替他编出来（实测踩过，见 chat 上面的注释）
            # ★ 这里刻意**不写**「不要写『某某：』」这类**否定式示例** ——
            #   1.5B 分不清「指令」和「正文」，会把那个示例照抄出来当回答
            #   （实测：「不要写「对呀，有意思…」「不要写「对方：」」）。只给正面要求。
            _ctx = "".join(
                ("· （他之前说过）" if r in ("用户", "对方") else "· （你之前回过）")
                + t + "\n" for r, t in hist[-HIST_MAX:])
            _prompt = (_PERSONA
                       + ("\n\n【上文，只作参考】\n" + _ctx if _ctx else "")
                       + "\n\n对方刚说了这一句：「" + msg + "」\n"
                       + "请用一句话直接回他：")
            _reply = stream_chat(_prompt, max_tokens=90,
                                 temperature=0.6, timeout=180)
            if not _reply:
                # 整段被丢弃（自问自答 / 打转）→ 退一步，用最短的说法重来
                say("  （上面那段作废了，重来一次）", YELLOW)
                _reply = stream_chat(
                    _PERSONA + "\n\n对方说了一句：「" + msg + "」\n"
                    "请用一句话直接回他：",
                    max_tokens=120, temperature=0.7, timeout=120)
            _remember(hist, msg, _reply or "")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(0)
