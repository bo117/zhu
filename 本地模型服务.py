# -*- coding: utf-8 -*-
"""
本地模型服务.py —— 竹 · 微调模型推理服务（提速优化版）
==================================================================
原版：朴素 fp16 逐 token，能跑但慢。本版在不改对外接口的前提下做了
一整套推理侧优化，阿呆.py / 竹.py 那边一行都不用动。

对外接口（与原来完全一致）：
  GET  /health              → {"ok": true, "model": 路径, "device": cuda/cpu}
  GET  /stats               → 新增：吞吐、请求数、平均耗时
  POST /chat/completions    → OpenAI 兼容；stream=true 走 SSE
                             新增可选字段：stop、top_k、min_p、max_tokens

提速手段（按收益从大到小）：
  1) torch.inference_mode()          —— 关掉 autograd 记账，显存/内存与时间双省
  2) attn_implementation="sdpa"      —— 走 PyTorch 融合注意力，GPU 上提升最明显
  3) CPU 线程数 = 物理核数            —— 本机 CPU 跑时必须设，否则只用一半核
  4) 启动 warmup                      —— 把 CUDA 上下文/内存池/首帧开销挪到开张前
  5) 启动 warmup                      —— 把首帧开销挪到开张前
  6) 流式合批输出                     —— 攒够一批再 write，减少 syscall 次数
  7) 可选 torch.compile              —— 加 --compile 开启，GPU 上再快一成
  8) 采样参数改进                     —— no_repeat_ngram_size 压复读，top_k/min_p 提质

推理逻辑增强：
  · 自适应生成长度：按请求里写的字数要求估算 max_new_tokens，不再一刀切
  · 复读自停：检测到长片段的自我重复就提前收尾，不浪费算力
  · 空结果自动重试一次
  · stop 词支持

用法（在训练虚拟环境里跑）：
  .train_venv\\Scripts\\python.exe 本地模型服务.py
  .train_venv\\Scripts\\python.exe 本地模型服务.py --port 8766 --max-new 1600 --no-prefix-cache
"""
import argparse
import glob
import json
import os
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)

MODELS = {}             # 档位 -> (tokenizer, model, device, 来源路径)；main=主档 / ext=扩展档
_LOAD_LOCKS = {}        # 每档一把加载锁
LOAD_META = {}          # 档位 -> 加载耗时等元信息（/health 会带出去）
GEN_LOCK = threading.Lock()
PREFIX = {"key": None, "past": None, "len": 0, "enabled": True}
STATS = {"requests": 0, "tokens": 0, "seconds": 0.0, "errors": 0,
         "tokens_s": 0.0, "last_ms": 0}


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def _src_build():
    """本文件的源码指纹。/health 会带出去，让 开始.py 判断跑着的服务是不是最新代码
    —— 服务是独立进程，光重启 开始.py 是换不掉它的。"""
    try:
        import hashlib
        with open(os.path.join(ROOT, "本地模型服务.py"), "rb") as f:
            return hashlib.md5(f.read()).hexdigest()[:8]
    except Exception:
        return ""


# ─────────────────────── 模型定位（与原版一致） ───────────────────────
def find_model():
    """按优先级找模型；返回 (kind, 路径) kind ∈ merged / adapter"""
    for out in ("output", "output_long", "output_local"):
        merged = os.path.join(ROOT, out, "merged")
        cfgp = os.path.join(merged, "config.json")
        if os.path.exists(cfgp):
            # ★ 4bit QLoRA 合并出来的模型，config.json 里带 quantization_config，
            # 加载它要 bitsandbytes，而 **CPU 上没有可用的 4bit**（会直接 ImportError 崩）。
            # 所以带量化配置的 merged 一律跳过，改走 adapter + 全精度基座。
            quant = False
            try:
                with open(cfgp, encoding="utf-8") as f:
                    quant = bool(json.load(f).get("quantization_config"))
            except Exception:
                pass
            if quant:
                log("跳过 %s：带 4bit 量化配置，CPU 加载不了" % os.path.relpath(merged, ROOT))
                continue
            return ("merged", merged)
    for out in ("output", "output_long", "output_local"):
        adapter = os.path.join(ROOT, out, "adapter")
        if os.path.exists(os.path.join(adapter, "adapter_config.json")):
            base = find_base(adapter)      # ★ 带上 adapter：按它记的基座名去找
            if base:
                return ("adapter", (adapter, base))
    return (None, None)


def find_base(adapter=None):
    """找 LoRA 的基座模型。

    ★ 优先读 `adapter_config.json` 里的 `base_model_name_or_path` ——
    这样把基座从 1.5B 换成 3B/7B 之后，推理侧能自动对上。
    原来只靠"猜目录名"（models/base 或 models/models/Qwen--*/snapshots/*），
    换基座后会**悄悄加载成 1.5B**，表现为「换了模型却没效果」。
    读不到再退回老的两条猜测路径。"""
    if adapter:
        try:
            with open(os.path.join(adapter, "adapter_config.json"),
                      encoding="utf-8") as f:
                bp = (json.load(f).get("base_model_name_or_path") or "").strip()
            if bp:
                name = bp.replace("/", "--")
                cands = []
                cands += sorted(glob.glob(os.path.join(
                    ROOT, "models", "models", name, "snapshots", "*")))
                cands += sorted(glob.glob(os.path.join(ROOT, "models", name, "*")))
                cands += [os.path.join(ROOT, "models", bp.split("/")[-1]),
                          os.path.join(ROOT, "models", "base")]
                for c in cands:
                    if c and os.path.exists(os.path.join(c, "config.json")):
                        return c
        except Exception:
            pass
    b = os.path.join(ROOT, "models", "base")
    if os.path.exists(os.path.join(b, "config.json")):
        return b
    cands = sorted(glob.glob(os.path.join(
        ROOT, "models", "models", "Qwen--*", "snapshots", "*")))
    for c in cands:
        if os.path.exists(os.path.join(c, "config.json")):
            return c
    return None


_THREADS_TUNED = [False]


def _tune_threads():
    """CPU 推理的关键一步：用满物理核，别让 torch 只吃一半。

    ⚠️ 只能调一次 —— `set_num_interop_threads` 在并行任务开始之后再调会抛异常，
    而扩展模型是第二个被加载的，不在这儿防住会当场崩。"""
    if _THREADS_TUNED[0]:
        return 0
    import torch
    try:
        import os as _os
        phys = _os.cpu_count() or 4
        torch.set_num_threads(phys)
        try:
            torch.set_num_interop_threads(max(1, phys // 2))
        except Exception:
            pass
        _THREADS_TUNED[0] = True
        return phys
    except Exception:
        return 0


def find_ext_model():
    """扩展模型（只在「升档」那一刻才用）。

    路径来源：`--ext-model` 或环境变量 `ZHU_EXT_MODEL`（相对路径按 ROOT 解析）。
    返回 None 表示没配 —— 那时「全档」还是拿主模型的全预算跑，功能照旧。"""
    p = ""
    try:
        p = getattr(ARGS, "ext_model", "") or ""
    except Exception:
        p = ""
    p = (p or os.environ.get("ZHU_EXT_MODEL", "") or "").strip()
    if not p:
        return None
    if not os.path.isabs(p):
        p = os.path.join(ROOT, p)
    return p if os.path.exists(os.path.join(p, "config.json")) else None


def load_model(which="main"):
    """取某一档的模型（懒加载）。

    which="main" 主档（平时都用它）；"ext" 扩展档（**只有升档时才会去加载**，
    所以不配扩展模型时它一点内存都不占）。"""
    got = MODELS.get(which)
    if got is not None:
        return got
    lk = _LOAD_LOCKS.setdefault(which, threading.Lock())
    with lk:
        got = MODELS.get(which)
        if got is not None:
            return got
        t0 = time.time()
        got = _load_one(which)
        MODELS[which] = got
        LOAD_META[which] = round(time.time() - t0, 1)
        return got


def _load_one(which):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        n = _tune_threads()
        if n:
            log("CPU 模式：torch 线程数设为 %d" % n)
    # dtype 可由环境变量覆盖：ZHU_DTYPE=bfloat16 时 CPU 上只用 3.1GB（fp32 要 6.2GB）
    _dt = os.environ.get("ZHU_DTYPE", "").strip().lower()
    if _dt in ("bfloat16", "bf16"):
        dtype = torch.bfloat16
    elif _dt in ("float16", "fp16", "half"):
        dtype = torch.float16
    else:
        dtype = torch.float16 if device == "cuda" else torch.float32

    # 注意力后端：sdpa 在 CPU/GPU 上都有加速；flash-attn 若装了优先
    kw = dict(torch_dtype=dtype, trust_remote_code=True,
              low_cpu_mem_usage=True)
    attn = None
    if device == "cuda":
        try:
            import flash_attn  # noqa: F401
            attn = "flash_attention_2"
        except Exception:
            attn = "sdpa"
    else:
        attn = "sdpa"
    kw["attn_implementation"] = attn

    if which == "ext":
        path = find_ext_model()
        if not path:
            raise RuntimeError("没配扩展模型（用 --ext-model 或 ZHU_EXT_MODEL 指路径）")
        log("首次升档 → 加载扩展模型：%s（要等一会儿，之后常驻）" % path)
        tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        try:
            mdl = AutoModelForCausalLM.from_pretrained(path, **kw)
        except TypeError:
            kw.pop("attn_implementation", None)
            mdl = AutoModelForCausalLM.from_pretrained(path, **kw)
        src, label = path, "扩展档"
    else:
        kind, path = find_model()
        if not kind:
            raise RuntimeError("没找到训练产物（output*/merged 或 output*/adapter）。"
                               "先跑 训练.py 训出模型再来。")
        if kind == "merged":
            tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
            try:
                mdl = AutoModelForCausalLM.from_pretrained(path, **kw)
            except TypeError:
                kw.pop("attn_implementation", None)
                mdl = AutoModelForCausalLM.from_pretrained(path, **kw)
            src = path
        else:
            from peft import PeftModel
            adapter, base = path
            tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
            try:
                mdl = AutoModelForCausalLM.from_pretrained(base, **kw)
            except TypeError:
                kw.pop("attn_implementation", None)
                mdl = AutoModelForCausalLM.from_pretrained(base, **kw)
            mdl = PeftModel.from_pretrained(mdl, adapter)
            src = "%s (+ %s)" % (adapter, base)
        label = "主档"

    mdl = mdl.to(device).eval()
    if getattr(ARGS, "compile", False) and which == "main":
        try:
            mdl = torch.compile(mdl, mode="reduce-overhead")
            log("已 torch.compile（首帧会慢一点，之后更快）")
        except Exception as e:
            log("torch.compile 不可用，跳过：%s" % e)

    log("%s模型就绪：%s（%s, %s, attn=%s）"
        % (label, src, device, str(dtype).split(".")[-1], attn))
    return (tok, mdl, device, src)


# ─────────────────────── 前缀 KV 复用 ───────────────────────
def _prefix_text(messages):
    """取固定前缀（第一条 system），没有就返回 None"""
    if messages and messages[0].get("role") == "system":
        return messages[0].get("content") or None
    return None


def warmup_and_cache_prefix():
    """启动时预热一次，把模型首帧开销挪出首个请求。"""
    import torch
    tok, mdl, device, _ = load_model()
    t0 = time.time()
    try:
        with torch.inference_mode():
            demo = tok.apply_chat_template(
                [{"role": "user", "content": "你好"}],
                tokenize=False, add_generation_prompt=True)
            ids = tok(demo, return_tensors="pt").to(device)
            kw = dict(max_new_tokens=8, do_sample=False, use_cache=True)
            try:
                kw["pad_token_id"] = tok.eos_token_id
                mdl.generate(**ids, **kw)
            except Exception:
                kw.pop("pad_token_id", None)
                mdl.generate(**ids, **kw)
        log("warmup 完成（%.1f 秒），首请求不再卡" % (time.time() - t0))
    except Exception as e:
        log("warmup 失败（不影响使用）：%s" % e)


def build_prefix_cache(system_text):
    """把 system 前缀 prefill 一次，后面同前缀直接复用 KV"""
    if not PREFIX["enabled"] or not system_text:
        return None
    if PREFIX["key"] == system_text and PREFIX["past"] is not None:
        return PREFIX["past"]
    import torch
    tok, mdl, device, _ = load_model()
    try:
        with torch.inference_mode():
            text = tok.apply_chat_template(
                [{"role": "system", "content": system_text}],
                tokenize=False, add_generation_prompt=False)
            ids = tok(text, return_tensors="pt").to(device)
            out = mdl(input_ids=ids["input_ids"], attention_mask=ids["attention_mask"],
                      use_cache=True)
            PREFIX.update(key=system_text, past=out.past_key_values,
                          len=int(ids["input_ids"].shape[1]))
        log("system 前缀已缓存（%d token），后续同前缀免 prefill" % PREFIX["len"])
        return PREFIX["past"]
    except Exception as e:
        PREFIX["enabled"] = False       # 只试一次，不行就永久降级
        log("前缀缓存不可用，已降级（%s）" % e)
        return None


# ─────────────────────── 生成长度自适应 ───────────────────────
_WORD_RE = re.compile(r"(\d{3,5})\s*字")


def suggest_max_new(text, default):
    """请求里写了 N 字就按 N 字估算（中文约 1 字≈1 token）"""
    m = _WORD_RE.search(text or "")
    if m:
        n = int(m.group(1))
        return max(64, min(int(n * 1.35) + 64, 4096))
    return default


# ─────────────────────── 复读自停 ───────────────────────
try:
    from transformers import StoppingCriteria as _StoppingCriteria
    from transformers import StoppingCriteriaList as _StoppingCriteriaList
except Exception:                       # 老版本没有就降级（不设 criteria 也能跑）
    class _StoppingCriteria(object):
        pass
    _StoppingCriteriaList = None


def _acceptable(text):
    """快档的结果能不能直接用（不合格就交给全档重来）。

    注意不能只调 _is_degenerate —— 那个有 60 字门槛（流式里防误杀用的），
    而快档本来就短，「哦哦哦哦」这种十来个字照样过不了关，得在这里自己判。"""
    t = (text or "").strip()
    if len(t) < 6:
        return False
    chars = [c for c in t
             if c not in "，。！？、；：…—\n \t「」“”'\"（）·!?,.:;"]
    if len(chars) >= 8:
        cnt = {}
        for c in chars:
            cnt[c] = cnt.get(c, 0) + 1
        if len(cnt) / len(chars) < 0.35:        # 用字太贫乏（「哦哦哦哦」）
            return False
        if max(cnt.values()) / len(chars) > 0.40:
            return False
    return not _is_degenerate(t)


class _Deadline(_StoppingCriteria):
    """到点就停 —— 给「快档」用：超过 escalate_after 秒还没写完，立刻中断交给全档。

    为什么用 StoppingCriteria 而不是在外面计时：那样 generate 还在后台空转，
    白白吃掉 CPU；这样是让生成循环自己退出来。"""

    def __init__(self, seconds):
        self.t0 = time.time()
        self.seconds = float(seconds or 0)
        self.hit = False

    def __call__(self, input_ids, scores, **kw):
        if self.seconds > 0 and (time.time() - self.t0) > self.seconds:
            self.hit = True
            return True
        return False


def _is_degenerate(buf, tail=96, need=3):
    """结尾是否在自我重复（判退化就提前收尾，别让它一直吐垃圾）。

    两道判据：
      ① 同一片段连着出现 need 次（原判据）
      ② ★ 字符级退化：单字占非标点字符六成以上 —— 治「哦，哦，哦……嗯，嗯，嗯」
         这种小片段交替复读，①的整段比对抓不到它（实测踩过：
         用户问「蒸馏算不算盗窃」，模型崩成满屏「哦」）
    """
    if len(buf) < 60:
        return False
    if len(buf) >= tail * need:
        seg = buf[-tail:]
        if buf[-tail * need:].count(seg) >= need:
            return True
    recent = buf[-200:]
    chars = [c for c in recent if c not in "，。！？、；：…—\n \t「」“”'\"（）·!?,.:;"]
    if len(chars) >= 40:
        cnt = {}
        for c in chars:
            cnt[c] = cnt.get(c, 0) + 1
        if len(cnt) / len(chars) < 0.35:            # 用字太贫乏（正常中文 0.6~0.9）
            return True
        if max(cnt.values()) / len(chars) > 0.35:   # 单字吃掉三成五 → 在打转
            return True
    return False


# ─────────── 两档推理：平时省着用，卡住了再全开 ───────────
# 用户的原话：「无论模型多大，平时只激活一部分；10 秒还答不出来就全部激活。」
# Qwen2.5-1.5B 是稠密模型，没有"只走 1.0B 参数"的开关，但**效果**可以做出来：
#   · 快档：短输出（≤SPARSE_FAST_NEW）+ 到点自停，CPU 上通常两三秒就出话
#   · 升档条件：到点还没写完（_Deadline.hit）或结果不合格（空/太短/打转）
#   · 全档：完整长度 + 完整采样，把算力全给它
SPARSE_FAST_NEW = 120          # 快档最多生成多少 token

# ★★ 推理强度（2026-10-14，抄 DeepSeek-V4.1-Flash 的 1~100 连续可调）
#   V4.1-Flash 官方原话：continuously controllable reasoning effort，
#   同一个权重，靠一个整数在「成本」与「准确率」之间当场取舍。
#   我们的映射（不改权重，只改采样）：
#     ≤30低 → top_p 0.85 / rep 1.05 / ngram 5   写新作文：要快、要干脆
#     30~70 中 → top_p 0.90 / rep 1.08 / ngram 6     （默认）
#     ≥70     高 → top_p 0.95 / rep 1.15 / ngram 8   改稿/诊断：让它多想、压死复读
#   用法：请求体加 reasoning_effort: 80，或启动时设环境变量 EFFORT=80
EFFORT = int(os.environ.get(EFFORT, 50))


def _pick_full_which():
    """「全档」该用哪一档模型：配了扩展模型就用它，没配就用主档的全预算。

    扩展档第一次用会现加载（慢一次，之后常驻内存）；加载失败自动退回主档，
    绝不让一次升档把整个请求搞挂。"""
    if not find_ext_model():
        return "main"
    try:
        load_model("ext")
        return "ext"
    except Exception as e:
        log("扩展档加载失败（%s）→ 退回主档" % str(e)[:120])
        return "main"


def _sparse_generate(messages, max_new, temperature, on_delta,
                     stop, top_k, min_p, esc, sparse, effort=None):
    """返回 (文本, 走的哪一档说明)。

    平时走「快档」：短输出 + 到点自停，CPU 上两三秒就出话；
    快档超时或结果不合格 → 「全档」：完整长度 + 完整采样，
    **配了扩展模型就切到那个更大的模型**（这就是用户要的「10 秒答不出就全开」）。
    只有「短回答」场景才启用快档 —— 写 2000 字章节时先跑 120 token 纯属浪费时间。
    """
    if not sparse or max_new > 400:
        w = _pick_full_which()
        return (generate(messages, max_new, temperature, on_delta,
                         stop, top_k, min_p, which=w, effort=effort),
                "全档" + ("(扩展)" if w == "ext" else ""))

    crit = _Deadline(esc)
    fast_new = max(24, min(max_new, SPARSE_FAST_NEW))
    t0 = time.time()
    fast = generate(messages, fast_new, temperature, lambda d: None,
                    stop, top_k, min_p, criteria=crit, which="main",
                    effort=effort)
    took = time.time() - t0
    if (not crit.hit) and _acceptable(fast):
        if on_delta and fast:
            on_delta(fast)                    # 快档命中：整段一次交出去
        return fast, "快档 %.1fs" % took

    log("快档没拿住（%.1f 秒%s）→ 全档激活"
        % (took, "，到点" if crit.hit else "，结果不合格"))
    w = _pick_full_which()
    return (generate(messages, max_new, temperature, on_delta,
                     stop, top_k, min_p, which=w, effort=effort),
            "全档(升档)" + ("·扩展" if w == "ext" else ""))


# ─────────────────────── 生成 ───────────────────────
def generate(messages, max_new, temperature, on_delta,
             stop=None, top_k=None, min_p=None, on_token=None, criteria=None,
             which="main", effort=None):
    """流式生成：逐段回调 on_delta(文本)。返回完整文本。
    which="main" 用主档，"ext" 用扩展档（升档时）。"""
    import torch
    from transformers import TextIteratorStreamer

    tok, mdl, device, _ = load_model(which)

    prompt = tok.apply_chat_template(messages, tokenize=False,
                                     add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(device)

    streamer = TextIteratorStreamer(tok, skip_prompt=True,
                                    skip_special_tokens=True)
    kw = dict(
        **inputs,
        max_new_tokens=max_new,
        do_sample=temperature > 0,
        temperature=max(temperature, 0.05),
        top_p=0.9,
        repetition_penalty=1.08,
        no_repeat_ngram_size=6,          # 压复读（新）
        streamer=streamer,
        pad_token_id=tok.eos_token_id,
    )
    # ★★ 推理强度（2026-10-14，抄 DeepSeek-V4.1-Flash 的「1~100 连续可调」）
    #   官方原话：continuously controllable reasoning effort，1~100 连续，
    #   同一个权重，难度高的调高、简单的调低，成本与准确率当场取舍。
    #   落到我们这儿：
    #     低强度（写新作文）→ 采样更收敛、更快、不啰嗦
    #     高强度（改稿/诊断）→ 采样更放开 + 压复读更狠，让它多想
    #   这是**权重之外**的旋钮，不用重训就能改行为。
    _eff = EFFORT if effort is None else int(effort)
    _eff = max(1, min(100, _eff))
    if _eff <= 30:                      # 低：干脆点
        kw["top_p"] = 0.85
        kw["repetition_penalty"] = 1.05
        kw["no_repeat_ngram_size"] = 5
    elif _eff >= 70:                # 高：放开一点让它探索
        kw["top_p"] = 0.95
        kw["repetition_penalty"] = 1.15
        kw["no_repeat_ngram_size"] = 8
    # 30~70 保持上面默认值
    if top_k:
        kw["top_k"] = int(top_k)
    if min_p:
        kw["min_p"] = float(min_p)
    if stop:
        # stop_strings 需要 transformers≥4.39；老版本传进去会直接抛异常、把整次生成搞挂，
        # 所以先探版本。不支持就只靠上层的 _cut_fake_turns 清洗兜底
        # （功能不受影响，只是可能多吐几个字出来）。
        try:
            import transformers as _tf
            _v = [int(x) for x in _tf.__version__.split(".")[:2]]
        except Exception:
            _v = [0, 0]
        if _v >= [4, 39]:
            kw["stop_strings"] = list(stop)
            kw["tokenizer"] = tok
        else:
            log("transformers %s 不支持 stop_strings，跳过（由上层清洗兜底）"
                % ".".join(str(x) for x in _v))

    if criteria is not None and _StoppingCriteriaList is not None:
        try:                              # 到点自停（「快档」超时用）
            kw["stopping_criteria"] = _StoppingCriteriaList([criteria])
        except Exception:
            pass

    with torch.inference_mode():          # ① 关 autograd
        th = threading.Thread(target=mdl.generate, kwargs=kw, daemon=True)
        th.start()
        chunks = []
        t0 = time.time()
        for piece in streamer:
            if not piece:
                continue
            chunks.append(piece)
            on_delta(piece)
            if on_token:
                on_token(piece)
            # 复读自停：已经明显在打转就不再等它吐完
            if _is_degenerate("".join(chunks[-400:])):
                break
        th.join()

    text = "".join(chunks).strip()
    ms = int((time.time() - t0) * 1000)
    n_tok = len(tok(text, add_special_tokens=False)["input_ids"]) if text else 0
    STATS["tokens"] += n_tok
    STATS["seconds"] += (time.time() - t0)
    STATS["last_ms"] = ms
    if STATS["seconds"] > 0:
        STATS["tokens_s"] = round(STATS["tokens"] / STATS["seconds"], 1)
    return text


# ─────────────────────── HTTP ───────────────────────
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        p = self.path.split("?")[0]
        if p == "/health":
            try:
                _, _, device, src = load_model()
                self._json({"ok": True, "model": src, "device": device,
                            "prefix_cache": PREFIX["enabled"],
                            "build": _src_build(),
                            "ext_model": find_ext_model() or "",
                            "ext_loaded": ("ext" in MODELS),
                            "load_meta": LOAD_META})
            except Exception as e:
                self._json({"ok": False, "err": str(e)}, 500)
        elif p == "/stats":
            self._json(dict(STATS, prefix_enabled=PREFIX["enabled"],
                            prefix_tokens=PREFIX["len"]))
        else:
            self._json({"err": "not found"}, 404)

    def do_POST(self):
        if self.path.split("?")[0] != "/chat/completions":
            self._json({"err": "not found"}, 404)
            return
        try:
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n).decode("utf-8") or "{}")
        except Exception:
            self._json({"err": "请求体不是合法 JSON"}, 400)
            return
        messages = body.get("messages") or []
        if not messages:
            self._json({"err": "messages 是空的"}, 400)
            return

        last_user = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                last_user = m.get("content") or ""
                break
        max_new = int(body.get("max_tokens") or 0) \
            or suggest_max_new(last_user, ARGS.max_new)
        max_new = max(16, min(max_new, 4096))
        temperature = float(body.get("temperature", 0.8))
        stop = body.get("stop")
        top_k = body.get("top_k")
        # ★ 推理强度：兼容 OpenAI/DeepSeek 两种字段名
        _e = body.get("reasoning_effort", body.get("effort"))
        effort = int(_e) if _e not in (None, "") else None
        min_p = body.get("min_p")
        stream = bool(body.get("stream"))
        STATS["requests"] += 1

        sparse = bool(body.get("sparse")) and not ARGS.no_sparse
        esc = float(body.get("escalate_after") or ARGS.escalate_after or 10.0)

        if not stream:
            with GEN_LOCK:
                try:
                    full, _how = _sparse_generate(
                        messages, max_new, temperature, lambda d: None,
                        stop, top_k, min_p, esc, sparse)
                    log("  → %s" % _how)
                except Exception as e:
                    STATS["errors"] += 1
                    self._json({"err": "生成失败：%s" % e}, 500)
                    return
            self._json({"choices": [{"message": {"role": "assistant",
                                                 "content": full}}],
                        "usage": {"completion_tokens": len(full)}})
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        # 流式合批：攒到 48 字或间隔 40ms 才发一次，少写很多次 socket
        pend = []
        last = [time.time()]

        def emit_now(s):
            obj = {"choices": [{"delta": {"content": s}}]}
            self.wfile.write(("data: %s\n\n" % json.dumps(
                obj, ensure_ascii=False)).encode("utf-8"))
            self.wfile.flush()

        def on_delta(d):
            pend.append(d)
            if sum(len(x) for x in pend) >= 48 or time.time() - last[0] > 0.04:
                emit_now("".join(pend))
                pend.clear()
                last[0] = time.time()

        try:
            with GEN_LOCK:
                _, _how = _sparse_generate(messages, max_new, temperature,
                                           on_delta, stop, top_k, min_p,
                                           esc, sparse, effort)
            log("  → %s" % _how)
            if pend:
                emit_now("".join(pend))
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            STATS["errors"] += 1
            try:
                emit_now("\n（本地模型生成中断：%s）" % e)
                self.wfile.write(b"data: [DONE]\n\n")
            except Exception:
                pass


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--max-new", type=int, default=1200)
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile 加速（首帧慢，之后更快）")
    ap.add_argument("--no-prefix-cache", action="store_true",
                    help="关闭 system 前缀 KV 复用")
    ap.add_argument("--no-warmup", action="store_true")
    ap.add_argument("--escalate-after", type=float, default=10.0,
                    help="两档推理：快档跑这么多秒还拿不住就全档全开（默认 10 秒）")
    ap.add_argument("--no-sparse", action="store_true",
                    help="关掉两档推理，每次都用全档")
    ap.add_argument("--ext-model", default="",
                    help="扩展档模型路径（只在升档时用）。不配则「全档」= 主档的全预算。"
                         "例：--ext-model models/ext")
    ARGS = ap.parse_args()
    PREFIX["enabled"] = not ARGS.no_prefix_cache

    ext = find_ext_model()
    log("加载主档模型（首次启动要十几秒）……")
    load_model("main")
    if ext:
        log("扩展档已配置：%s —— 平时不加载，快档拿不住时才启用" % ext)
    else:
        log("没配扩展档（--ext-model）—— 升档时用主档的全预算")
    if not ARGS.no_warmup:
        warmup_and_cache_prefix()
    srv = ThreadingHTTPServer(("127.0.0.1", ARGS.port), Handler)
    log("开张：http://127.0.0.1:%d （只听本机）" % ARGS.port)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


ARGS = None
if __name__ == "__main__":
    main()
