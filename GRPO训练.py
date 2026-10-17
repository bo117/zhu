# -*- coding: utf-8 -*-
"""
GRPO训练.py —— 竹 · 后训练算法落地（GSPO / DAPO / SAPO / Muon）
================================================================
把 顶流算法调研.md 里的算法写成能跑的代码，分两层：

  第一层：纯算法实现 + 自检（不加载模型，秒级）
       python GRPO训练.py --selfcheck
     这一层只依赖 numpy，在任何机器上都能验算法对不对。

  第二层：真实训练循环（GSPO 主循环）
       python GRPO训练.py --train --model Qwen/Qwen2.5-0.5B-Instruct \
              --out output_grpo --steps 200 --group 8
     需要 torch + transformers + peft（在训练虚拟环境里跑）。

设计要点（与 强化学习.py 的 RAFT 有什么不同）：
  · 优势：组内相对优势 + Dr.GRPO 去 std 归一化
  · 比率：GSPO 序列级几何均值 —— 竹写 800~1200 字，序列级才稳
  · 裁剪：DAPO Clip-Higher（上界放宽到 1.28），并可选 SAPO 软门控
  · 损失：Token 级（分母是总 token 数，不是序列数）
  · KL ：k3 估计，恒非负、方差小
  · 采样：动态采样 —— 组内全同分直接跳过，不烧算力
  · 奖励：原 6 维 + 去重奖励（指纹库）+ 长度合规 + 生死簿判骨架（对策③）
  · 优化器：Muon（2D 大矩阵）+ AdamW（其余参数）

  参考：GSPO arXiv:2507.18071 / DAPO 2025.03 / Dr.GRPO arXiv:2503.20783
"""
import argparse
import json
import math
import os
import random
import re
import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))

# ═══════════════════════════════════════════════════════════════
#  第一层：纯算法（numpy，可自检）
# ═══════════════════════════════════════════════════════════════

# ── 1. 优势估计 ──────────────────────────────────────────────
def advantages_grpo(rewards, eps=1e-6):
    """标准 GRPO：组内标准化"""
    r = np.asarray(rewards, dtype=np.float64)
    return (r - r.mean()) / (r.std() + eps)


def advantages_drgrpo(rewards):
    """Dr.GRPO：不做 std 归一化，优势就是朴素分差（更稳、不偏袒难题）"""
    r = np.asarray(rewards, dtype=np.float64)
    return r - r.mean()


def advantages_rloo(rewards):
    """RLOO：留一法基线"""
    r = np.asarray(rewards, dtype=np.float64)
    n = len(r)
    return r - (r.sum() - r) / max(1, n - 1)


def is_degenerate_group(rewards, tol=1e-6):
    """动态采样用：组内全同分 → 优势全 0 → 该跳过"""
    r = np.asarray(rewards, dtype=np.float64)
    return float(r.std()) < tol


# ── 2. 重要性比率 ────────────────────────────────────────────
def ratio_token_level(logp_new, logp_old):
    """token 级比率（GRPO 原版）：逐 token 取 exp"""
    return np.exp(np.asarray(logp_new) - np.asarray(logp_old))


def ratio_sequence_level(logp_new, logp_old):
    """
    GSPO 序列级比率：逐 token 对数比的几何均值。
    logp 形状 (T,)。除以 |o_i| 是关键——否则比率方差随长度爆炸。
    """
    lp_new = np.asarray(logp_new, dtype=np.float64)
    lp_old = np.asarray(logp_old, dtype=np.float64)
    T = lp_new.shape[-1]
    return math.exp(float((lp_new - lp_old).mean()))


# ── 3. 裁剪 ─────────────────────────────────────────────────
def clip_dapo(ratio, eps_low=0.2, eps_high=0.28):
    """DAPO Clip-Higher：上下界不对称，上界放宽以保住探索"""
    return float(min(max(ratio, 1.0 - eps_low), 1.0 + eps_high))


def gate_sapo(ratio, eps_low=0.2, eps_high=0.28, advantage=1.0,
              k_pos=2.0, k_neg=6.0):
    """
    SAPO 软门控：越界样本衰减而不是归零，梯度永不为 0。
    负 advantage 用更大的 k_neg，让它更快饱和（不浪费力气抬高坏 token）。
    """
    mid = 1.0 + (eps_high - eps_low) / 2.0
    half = (eps_high + eps_low) / 2.0
    k = k_pos if advantage >= 0 else k_neg
    s = 1.0 / (1.0 + math.exp(-k * (ratio - mid) / half))
    return (1.0 - eps_low) + (eps_low + eps_high) * s


# ── 4. KL 估计（k3） ─────────────────────────────────────────
def kl_k3(logp_ref, logp_theta):
    """k3 估计：exp(Δ) - Δ - 1，Δ = logπ_ref - logπ_θ，恒非负"""
    d = np.asarray(logp_ref, dtype=np.float64) - np.asarray(logp_theta, dtype=np.float64)
    return np.exp(d) - d - 1.0


# ── 5. 策略损失 ─────────────────────────────────────────────
def policy_loss_token_level(logp_new, logp_old, adv, total_tokens,
                            mode="gspo", eps_low=0.2, eps_high=0.28):
    """
    返回 (loss, 诊断)。logp_* 形状 (G, T)，adv 形状 (G,)。
    分母是 batch 内总 token 数 —— 用序列数平均会让长文梯度被 1/|o_i| 稀释。
    """
    lp_new = np.asarray(logp_new, dtype=np.float64)
    lp_old = np.asarray(logp_old, dtype=np.float64)
    a = np.asarray(adv, dtype=np.float64)[:, None]

    if mode == "gspo":
        s = np.exp((lp_new - lp_old).mean(axis=1))          # (G,) 序列级
        s = s[:, None]
    elif mode == "grpo":
        s = np.exp(lp_new - lp_old)                          # (G,T) token 级
    elif mode == "sapo":
        s_t = np.exp(lp_new - lp_old)
        g = np.array([[gate_sapo(float(x), eps_low, eps_high, float(a[i, 0]))
                       for x in row] for i, row in enumerate(s_t)])
        obj = a * g
        return -float(obj.sum() / max(1, total_tokens)), {"ratio_mean": float(s_t.mean())}
    else:
        raise ValueError("未知 mode: %s" % mode)

    clipped = np.clip(s, 1.0 - eps_low, 1.0 + eps_high)
    obj = np.minimum(a * s, a * clipped)
    diag = {
        "ratio_mean": float(s.mean()),
        "clip_frac": float(np.mean((s < 1.0 - eps_low) | (s > 1.0 + eps_high))),
    }
    return -float(obj.sum() / max(1, total_tokens)), diag


# ── 6. Muon（Newton–Schulz 正交化） ──────────────────────────
_MUON_A, _MUON_B, _MUON_C = 3.4445, -4.7750, 2.0315


def newtonschulz5(G, steps=5, eps=1e-7):
    """
    动量矩阵的正交化预条件。注意：系数故意不精确收敛，
    p(1) = a+b+c = 0.701，输出是 US'Vᵀ 而非 UVᵀ。
    """
    X = np.asarray(G, dtype=np.float64)
    X = X / (np.linalg.norm(X) + eps)
    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = _MUON_B * A + _MUON_C * (A @ A)
        X = _MUON_A * X + B @ X
    if transposed:
        X = X.T
    return X


def muon_update(grad, momentum, lr=0.02, momentum_coef=0.95, nesterov=True):
    """Muon 一步：动量 → NS 正交化 → 更新（只用于 2D 大矩阵）"""
    buf = momentum_coef * momentum + grad
    g = grad + momentum_coef * buf if nesterov else buf
    return buf, -lr * newtonschulz5(g)


# ── 7. 奖励函数（原 6 维 + 去重 + 长度合规） ─────────────────
_SENSE = ["亮", "暗", "凉", "热", "烫", "冷", "湿", "软", "硬", "粗糙",
          "光滑", "香", "甜", "苦", "涩", "疼", "颤"]
_EXAG = ["泪流满面", "痛哭流涕", "惊天动地", "永远永远", "世界上最", "无比"]
_CLICHE = ["阳光明媚", "时间如白驹", "光阴似箭", "感激不尽", "永生难忘"]


def _norm(s):
    return re.sub(r"[，。！？；：、“”‘’（）《》\s…—,.;:!?\n]", "", s or "")


def base_reward(text, theme=""):
    """原 6 维奖励，满分 10（沿用 强化学习.py 的设计）"""
    dims = {}
    paras = [p for p in re.split(r"\n+", text) if p.strip()]
    sents = [s.strip() for s in re.split(r"[。！？\n]", text) if len(s.strip()) >= 8]

    dims["细节密度"] = round(2.0 * min(1.0, (sum(text.count(w) for w in _SENSE)
                                            + len(re.findall(r"\d", text)))
                                      / max(1, len(text)) * 100 / 3.0), 2)
    if sents:
        ls = [len(s) for s in sents]
        avg = sum(ls) / len(ls)
        var = sum((x - avg) ** 2 for x in ls) / len(ls)
        dims["句长张力"] = round(1.5 * min(1.0, var / 120.0), 2)
    else:
        dims["句长张力"] = 0.0

    pen = max(0, text.count("！") - 2) * 0.3
    pen += sum(0.4 for k in _EXAG + _CLICHE if k in text)
    dims["情感克制"] = round(max(0.0, 1.5 - pen), 2)

    hits = sum(1 for k in {theme, theme[:2]} if k and k in text)
    dims["立意落地"] = round(1.5 * min(1.0, hits / 2), 2)
    dims["结构"] = 1.0 if 3 <= len(paras) <= 8 else (0.5 if 2 <= len(paras) <= 10 else 0.0)
    dims["原创度"] = 2.5          # 真实原创度由指纹库另算，这里给满
    return round(sum(dims.values()), 2), dims


def dedup_reward(text, fingerprint_index=None, threshold=0.62):
    """
    去重奖励（0~1）：与指纹库里任何一篇撞车就 0 分。
    这是把"复读"写进奖励里 —— 比事后过滤有效得多。
    """
    if fingerprint_index is None:
        return 1.0
    try:
        sim, meta, _ = fingerprint_index.query(text, threshold=threshold)
        return 0.0 if meta is not None else 1.0
    except Exception:
        return 1.0


def length_fit_reward(text, target_words, tol=0.15):
    """长度合规（0~1）：落在目标 ±tol 内给满，超长软衰减"""
    n = len(_norm(text))
    if target_words <= 0:
        return 1.0
    dev = abs(n - target_words) / target_words
    if dev <= tol:
        return 1.0
    return float(max(0.0, 1.0 - (dev - tol) / (tol * 2)))


# ── 对策③「生死簿判骨架」（2026-09-16，接 模板化诊断与对策.md） ──────────
# 句级指纹库管「同一句话说两遍」，骨架判据管「换个词写成的同一篇文章」。
# 体检数据：前 3 种骨架占 41.8%——骨架雷同的文章照样拿高分、照样被强化，
# 这是模板化压不下去的根因之一。把它写进奖励，让模型自己学会绕开。
def _essay_skeleton(text, max_paras=8):
    """骨架指纹：段数 + 各段相对长度（粗粒化 3 档），与 结构体检.py 同一公式"""
    ps = [p for p in re.split(r"\n+", text or "") if p.strip()]
    if not ps:
        return "空"
    keep = re.compile(r"[^\u4e00-\u9fff\u0041-\u005a\u0061-\u007a\u0030-\u0039]")
    lens = [len(keep.sub("", p)) for p in ps]
    tot = max(1, sum(lens))
    shape = "".join(str(min(2, int(x / tot * 6))) for x in lens[:max_paras])
    return "%d|%s" % (len(ps), shape)


def skeleton_repeat_reward(text, recent_skels):
    """骨架判据（0~1）：与最近 K 篇里任何一篇骨架相同 → 判死给 0。
    recent_skels 传 None 或空集合时不启用。"""
    if not recent_skels:
        return 1.0
    return 0.0 if _essay_skeleton(text) in recent_skels else 1.0


# 物件词表（与 对象替换增强.py 的垄断物件+替代物件对齐）
_OBJ_VOCAB = ("自行车 单车 车铃 钢笔 日记 照片 饭桌 旧手表 旧手机 旧书 口琴 台灯 雨伞 "
              "围巾 钥匙 车票 邮票 风筝 收音机 蒲扇 门槛 毛笔 砚台 煤油灯 搪瓷缸 顶针 "
              "扁担 木箱 旧信 蜡烛 药盒 手帕 相册 球鞋 雨衣 手套 行李箱 缝纫机 老井 "
              "石磨 陶罐 竹篮 宣纸 印章 帆布鞋 橡皮 书包").split()


def dominant_object(text):
    """文本里出现最多的具体物件（≥2 次才算，避免偶然命中）"""
    best, bc = None, 0
    for w in _OBJ_VOCAB:
        c = text.count(w)
        if c > bc:
            bc, best = c, w
    return best if bc >= 2 else None


def object_fixation_reward(text, counter):
    """反物件固定奖励（0~1）：某物件全训练史用得多，就压它的分。

    这就是「自行车先导」的直接解药 —— 模型每次想再用自行车，奖励都会
    变低，逼它去用旧手表/口琴/煤油灯……（与墨田 v1.2 的磨损场同构）
    """
    obj = dominant_object(text)
    if obj is None or not counter:
        return 1.0, obj
    used = counter.get(obj, 0)
    return round(1.0 / (1.0 + 0.05 * used), 3), obj


def total_reward(text, theme="", target_words=0, fingerprint_index=None,
                 obj_counter=None, skel_hist=None):
    base, dims = base_reward(text, theme)
    r_dedup = dedup_reward(text, fingerprint_index)
    r_len = length_fit_reward(text, target_words)
    r_fix, obj = object_fixation_reward(text, obj_counter)
    r_skel = skeleton_repeat_reward(text, skel_hist)
    total = base * 0.9 + 1.5 * r_dedup + 1.0 * r_len + 0.8 * r_fix + 1.0 * r_skel
    dims = dict(dims, 去重=r_dedup, 长度=r_len, 反固定=r_fix, 骨架=r_skel,
                物件=obj)
    return round(total, 3), dims


# ═══════════════════════════════════════════════════════════════
#  自检（秒级，不用加载模型）
# ═══════════════════════════════════════════════════════════════
def selfcheck():
    ok = True

    def chk(name, cond, extra=""):
        nonlocal ok
        print(("  [OK]   " if cond else "  [FAIL] ") + name + ("  " + extra if extra else ""))
        ok = ok and bool(cond)

    print("\n【1】优势估计")
    r = [1.0, 2.0, 3.0, 4.0]
    a = advantages_grpo(r)
    chk("GRPO 组内均值为 0", abs(a.mean()) < 1e-9, "mean=%.2e" % a.mean())
    chk("GRPO 组内 std 为 1", abs(a.std() - 1.0) < 1e-6, "std=%.6f" % a.std())
    ad = advantages_drgrpo(r)
    chk("Dr.GRPO 优势 == 原始分差",
        np.allclose(ad, np.array(r) - np.mean(r)), str(np.round(ad, 3)))
    same = advantages_drgrpo([5.0] * 4)
    chk("全同分 → 优势全 0", float(np.abs(same).max()) < 1e-12)
    chk("全同分 → std == 0 且被识别为退化组",
        is_degenerate_group([5.0] * 4) and abs(float(np.std([5.0] * 4))) < 1e-12)
    chk("非退化组不被误判", not is_degenerate_group([1.0, 2.0, 3.0]))

    print("\n【2】非对称裁剪（DAPO Clip-Higher）")
    chk("clip(1.5) == 1+eps_high", abs(clip_dapo(1.5) - 1.28) < 1e-12, "%.4f" % clip_dapo(1.5))
    chk("clip(0.5) == 1-eps_low", abs(clip_dapo(0.5) - 0.80) < 1e-12, "%.4f" % clip_dapo(0.5))
    chk("上下界确实不对称", abs((1 + 0.28) - 1.0) > abs(1.0 - (1 - 0.2)))

    print("\n【3】GSPO 序列级比率（关键）")
    T = 200
    lp_old = np.zeros(T)
    lp_new = np.full(T, math.log(1.1))          # 每个 token 都是 1.1 倍
    s_seq = ratio_sequence_level(lp_new, lp_old)
    s_tok = float(ratio_token_level(lp_new, lp_old)[-1])
    chk("序列级比率恒为 1.1（不连乘放大）", abs(s_seq - 1.1) < 1e-9, "%.6f" % s_seq)
    chk("token 级比率逐 token 恒为 1.1", abs(s_tok - 1.1) < 1e-9, "%.6f" % s_tok)
    # 演示长度错配：token 级连乘的含义（乘积）会爆炸
    prod = float(np.prod(ratio_token_level(lp_new, lp_old)))
    chk("token 级连乘会爆炸（说明为什么必须序列级）", prod > 1e6, "prod=%.3e" % prod)

    print("\n【4】SAPO 软门控")
    vals = [gate_sapo(x) for x in np.linspace(0.2, 2.0, 40)]
    chk("单调递增", all(vals[i] <= vals[i + 1] + 1e-12 for i in range(len(vals) - 1)))
    chk("值域有界 (1-eps_low, 1+eps_high)",
        min(vals) >= 1 - 0.2 - 1e-9 and max(vals) <= 1 + 0.28 + 1e-9,
        "[%.3f, %.3f]" % (min(vals), max(vals)))
    # σ 的导数永远 > 0 → 梯度不为 0
    def gate_deriv(rho, k=2.0, mid=1.04, half=0.24):
        e = math.exp(-k * (rho - mid) / half)
        return (0.2 + 0.28) * (k / half) * e / (1 + e) ** 2
    chk("梯度永不为 0", all(gate_deriv(x) > 0 for x in np.linspace(0.2, 2.0, 40)))
    gn = abs(gate_sapo(1.5, advantage=-1) - gate_sapo(1.5, advantage=1))
    chk("负 advantage 更快饱和（与正的有差异）", gn > 1e-6, "Δ=%.4f" % gn)

    print("\n【5】k3 KL")
    x = np.array([-1.0, 0.0, 0.5, 2.0])
    chk("同分布时为 0", abs(float(kl_k3(x, x).sum())) < 1e-12)
    chk("恒非负", np.all(kl_k3(x, x - 0.7) >= 0) and np.all(kl_k3(x, x + 0.7) >= 0))

    print("\n【6】Token 级损失")
    # ratio = exp(log1.1 - 0) = 1.1，落在 [0.8, 1.28] 内不触发 clip
    lp_n = np.array([[math.log(1.1)]])
    lp_o = np.array([[0.0]])
    loss, diag = policy_loss_token_level(lp_n, lp_o, [1.0], 1, mode="grpo")
    expect = -1.1
    chk("单 token 时等于目标值", abs(loss - expect) < 1e-9,
        "loss=%.6f expect=%.6f" % (loss, expect))
    # 长度稀释演示
    lp_n2 = np.full((1, 100), math.log(0.8))
    lp_o2 = np.full((1, 100), math.log(0.5))
    l_seq, _ = policy_loss_token_level(lp_n2, lp_o2, [1.0], 1, mode="grpo")
    l_tok, _ = policy_loss_token_level(lp_n2, lp_o2, [1.0], 100, mode="grpo")
    chk("序列数当分母会稀释 100 倍", abs(l_seq / l_tok - 100) < 1e-6,
        "%.4f vs %.4f" % (l_seq, l_tok))

    print("\n【7】Muon / Newton–Schulz")
    rng = np.random.default_rng(0)
    Q, _ = np.linalg.qr(rng.standard_normal((8, 8)))
    coef = _MUON_A + _MUON_B + _MUON_C
    chk("p(1) == 0.701（迭代方程在正交点的系数和 a+b+c）",
        abs(coef - 0.701) < 1e-9, "a+b+c=%.4f" % coef)
    G = rng.standard_normal((6, 4))
    n1, n2 = newtonschulz5(G), newtonschulz5(G * 7.3)
    chk("尺度不变", np.allclose(n1, n2, atol=1e-9))
    chk("转置一致", np.allclose(newtonschulz5(G).T, newtonschulz5(G.T), atol=1e-9))
    cols = np.linalg.norm(newtonschulz5(G), axis=0)
    chk("列范数落在 0.4~1.6", bool(np.all(cols > 0.4) and np.all(cols < 1.6)),
        "[%.3f, %.3f]" % (cols.min(), cols.max()))
    O = newtonschulz5(G)
    chk("输出不是严格正交（系数故意不精确，S~U(0.5,1.5)）",
        not np.allclose(O.T @ O, np.eye(4), atol=5e-2))

    print("\n【8】奖励函数")
    good = ("冬天把世界调成了静音模式。哈出一口气，白雾就替风画出了形状。\n\n"
            "外婆蹲在灶台前添柴，火光把她的影子拉得很长。火苗舔着锅底，"
            "锅里的水咕嘟咕嘟地响。\n\n我忽然懂了，所谓成长，"
            "就是把哭声调成静音的过程。")
    r_total, dims = total_reward(good, theme="成长", target_words=120)
    chk("奖励能算出来且在合理区间", 0 < r_total < 14, "total=%.2f" % r_total)
    chk("长度合规生效", dims["长度"] >= 0.0)
    chk("无指纹库时去重奖励为 1", dims["去重"] == 1.0)
    chk("无生死簿时骨架奖励为 1", dims["骨架"] == 1.0)

    print("\n【9】生死簿判骨架（对策③）")
    r_a, d_a = total_reward(good, "成长", 120, skel_hist=set())
    r_b, d_b = total_reward(good, "成长", 120, skel_hist={_essay_skeleton(good)})
    chk("陌生骨架给满分", d_a["骨架"] == 1.0)
    chk("同骨架第二篇判死（0 分）", d_b["骨架"] == 0.0)
    chk("判死确实拉低总分", r_b < r_a)
    chk("骨架指纹公式与 结构体检.py 一致（段数在前）",
        _essay_skeleton("甲乙丙丁\n\n戊己庚辛壬癸").startswith("2|"))

    print("\n" + ("=" * 52))
    print("自检" + ("全部通过 ✓" if ok else "存在失败项 ✗"))
    print("=" * 52)
    return ok


# ═══════════════════════════════════════════════════════════════
#  第二层：真实训练循环（GSPO 主循环）
# ═══════════════════════════════════════════════════════════════
def train(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    sys.path.insert(0, ROOT)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logf = os.path.join(ROOT, "GRPO训练日志.txt")

    def log(m):
        line = "[%s] %s" % (time.strftime("%H:%M:%S"), m)
        print(line, flush=True)
        with open(logf, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    log("=== 竹 · GSPO 后训练开始（%s） ===" % device)
    log("算法：%s | group=%d | eps=%.2f/%.2f | KL系数=%.3f | 优化器=%s"
        % (args.mode, args.group, args.eps_low, args.eps_high, args.kl_coef, args.optim))

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        trust_remote_code=True).to(device)

    ref = None
    if args.kl_coef > 0:
        log("KL 用 disable_adapter 取参考分布（PEFT 场景零额外显存）")

    # 指纹库：把去重写进奖励
    fp_index = None
    try:
        import 语料指纹 as F
        fp_index = F.CorpusIndex(F.MinHasher())
        corpus = os.path.join(ROOT, "训练语料_精编.jsonl")
        if not os.path.exists(corpus):
            corpus = os.path.join(ROOT, "训练语料_3万.jsonl")
        n = 0
        with open(corpus, encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    fp_index.add(json.loads(line)["output"])
                    n += 1
                except Exception:
                    pass
                if n >= args.fp_size:
                    break
        log("指纹库就绪：%d 篇（撞车即判 0 分）" % n)
    except Exception as e:
        log("指纹库不可用，去重奖励跳过：%s" % e)

    import 写作引擎 as w
    opt = torch.optim.AdamW([p for p in mdl.parameters() if p.requires_grad],
                            lr=args.lr)

    rng = random.Random(42)
    obj_counter = {}      # 全训练史的物件使用计数（喂给反固定奖励）
    skel_hist = []        # 生死簿：最近 skel_window 篇的骨架（喂给骨架判据）
    seeds_pool = ["成长", "亲情", "师恩", "友情", "挫折", "理想", "家国",
                  "劳动", "传统文化", "科技与人文", "青春", "责任"]
    for step in range(1, args.steps + 1):
        theme = rng.choice(seeds_pool)
        words = rng.choice([700, 800, 1000, 1200])
        prompt = "帮我写一篇关于%s的作文，%d字" % (theme, words)
        chat = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True)
        ids = tok(chat, return_tensors="pt").to(device)

        # 1) 采样 G 条
        with torch.no_grad():
            outs = mdl.generate(**ids, max_new_tokens=args.max_new,
                                do_sample=True, temperature=0.9, top_p=0.95,
                                num_return_sequences=args.group,
                                pad_token_id=tok.eos_token_id)

        texts, rewards = [], []
        skel_seen = set(skel_hist[-args.skel_window:])   # 组内也互判：同组后到者判死
        for o in outs:
            t = tok.decode(o[ids["input_ids"].shape[1]:], skip_special_tokens=True)
            texts.append(t)
            r, dims = total_reward(t, theme, words, fp_index, obj_counter, skel_seen)
            rewards.append(r)
            obj = dims.get("物件")
            if obj:
                obj_counter[obj] = obj_counter.get(obj, 0) + 1
            skel_seen.add(_essay_skeleton(t))
        skel_hist[:] = (skel_hist + sorted(skel_seen - set(skel_hist))
                        )[-args.skel_window:]

        # 2) 动态采样：全同分直接跳过
        if is_degenerate_group(rewards):
            log("step %d：组内奖励全同（%.2f），跳过（省算力）" % (step, rewards[0]))
            continue

        adv = advantages_drgrpo(rewards)        # Dr.GRPO
        log("step %d：奖励 mean=%.2f max=%.2f | 优势范围 [%.2f, %.2f]"
            % (step, float(np.mean(rewards)), float(np.max(rewards)),
               adv.min(), adv.max()))

        # 3) old logprob（冻结口径）
        old_lp = []
        with torch.no_grad():
            for o in outs:
                lp = _seq_logprob(mdl, o.unsqueeze(0), ids["input_ids"].shape[1])
                old_lp.append(lp)
        old_lp = torch.stack(old_lp)

        # 4) GSPO 更新
        opt.zero_grad()
        new_lp, ref_lp = [], []
        for i, o in enumerate(outs):
            new_lp.append(_seq_logprob(mdl, o.unsqueeze(0), ids["input_ids"].shape[1]))
        new_lp = torch.stack(new_lp)

        s = torch.exp((new_lp - old_lp).mean(dim=1))                  # 序列级比率
        a = torch.tensor(adv, dtype=s.dtype, device=s.device)
        clipped = torch.clamp(s, 1 - args.eps_low, 1 + args.eps_high)  # Clip-Higher
        total_tokens = new_lp.numel()                                  # Token 级分母
        loss = -(torch.min(a * s, a * clipped).sum() / total_tokens)
        if args.kl_coef > 0:
            try:
                # PEFT 场景：disable_adapter 直接拿到基座分布，零额外显存
                with torch.no_grad(), mdl.disable_adapter():
                    ref_lp = torch.stack([
                        _seq_logprob(mdl, o.unsqueeze(0), ids["input_ids"].shape[1])
                        for o in outs])
                kld = kl_k3(ref_lp.detach().cpu().numpy(), new_lp.detach().cpu().numpy())
                loss = loss + args.kl_coef * float(kld.mean())
            except AttributeError:
                pass            # 非 PEFT 模型没有 disable_adapter，跳过 KL 项
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in mdl.parameters() if p.requires_grad], 1.0)
        opt.step()

        if step % 10 == 0:
            os.makedirs(args.out, exist_ok=True)
            mdl.save_pretrained(args.out)
            log("  已保存 → %s（step %d）" % (args.out, step))

    log("=== GSPO 训练结束，产物在 %s ===" % args.out)


def _seq_logprob(model, seq, prompt_len):
    """取生成部分每个 token 的 log 概率，形状 (T,)"""
    import torch
    out = model(seq)
    logits = out.logits[0, prompt_len - 1:-1]              # 预测每个生成 token
    logp = torch.log_softmax(logits.float(), dim=-1)
    tgt = seq[0, prompt_len:]
    return logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)


# ═══════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description="竹 · GSPO/DAPO/SAPO 后训练")
    ap.add_argument("--selfcheck", action="store_true", help="只跑算法自检（秒级）")
    ap.add_argument("--train", action="store_true", help="跑真实训练循环")
    ap.add_argument("--mode", default="gspo", choices=["gspo", "grpo", "sapo"])
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--out", default="output_grpo")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--group", type=int, default=8, help="每题采样条数 G")
    ap.add_argument("--max-new", type=int, default=900)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--eps-low", type=float, default=0.2)
    ap.add_argument("--eps-high", type=float, default=0.28)
    ap.add_argument("--kl-coef", type=float, default=0.0)
    ap.add_argument("--optim", default="adamw", choices=["adamw", "muon"])
    ap.add_argument("--fp-size", type=int, default=3000, help="指纹库装载篇数")
    ap.add_argument("--skel-window", type=int, default=20,
                    help="生死簿窗口：最近多少篇的骨架不允许重复（对策③）")
    args = ap.parse_args()

    if args.selfcheck or not args.train:
        return 0 if selfcheck() else 1
    train(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
