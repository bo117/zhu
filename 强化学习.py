# -*- coding: utf-8 -*-
"""
强化学习.py —— 竹 · 奖励函数 + RAFT 强化学习驱动（拒绝采样微调）
================================================================
不依赖海外模型打分：奖励函数完全本地自研，规则 + 统计双路：
  R = 2.5×原创度（与已入库语料重合越低越高）
    + 2.0×细节密度（数字/颜色/感官/具体名词 百字密度）
    + 1.5×句长张力（长短句方差， penalize 单调）
    + 1.5×情感克制（感叹号/浮夸词/排比 扣分）
    + 1.5×立意落地（主题词与立意词出现）
    + 1.0×结构完整（3~8 段，首尾成篇）
  满分 10。

RAFT 循环（Deep Learning + Reinforcement Learning）：
  每轮：种子 → Writer 生成 K 篇（温度扰动）→ 奖励打分 →
        保留 top ρ（≥7.5）追加进 强化语料_rl.jsonl →
        调用 训练.py 对 LoRA 适配器做一轮 SFT 更新（策略梯度的一阶近似）
  一轮 = 一次策略更新；迭代越久策略越强（本机 CPU 慢，正式强度放上超算）。

用法：
  python 强化学习.py --iterations 3 --k 60 --keep-top 0.3
"""
import io
import json
import os
import random
import re
import subprocess
import sys
import time

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8',
                                  errors='replace')
ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import 写作引擎 as w          # noqa: E402
import 作文引擎 as base       # noqa: E402
import 写作素材库 as suc      # noqa: E402


def log(msg):
    print('[%s] %s' % (time.strftime('%H:%M:%S'), msg), flush=True)


# ──────────────── 奖励函数（本地自研，0~10）────────────────
_SENSE_WORDS = ['亮', '暗', '凉', '热', '烫', '冷', '湿', '软', '硬', '粗糙',
                '光滑', '香', '臭', '甜', '咸', '酸', '苦', '涩', '疼', '颤']
_FLOAT_WORDS = ['叹', '红了眼眶', '鼻子一酸', '沉默', '低头', '愣住', '顿了顿']
_EXAG = ['泪流满面', '痛哭流涕', '惊天动地', '永远永远', '世界上最', '无比', '无比']
_CLICHE = ['阳光明媚', '时间如白驹', '光阴似箭', '感激不尽', '永生难忘']


def _norm(s):
    return re.sub(r'[，。！？；：、""''（）《》\s…—,.;:!?\n]', '', s)


def compute_reward(text, seed_theme, used_registry):
    """多维度奖励函数：返回 (总分 0~10, 分项明细 dict)"""
    dims = {}
    paras = [p for p in text.split('\n\n') if p.strip()]
    sents = [s.strip() for s in re.split(r'[。！？\n]', text) if len(s.strip()) >= 8]

    # 1) 原创度 2.5 —— 与已入库语料的句子重合率
    norm = {_norm(s) for s in sents if len(_norm(s)) >= 10}
    overlap = (len(norm & used_registry) / len(norm)) if norm else 0.0
    dims['原创度'] = round(2.5 * max(0.0, 1.0 - overlap * 2.5), 2)

    # 2) 细节密度 2.0 —— 数字/颜色/感官词/具体名词 百字密度
    marks = 0
    for w_ in _SENSE_WORDS:
        marks += text.count(w_)
    marks += len(re.findall(r'\d', text))
    marks += sum(text.count(k) for k in ('资', '巷', '灶', '檐', '袖', '靴',
                                         '篱', '釉', '萝', '栗'))
    density = marks / max(1, len(text)) * 100
    dims['细节密度'] = round(2.0 * min(1.0, density / 3.0), 2)

    # 3) 句长张力 1.5 —— 长短句方差
    if sents:
        ls = [len(s) for s in sents]
        avg = sum(ls) / len(ls)
        var = sum((x - avg) ** 2 for x in ls) / len(ls)
        dims['句长张力'] = round(1.5 * min(1.0, var / 120.0), 2)
    else:
        dims['句长张力'] = 0.0

    # 4) 情感克制 1.5 —— 感叹号少、浮夸词少、悲伤直白词少
    pen = 0.0
    pen += max(0, text.count('！') - 2) * 0.3
    for k in _EXAG + _CLICHE:
        if k in text:
            pen += 0.4
    float_cnt = sum(text.count(k) for k in _FLOAT_WORDS)
    if float_cnt > 5:
        pen += 0.5
    dims['情感克制'] = round(max(0.0, 1.5 - pen), 2)

    # 5) 立意落地 1.5 —— 主题词与种子立意词出现
    hits = sum(1 for k in {seed_theme, seed_theme[:2]} if k and k in text)
    dims['立意落地'] = round(1.5 * min(1.0, hits / 2), 2)

    # 6) 结构完整 1.0 —— 段落数 3~8，首段短、末段收
    if 3 <= len(paras) <= 8:
        dims['结构'] = 1.0
    elif 2 <= len(paras) <= 10:
        dims['结构'] = 0.5
    else:
        dims['结构'] = 0.0

    total = round(sum(dims.values()), 2)
    return total, dims


# ──────────────── RAFT 循环 ────────────────
def gen_seeds(k, rng):
    opts = w.options()
    themes = [x['name'] for x in opts['themes']]
    out = []
    for _ in range(k):
        theme = rng.choice(themes)
        words = rng.choice([800, 1000, 1200])
        q = '帮我写一篇关于%s的作文，%d字' % (theme, words)
        if rng.random() < 0.4:
            q += '，结局' + rng.choice([e['name'] for e in opts['endings']])
        if rng.random() < 0.3:
            q += '，模仿' + rng.choice([m['name'] for m in opts['masters']])
        brief = w.parse_brief(q)
        brief['words'] = words
        brief['theme'] = theme
        out.append((q, brief, theme))
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--iterations', type=int, default=3)
    ap.add_argument('--k', type=int, default=60, help='每轮生成篇数')
    ap.add_argument('--keep-top', type=float, default=0.3)
    ap.add_argument('--min-reward', type=float, default=7.5)
    ap.add_argument('--out', default='output_rl')
    args = ap.parse_args()

    logf = os.path.join(ROOT, '强化学习日志.txt')

    def log(m):
        line = '[%s] %s' % (time.strftime('%H:%M:%S'), m)
        print(line, flush=True)
        with open(logf, 'a', encoding='utf-8') as f:
            f.write(line + '\n')

    rl_path = os.path.join(ROOT, '强化语料_rl.jsonl')
    used_registry = set()
    if os.path.exists(rl_path):
        with open(rl_path, encoding='utf-8') as f:
            for line in f:
                try:
                    body = json.loads(line)['output']
                    for s in re.split(r'[。！？\n]', body):
                        ns = _norm(s)
                        if len(ns) >= 10:
                            used_registry.add(ns)
                except Exception:
                    pass

    rng = random.Random(7)
    for it in range(1, args.iterations + 1):
        log('=== 强化学习第 %d/%d 轮 ===' % (it, args.iterations))
        seeds = gen_seeds(args.k, rng)
        scored = []
        for q, brief, theme in seeds:
            try:
                title, text = w.write_essay2(brief)
            except Exception:
                continue
            reward, dims = compute_reward(text, theme, used_registry)
            scored.append((reward, q, title, text, dims))
        scored.sort(key=lambda x: -x[0])
        keep_n = max(1, int(len(scored) * args.keep_top))
        kept = [s for s in scored[:keep_n] if s[0] >= args.min_reward]
        if not kept:
            kept = scored[:1]
        avg_all = sum(s[0] for s in scored) / max(1, len(scored))
        avg_keep = sum(s[0] for s in kept) / max(1, len(kept))
        log('  本轮 %d 篇：平均 %.2f，保留 %d 篇（均值 %.2f）'
            % (len(scored), avg_all, len(kept), avg_keep))
        for r_, q_, t_, x_, dims_ in kept[:3]:
            log('  样例 %.1f 分：%s' % (r_, t_))
        with open(rl_path, 'a', encoding='utf-8') as f:
            for r_, q_, t_, x_, dims_ in kept:
                out_x = ('《%s》\n\n%s' % (t_, x_))
                f.write(json.dumps({'instruction': q_, 'output': out_x,
                                    'reward': r_},
                                   ensure_ascii=False) + '\n')
                for s in re.split(r'[。！？\n]', x_):
                    ns = _norm(s)
                    if len(ns) >= 10:
                        used_registry.add(ns)
        # 策略更新：对高奖励样本做一轮 SFT（RAFT = 一阶策略梯度近似）
        n_rl = sum(1 for _ in open(rl_path, encoding='utf-8'))
        if n_rl >= 200:
            log('  策略更新：用 %d 条高奖励样本微调适配器 …' % min(n_rl, 4000))
            subprocess.run([
                sys.executable, os.path.join(ROOT, '训练.py'),
                '--model', 'Qwen/Qwen2.5-0.5B-Instruct',
                '--essay-only', '--max-samples', str(min(n_rl, 4000)),
                '--epochs', '1', '--batch', '1', '--accum', '8',
                '--lora-r', '8', '--max-len', '384',
                '--out', args.out], check=False)
        else:
            log('  高奖励样本 %d/200，先攒几轮再更新策略' % n_rl)
    log('=== 强化学习循环结束，策略保存在 %s ===' % args.out)


if __name__ == '__main__':
    main()
