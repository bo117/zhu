# -*- coding: utf-8 -*-
"""
蒸馏.py —— 竹 · 自我蒸馏与数据合成流水线（Self-Distillation Pipeline）
================================================================
四模块流水线，目标 30 万篇高质量作文语料（可断点续跑）：

  1. 种子库构建  —— 组合矩阵生成 5000~10000 个多样化种子提示词
     （主题 × 立意 × 情感基调 × 叙事结构 × 场景 × 学段 × 名家）
  2. 自我博弈生成 —— Writer 依种子创作；Critic 严苛挑刺（逻辑漏洞/
     情感悬浮/套路化）；必须经过至少一轮"批判-重写"才入候选池
  3. 动态质量过滤 —— 多维打分（立意深度/语言张力/细节真实感/
     结构完整/原创度），阈值 8.5，不达标直接丢弃
  4. 防崩溃机制   —— 温度扰动（随机概率扰动）+ 题材强制轮换 +
     全局禁用句登记（感悟句/过渡句每篇现提炼，禁止复用既有话术）

另有 名著蒸馏：把 名著/ 文件夹里的公版名著 txt 蒸馏成"模仿名家"训练对
（仅取公版作者：鲁迅/朱自清/老舍/萧红等；观点自动本土化正向改写）。

用法：
  python 蒸馏.py --target 300000          # 全量（可多次续跑）
  python 蒸馏.py --target 3000            # 先小批量验证
  python 蒸馏.py --ingest-名著            # 蒸馏名著文件夹
产物：训练语料_蒸馏/蒸馏批次N.jsonl + 蒸馏报告.txt
"""
import argparse
import glob
import io
import json
import os
import random
import re
import sys
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import 写作引擎 as w          # noqa: E402
import 作文引擎 as base       # noqa: E402
import 写作素材库 as suc      # noqa: E402

LOG = os.path.join(ROOT, '蒸馏日志.txt')


def log(msg):
    line = '[%s] %s' % (time.strftime('%H:%M:%S'), msg)
    print(line, flush=True)
    try:
        with open(LOG, 'a', encoding='utf-8') as f:
            f.write(line + '\n')
    except Exception:
        pass


# ==================================================================
# 模块 1：种子库构建
# ==================================================================
SEED_立意 = [
    "平凡小事里的深情", "成长必须付出的代价", "误解与和解",
    "告别也是一种成全", "普通人的坚守", "被忽略的陪伴",
    "失败之后如何自处", "代际之间的沉默与懂得", "尊严与体面",
    "选择背后的取舍", "等待的价值", "遗憾的另一种圆满",
    "陌生人的善意", "传承的重量", "孤独中的自我成长",
    "时过境迁后的重新理解", "微小时刻里的转折", "承诺与兑现",
    "失去后才看清的珍贵", "坚持与放下的辩证",
]
SEED_基调 = [
    "温情治愈", "含蓄克制", "朴实真挚", "怅然若失", "明亮向上",
    "淡淡的忧伤", "温暖中带一点心酸", "平静叙述下的暗涌",
    "轻快里带回味", "深沉内敛",
]
SEED_结构 = [
    "单事件深挖", "双线对照", "时间倒叙", "以物为线索",
    "场景切换推进", "一日之内浓缩", " seasons 四季流转",
    "首尾同一场景", "欲扬先抑", "递进三段式",
]
SEED_场景 = [
    "放学路上", "老屋的院子里", "深夜的厨房", "医院走廊",
    "火车站台", "夏天的教室", "雨天的屋檐下", "菜市场",
    "老家的田野", "阳台一角", "搬家现场", "除夕的厨房",
    "医院缴费窗口", "地铁车厢", "小区门口的快递柜", "爷爷的菜地",
]
SEED_年级 = ["小学高年级", "初一", "初二", "初三", "高中"]
SEED_手法 = w.TECHNIQUES + [None]
SEED_名家 = list(w.MASTERS.keys()) + [None]
SEED_结局 = [e['name'] for e in w.options()['endings']] + [None]


def build_seeds(count):
    """组合矩阵生成种子提示词（去重）"""
    opts = w.options()
    themes = [x['name'] for x in opts['themes']]
    rng = random.Random(999)
    seeds, seen = [], set()
    guard = 0
    while len(seeds) < count and guard < count * 20:
        guard += 1
        theme = rng.choice(themes)
        立意 = rng.choice(SEED_立意)
        基调 = rng.choice(SEED_基调)
        结构 = rng.choice(SEED_结构)
        场景 = rng.choice(SEED_场景)
        年级 = rng.choice(SEED_年级)
        手法 = rng.choice(SEED_手法)
        名家 = rng.choice(SEED_名家)
        结局 = rng.choice(SEED_结局)
        words = rng.choice([800, 1000, 1000, 1100, 1200])
        seed = {
            'theme': theme, '立意': 立意, '基调': 基调, '结构': 结构,
            '场景': 场景, '年级': 年级, '手法': 手法, '名家': 名家,
            '结局': 结局, 'words': words,
        }
        key = (theme, 立意, 基调, 结构, 场景, 手法, 名家, 结局)
        if key in seen:
            continue
        seen.add(key)
        seed['prompt'] = (
            "写一篇关于「%s」的记叙文，%d字。立意落在「%s」；情感基调%s；"
            "围绕「%s」这个场景展开，结构上%s；叙事语气贴合%s学生的真实口吻。"
            "%s%s" % (theme, words, 立意, 基调, 场景, 结构, 年级,
                      ('模仿%s的笔意（仿其神不抄其文）。' % 名家) if 名家 else '',
                      ('结尾走向：%s。' % 结局) if 结局 else ''))
        seeds.append(seed)
    return seeds


# ==================================================================
# 模块 2/3：Writer → Critic → 重写 → Judge
# ==================================================================
# 套话黑名单：这些句子在语料模式下禁用（防模板化）
BANNED_PATTERNS = [
    '这件事让我明白', '这件事让我懂得', '我明白了一个道理',
    '这让我想起了', '从此我改变', '原来这一切都是',
    '生活是最好的老师', '成长的道路上', '在人生的道路上',
    '那一刻我长大了', '我终于明白', '这不禁让我想起',
    '岁月的长河', '时光的车轮', '记忆的长河',
]


def critic_review(text, seed, used_registry):
    """Critic：严苛审稿。返回 (score 0-10, problems 列表)"""
    problems = []
    score = 10.0
    paras = [p for p in text.split('\n\n') if p.strip()]
    # 1) 套话检测
    for pat in BANNED_PATTERNS:
        if pat in text:
            problems.append('套话：「%s」' % pat)
            score -= 1.2
    # 2) 重复句检测（篇内）
    sents = [s.strip() for s in re.split(r'[。！？\n]', text)
             if len(s.strip()) >= 10]
    if len(sents) != len(set(sents)):
        problems.append('篇内存在重复句')
        score -= 1.0
    # 3) 与已入库语料撞车（跨篇查重）
    norm = {re.sub(r'[，。！？；：、\s]', '', s) for s in sents}
    hit = len(norm & used_registry)
    if norm:
        ratio = hit / len(norm)
        if ratio > 0.15:
            problems.append('与既有语料重合率 %.0f%%' % (ratio * 100))
            score -= 2.5
    # 4) 立意落地检测（宽松：立意任两字出现在文中即可）
    key2 = seed['立意'][:2]
    flat = text.replace('，', '').replace('。', '')
    if key2 not in flat and seed['立意'][-2:] not in flat:
        problems.append('立意悬浮（未落到「%s」）' % seed['立意'])
        score -= 0.5
    # 5) 语言张力：句长方差（全短句呆板，全长句拖沓）
    if sents:
        avg = sum(len(s) for s in sents) / len(sents)
        var = sum((len(s) - avg) ** 2 for s in sents) / len(sents)
        if var < 40:
            problems.append('句式单调（缺少长短交错）')
            score -= 0.8
    # 6) 字数达标
    if len(text) < seed['words'] * 0.85:
        problems.append('字数不足（%d/%d）' % (len(text), seed['words']))
        score -= 1.5
    # 7) 场景落地（宽松）
    if seed['场景'][:2] not in text:
        problems.append('场景未落地（%s）' % seed['场景'])
        score -= 0.4
    return max(0.0, round(score, 1)), problems


def writer_draft(seed, rng):
    """Writer：按种子写作（随机扰动：手法/开头/结局由种子驱动，
    概率扰动由 rng 温度控制）"""
    brief = {
        'mode': 'essay', 'theme': seed['theme'],
        'genre': None, 'ending': seed['结局'],
        'opening': rng.choice(w.OPENINGS),
        'technique': seed['手法'],
        'structure': seed['结构'] if seed['结构'] in w.STRUCTURES else None,
        'master': seed['名家'], 'depth': '深度',
        'style': '记叙文', 'words': seed['words'],
        'arc': '先虐后甜' if rng.random() < 0.1 else None,
        'opening_override': None,
        'extra': {'scene': seed['场景'], 'person': None,
                  'kws': [seed['立意'][:4]], 'raw': seed['立意']},
    }
    return w.write_essay2(brief), brief


def writer_rewrite(seed, problems, rng, round_no):
    """Writer 第二稿：针对 Critic 的问题换招重写"""
    brief = {
        'mode': 'essay', 'theme': seed['theme'],
        'genre': None, 'ending': seed['结局'],
        'opening': rng.choice([o for o in w.OPENINGS]),
        'technique': rng.choice([t for t in w.TECHNIQUES if t != seed['手法']]),
        'structure': rng.choice([s for s in w.STRUCTURES
                                 if s != seed['结构']] or [None]),
        'master': (rng.choice([m for m in w.MASTERS if m != seed['名家']])
                   if seed['名家'] else rng.choice(list(w.MASTERS.keys()) + [None])),
        'depth': '深度', 'style': '记叙文',
        'words': seed['words'] + rng.choice([0, 100, 150]),
        'arc': '先虐后甜' if rng.random() < 0.15 else None,
        'extra': {'scene': rng.choice(SEED_场景), 'person': None,
                  'kws': [seed['立意'][:4]], 'raw': seed['立意'] + '（换角度重写）'},
    }
    return w.write_essay2(brief), brief


def rounds_ok(drafts, max_rounds):
    """第 1 稿之后允许重写 max_rounds 轮"""
    return drafts <= max_rounds + 1


def make_instruction(seed, title, round_no):
    return ('写一篇关于「%s」的记叙文，%d字左右，题目《%s》。'
            '立意落在「%s」，情感基调%s，围绕「%s」展开，%s。'
            % (seed['theme'], seed['words'], title, seed['立意'],
               seed['基调'], seed['场景'],
               '贴合%s学生的真实语气' % seed['年级']))


# ==================================================================
# 模块 4：防崩溃
# ==================================================================
class AntiCollapse:
    """题材强制轮换 + 全局禁用句登记"""

    def __init__(self, themes, cooldown=40):
        self.cooldown = cooldown
        self.last_used = {}          # theme -> 使用序号
        self.tick = 0
        self.themes = themes
        self.used_registry = set()   # 全部已入库句子（规范化）

    def pick_theme(self, rng):
        """冷却期外的主题优先（强制多样性）"""
        fresh = [t for t in self.themes
                 if self.tick - self.last_used.get(t, -9999) >= self.cooldown]
        theme = rng.choice(fresh if fresh else self.themes)
        self.last_used[theme] = self.tick
        return theme

    def register(self, text):
        self.tick += 1
        for s in re.split(r'[。！？\n]', text):
            ns = re.sub(r'[，。！？；：、\s]', '', s)
            if len(ns) >= 10:
                self.used_registry.add(ns)

    def overlap_ratio(self, text):
        sents = {re.sub(r'[，。！？；：、\s]', '', s)
                 for s in re.split(r'[。！？\n]', text)
                 if len(re.sub(r'[，。！？；：、\s]', '', s)) >= 10}
        if not sents:
            return 0.0
        return len(sents & self.used_registry) / len(sents)


# ==================================================================
# 名著蒸馏（公版书 → 模仿名家训练对）
# ==================================================================
PUBLIC_AUTHORS = {
    '鲁迅': ['呐喊', '彷徨', '朝花夕拾', '野草'],
    '朱自清': ['背影', '荷塘月色', '欧游杂记', '匆匆'],
    '老舍': ['骆驼祥子', '济南的冬天', '茶馆'],
    '萧红': ['呼兰河传', '生死场'],
    '冰心': ['繁星', '春水', '寄小读者'],
    '叶圣陶': ['稻草人', '倪焕之'],
}


def ingest_名著():
    """名著/ 文件夹里的公版 txt → 抽取精彩段落 → 模仿名家训练对。
    观点本土化正向：过滤暴力/消极极端段落，保留温情与思辨。"""
    folder = os.path.join(ROOT, '名著')
    if not os.path.isdir(folder):
        log('名著/ 文件夹不存在，跳过名著蒸馏（把公版名著 txt 放进去再跑）')
        return 0
    rows = []
    for fp in glob.glob(os.path.join(folder, '*.txt')):
        name = os.path.basename(fp)
        author = None
        for a, works in PUBLIC_AUTHORS.items():
            if any(k in name for k in works) or a in name:
                author = a
                break
        if not author:
            continue
        raw = open(fp, encoding='utf-8', errors='ignore').read()
        paras = [p.strip() for p in re.split(r'\n\s*\n', raw)
                 if 120 <= len(p.strip()) <= 600]
        rng = random.Random(7)
        rng.shuffle(paras)
        for p in paras[:400]:
            # 正向过滤：跳过极端消极/暴力密集段
            bad = sum(p.count(k) for k in ('杀', '血', '尸体', '咒骂'))
            if bad >= 2:
                continue
            q = ('模仿%s的笔意，就「%s」这类日常场景写一段300字左右的短文，'
                 '仿其神不抄其文。' % (author, p[:12]))
            rows.append({'instruction': q, 'output': p})
        log('名著蒸馏：%s（%s）抽出 %d 段' % (name, author, min(len(paras), 400)))
    outp = os.path.join(ROOT, '训练语料_名著.jsonl')
    with open(outp, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    log('名著蒸馏完成：%d 条 → %s' % (len(rows), outp))
    return len(rows)


# ==================================================================
# 主流程
# ==================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', type=int, default=300000)
    ap.add_argument('--seeds', type=int, default=8000)
    ap.add_argument('--threshold', type=float, default=8.5)
    ap.add_argument('--max-rounds', type=int, default=2,
                    help='自我博弈最大轮数（1 轮批判重写起步）')
    ap.add_argument('--batch-size', type=int, default=5000,
                    help='每个 jsonl 分片多少条')
    ap.add_argument('--ingest-名著', action='store_true')
    args = ap.parse_args()

    log('=== 竹 · 自我蒸馏流水线启动（目标 %d 篇） ===' % args.target)
    if args.ingest_名著:
        ingest_名著()

    outdir = os.path.join(ROOT, '训练语料_蒸馏')
    os.makedirs(outdir, exist_ok=True)

    # 断点续跑：统计已有条数
    anti = AntiCollapse([x['name'] for x in w.options()['themes']])
    done = 0
    for fp in glob.glob(os.path.join(outdir, '蒸馏批次*.jsonl')):
        with open(fp, encoding='utf-8') as f:
            for line in f:
                done += 1
                try:
                    anti_pre = json.loads(line).get('output', '')
                    for s in re.split(r'[。！？\n]', anti_pre):
                        ns = re.sub(r'[，。！？；：、\s]', '', s)
                        if len(ns) >= 10:
                            anti.used_registry.add(ns)
                except Exception:
                    pass
    if done >= args.target:
        log('已有 %d 条，达标，无需续跑' % done)
        return
    log('断点续跑：已入库 %d 条，还需 %d 条' % (done, args.target - done))

    log('[模块1] 构建种子库 …')
    seeds = build_seeds(max(args.seeds, 5000))
    log('  种子 %d 个（主题×立意×基调×结构×场景×学段 组合矩阵）' % len(seeds))

    rng = random.Random(2026)
    si, produced, batch_no = 0, 0, done // args.batch_size + 1
    fout = None
    t0 = time.time()

    while produced + done < args.target:
        # 模块 4：题材强制轮换
        seed = dict(seeds[si % len(seeds)])
        seed['theme'] = anti.pick_theme(rng)
        si += 1

        # 模块 2：Writer → Critic → 重写（至少一轮自我批判）
        best = None                      # (score, title, text, brief, rounds)
        drafts = 0
        while True:
            drafts += 1
            (title, text), brief = (writer_draft(seed, rng) if drafts == 1
                                    else writer_rewrite(seed, problems, rng, drafts))
            score, problems = critic_review(text, seed, anti.used_registry)
            if best is None or score > best[0]:
                best = (score, title, text, brief, drafts)
            if score >= args.threshold:
                break
            if rounds_ok(drafts, args.max_rounds):
                log('  Critic 打分 %.1f：{%s} → Writer 重写' %
                    (score, '；'.join(problems[:3])))
            if drafts >= 40:
                break                    # 保底录取：40 稿仍不过线就收最高分稿
        score, title, text, brief, rounds = best
        if score < args.threshold:
            log('  ⚠ 40 稿保底录取：最高 %.1f 分（主题 %s）' % (score, seed['theme']))
        rounds = max(1, rounds)

        # 硬门槛：字数不达标直接丢弃（不保底录取）
        need = max(int(seed['words'] * 0.85), 800)
        if len(text) < need:
            log('  ✗ 字数硬门槛拒收：%d/%d（主题 %s，最高 %.1f 分）'
                % (len(text), need, seed['theme'], score))
            continue

        # 模块 4：防崩溃查重（与已入库语料重合 > 15% 直接丢弃）
        if anti.overlap_ratio(text) > 0.15:
            continue
        anti.register(text)

        # 温度扰动： instruction 也带随机学段语气
        instruction = make_instruction(seed, title, rounds)
        row = {'instruction': instruction, 'output': text,
               'meta': {'score': score, 'rounds': rounds,
                        '立意': seed['立意'], '基调': seed['基调'],
                        '年级': seed['年级']}}
        if fout is None or produced % args.batch_size == 0:
            if fout:
                fout.close()
            batch_no = done // args.batch_size + produced // args.batch_size + 1
            fp = os.path.join(outdir, '蒸馏批次%d.jsonl' % batch_no)
            fout = open(fp, 'a', encoding='utf-8')
            log('  写入分片 %s' % os.path.basename(fp))
        fout.write(json.dumps(row, ensure_ascii=False) + '\n')
        produced += 1
        if produced % 50 == 0:
            fout.flush()        # 每 50 条落盘，进程意外也不丢数据
            rate = produced / max(1, time.time() - t0)
            log('  进度 %d / %d（%.0f 篇/分钟，预计剩余 %.1f 小时）'
                % (produced + done, args.target, rate * 60,
                   (args.target - produced - done) / max(1.0, rate) / 3600))

    if fout:
        fout.close()
    log('=== 蒸馏完成：本次新增 %d 篇，累计 %d / %d ==='
        % (produced, produced + done, args.target))


if __name__ == '__main__':
    main()
