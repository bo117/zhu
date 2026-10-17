# -*- coding: utf-8 -*-
"""
训练.py —— 竹 · 云端训练脚本（单文件即传即训版）
================================================================
给「国家超算互联网」(scnet.cn) 这类 Slurm 平台用的训练脚本。
整个训练只需要这一个文件：上传 → sbatch 提交 → 自动开训。

它会自动做完七件事（全程打日志，断了能续）：
  [0/7] 定位：先 cd 到本文件所在目录（修"在 home 直接跑找不到模块"的坑）
  [1/7] 装依赖：transformers / peft / bitsandbytes / datasets / accelerate
        / torch / modelscope（清华 pip 源，缺啥装啥，已有的跳过）
  [2/7] 下基座：Qwen2.5-1.5B-Instruct（魔搭 ModelScope 国内直连，
        失败自动退 hf-mirror.com）
  [3/7] 备数据：优先吃同目录下的真实语料（记忆/teaching.jsonl、
        distill/*.jsonl、*_qa.jsonl，q/a 或 instruction/output 字段都认）；
        不够就现场合成「写作指令 → 成品文」训练对（内置素材，零外部依赖）
  [4/7] LoRA 微调，双硬件阵营自适应：
        · NVIDIA 卡 → 4bit QLoRA（bitsandbytes，最省显存）
        · 海光 DCU（国家超算互联网免费额度，DTK/ROCm 环境）
          → 自动跳过 bitsandbytes，fp16 纯 LoRA，单卡显存足够跑 1.5B
        自动断点续训（认得自己上次的 checkpoint-*）；显存不够自动降批
  [5/7] 出产物：output/adapter（LoRA 适配器）+ output/merged（合并完整模型，
        显存不够自动跳过）+ output/训练报告.txt
  [6/7] 打包：把 output/ 压成 zip，平台下载一个包就走

用法（平台侧，两行就完事）：
  1) 把 训练.py 上传到你的工作目录，比如 /public/home/<用户名>/zhu/
  2) 提交作业：sbatch 提交作业.slurm
     （第一次运行会自动生成 提交作业.slurm 模板，改一下里面的目录路径即可）

本地试跑（可选）：python 训练.py --max-samples 200 --epochs 1

所有参数都可以用命令行盖，不传就用默认值——参数给得足，
反正算力是平台的，您只管提交。
"""
import argparse
import glob
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import time
import zipfile

# ---------------------------------------------------------------- [0/7] 定位
# 关键修复：Slurm 作业默认从 home 目录起跑，相对路径全断。
# 这里强制先把工作目录切到本文件所在的项目根目录。
ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

LOG_PATH = os.path.join(ROOT, "训练日志.txt")


def log(msg):
    line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def sh(cmd, check=False):
    """跑 shell 命令，实时回显"""
    log("$ " + cmd)
    return subprocess.run(cmd, shell=True, check=check)


def pip_install(pkgs):
    """清华源装包；装完自验，失败退官方源再试一次"""
    for mirror in ("https://pypi.tuna.tsinghua.edu.cn/simple", None):
        cmd = [sys.executable, "-m", "pip", "install", "-q"] + pkgs
        if mirror:
            cmd += ["-i", mirror]
        r = subprocess.run(cmd)
        if r.returncode == 0:
            return True
    return False


# ---------------------------------------------------------------- 数据合成
# 内置写作素材（精简版）：真实语料缺失时，现场合成训练对，保证空盘也能训
SYN_THEMES = {
    "父爱": ["父亲", "爸爸", "背影", "沉默"],
    "母爱": ["母亲", "妈妈", "唠叨", "饭菜"],
    "友情": ["朋友", "同桌", "少年", "陪伴"],
    "青春": ["十七岁", "操场", "夏天", "毕业"],
    "成长": ["长大", "跌倒", "学会", "明白"],
    "故乡": ["村庄", "老屋", "巷口", "炊烟"],
    "师恩": ["老师", "讲台", "粉笔", "灯光"],
    "梦想": ["远方", "火种", "坚持", "出发"],
    "挫折": ["跌倒", "爬起", "低谷", "重新"],
    "感恩": ["谢谢", "恩情", "记得", "回报"],
    "春天": ["风软", "花开", "细雨", "嫩芽"],
    "思念": ["月光", "旧物", "远行", "牵挂"],
}
SYN_OPEN = ["有些事像茶，当时只觉得苦，回过味来才知道有多香。",
            "人们常说{kw}如山，我家的这座山，沉默得像个哑巴。",
            "每当{kw}两个字被提起，我总会想起那个下午。",
            "故事要从那个再普通不过的清晨说起。",
            "风一软，记忆就藏不住了。"]
SYN_BIT = ["那天关于{kw}的细节，我记到今天，一件都没忘。",
           "很多人只看见结果，没人看见{kw}背后熬过的夜。",
           "{kw}这东西，说出口轻，担起来重，做起来更难。",
           "后来我才懂，所谓{kw}，是把最好的留给别人，把将就留给自己。",
           "日子一天天过去，{kw}变成了习惯，习惯变成了依靠。",
           "有人说岁月很长，其实岁月很短，短到一句再见就散了一场。"]
SYN_DEPTH = ["那一刻我忽然明白，{kw}从来不是嘴上的字，是行动里的人。",
             "原来最深的感情，都藏在最平常的日子里。",
             "这世上的深情，多半不动声色。"]
SYN_CLOSE = ["如今再想起那段日子，心里还是会轻轻一热。",
             "愿往后的岁月，都被温柔以待。",
             "路还长，我带着这些收获，继续往前走。",
             "{kw}两个字，我把它写进题目里，也写进往后的日子里。"]
SYN_GENRE_HOOK = {
    "都市爽文": "被公司开除那天，全部门都在看我的笑话。没人知道，他们挤破头想巴结的那位神秘投资人，昨天刚管我叫了一声'哥'。",
    "末日文": "病毒爆发的第三天，整座城市停了电。楼下的街道从车水马龙变成一片死寂，只用了七十二个小时。",
    "玄幻修仙": "测灵碑前，长老摇头宣布：'灵根全无，此生与仙道无缘。'台下哄笑一片。我默默攥紧袖中那枚温热的古玉。",
    "二次元": "如果这是动漫，我现在应该站在天台上喊出梦想。可惜我只是个普通人——直到那只会说话的黑猫踩着我的作业本开口。",
    "青春校园": "高二开学那天，班主任领进来一个转学生。她站在讲台上，白校服被风吹得贴在身上。我的笔尖，在草稿纸上洇出了一个墨点。",
    "悬疑推理": "雨夜，古董店老板死在了反锁的密室里，钥匙就放在他手边。警方说是自杀，可他右手紧握着的，是一枚不属于他的袖扣。",
    "武侠江湖": "风陵渡口的雪下了三天三夜。她掀帘进店，满座豪客的喧哗齐齐低了下去——那柄剑，二十年前饮过武林盟主的血。",
    "科幻星际": "'曙光号'跃迁出虫洞的那一刻，通讯官突然回头，脸色发白：'舰长，地球……联系不上了。'",
    "甜宠言情": "我把告白短信错发给了死对头。撤回超时的那三秒里，他回复了一条语音：'开门，我在你楼下。'",
    "穿越重生": "被车撞飞的那一刻，我脑子里闪过的是高三那年没敢递出去的情书。再睁眼，高考倒计时牌写着：100天。",
    "无限流": "再睁眼时，我站在纯白广场上，头顶悬浮着一行血字：'欢迎回来，轮回者。'",
    "恐怖灵异": "奶奶临终前叮嘱三件事：夜里听见喊名字别应；床头红绳别解；西屋那口棺材别开。头七那晚，三件事全破了。",
}


def syn_essay(rng, theme):
    """合成一篇记叙文（合成数据用，句式打乱，篇篇不同）"""
    kws = SYN_THEMES[theme]
    kw = rng.choice(kws)
    paras = [rng.choice(SYN_OPEN).format(kw=kw)]
    n = rng.randint(3, 4)
    bits = rng.sample(SYN_BIT, k=min(n, len(SYN_BIT)))
    for i, b in enumerate(bits):
        p = b.format(kw=rng.choice(kws))
        if rng.random() < 0.7:
            p += rng.choice(SYN_DEPTH).format(kw=rng.choice(kws))
        paras.append(p)
    paras.append(rng.choice(SYN_CLOSE).format(kw=kw))
    return "\n\n".join(paras)


def build_synthetic(target):
    """合成「写作指令 → 成品」训练对：作文 + 小说开头 + 改稿三类"""
    rng = random.Random(20260904)
    rows = []
    themes = list(SYN_THEMES.keys())
    genres = list(SYN_GENRE_HOOK.keys())
    i = 0
    while len(rows) < target:
        i += 1
        kind = i % 10
        theme = rng.choice(themes)
        if kind < 6:      # 作文
            words = rng.choice([400, 500, 600, 800])
            text = syn_essay(rng, theme)
            extra = ""
            r = rng.random()
            if r < 0.2:
                extra = "，结尾遗憾一点"
            elif r < 0.35:
                extra = "，要有个好结局"
            elif r < 0.45:
                extra = "，开头用悬念"
            elif r < 0.55:
                extra = "，总分总结构"
            q = "帮我写一篇关于%s的作文，%d字%s" % (theme, words, extra)
            a = "《%s的滋味》\n\n%s" % (theme, text) if rng.random() < 0.5 else text
        elif kind < 8:    # 小说开头
            g = rng.choice(genres)
            q = "写一本%s小说的开头，第一章，800字" % g
            a = "《%s》\n\n第1章\n\n%s\n\n（未完待续——说\"续写\"，我接着往下写。）" % (
                g, SYN_GENRE_HOOK[g])
        else:             # 改稿指令
            text = syn_essay(rng, theme)
            old = rng.choice(["他", "她", "那天", "后来"])
            new = rng.choice(["那位少年", "那位少女", "那个清晨", "许多年后"])
            q = "这是我写的作文，请把「%s」都改成「%s」，其余别动：\n\n%s" % (
                old, new, text)
            a = text.replace(old, new)
        rows.append({"instruction": q, "output": a})
    return rows


def _src_of(fp):
    """给每条语料打来源标签 —— 「作文/小说按比例混合」的分桶依据。

    ⚠️ 精编单独一类且默认排除：抽验 300 条，output 句子唯一率只有 15.6%
    （它出自 3 万篇模板语料的精选，复读病根还在）。混进 60/40 训练
    会把复读病带给 7B。蒸馏/范文那部分才是干净的作文语料。"""
    b = os.path.basename(fp)
    if "外部小说" in b:
        return "novel"
    if "精编" in b:
        return "old"
    if ("范文" in b or "distill" in fp.lower() or "蒸馏" in fp):
        return "essay"
    return "mix"


def collect_real_files(essay_only=False, distill_only=False):
    """收集同目录下能找到的真实语料（q/a、instruction/output、input/target）。
    essay_only=True 时只读作文语料（3万篇作文 + 真题范文），闲聊数据全部跳过。
    distill_only=True 时只吃「教师蒸馏 + 真人范文」的干净语料，3 万篇模板语料
    彻底排除——这是治复读的蒸馏训练模式。
    对策②（骨架分层）：若 骨架分层_训练语料.jsonl 存在（由 骨架分层采样.py 生成），
    作文主料改吃它——骨架分布已拉平，模型不再继承"挤在 3 种骨架"的语料先验。"""
    strat = os.path.join(ROOT, "骨架分层_训练语料.jsonl")
    # 单条样本的字符上限（q+a）。4000 字 ≈ 3000 token，约为 max_len(1536) 的 2 倍；
    # 超过这个量级的样本只会被截成开头一小段，学不到东西却照样占满训练步 ——
    # 36 小时的额度经不起这么烧。整本小说那类几万字的样本就被它挡在外面。
    MAX_PAIR_CHARS = 4000
    downsample = {}      # 文件名 → 每N 条取 1 条；在各分支里被补充
    if distill_only:
        log("  [蒸馏模式] 真人作文 + 外部小说 + distill/*.jsonl + 真人范文，"
            "排除 3 万篇模板语料")
        # 配比原则（2026-10-13 实测定）：
        #   ① 训练语料_真人作文_干净.jsonl —— 6518 篇，**用户可见的主料**。
        #      由 错字分流.py 从 8081 篇原稿分流而来：零高置信错字、
        #      零可疑生僻字、零句级复读，且已删掉 4518 篇末尾的「字数：625」
        #      统计栏。★ 必须用干净集，不能用 训练语料_真人作文.jsonl——
        #      那份是考场原稿，「确认健」「再接再励」会原样被 SFT 学走。
        #   ② distill/*.jsonl —— 商汤蒸馏 + 口语诊断 6400 条；
        #   ③ 训练语料_外部小说.jsonl —— 30 万条纯小说续写，只当"语言底子"，
        #      它 100% 是续写任务、作文类占比 0.43%，配比过高会**负迁移**：
        #      把模型推向"接上文模仿文风"，而作文要求"给主题从零造"。
        #      → 加权下采样到总量的 1/4，其余全给作文类。
        #★ 2026-10-14 重排配比（依据实测，不再拍脑袋）：
        #   · 作文扩增 5717 条（四年级以上真人作文 × 3~4 条不同角度指令）
        #     —— 低年级已剔除76%，一篇不改，只拉指令维度
        #   · 高考真题 86 条（题目+满分作文+阅卷评语，人工审过千万遍）
        #   · 小说提纯（30 万条按文体分层，A 全留 / D 只留 8%）
        #     —— 旧做法把 4997 本只用了 1/60，金子一起被稀释了
        pats = ["训练语料_真人作文_扩增.jsonl",
                "训练语料_真人作文_干净.jsonl",
                "训练语料_高考真题_成文.jsonl",
                "训练语料_高考真题_诊断.jsonl",
                "distill/*.jsonl",
                "作文范文_qa.jsonl", "记忆/作文范文_qa.jsonl",
                "训练语料_小说提纯.jsonl",
                "训练语料_外部小说.jsonl"]
        # 提纯语料已是筛选后的精华，不再下采样；
        # 原始小说降为兜底（只补提纯里没有的部分），每 200 条取 1。
        downsample = {"训练语料_外部小说.jsonl": 200}
        # ★ 干净集缺失 = 硬错误，绝不静默回落到带错字的原稿
        if not os.path.exists(os.path.join(ROOT, "训练语料_真人作文_干净.jsonl")):
            raise SystemExit(
                "✗ 找不到 训练语料_真人作文_干净.jsonl。\n"
                "  它由 错字分流.py 从 训练语料_真人作文.jsonl 生成，"
                "是**去掉错别字之后**的用户可见主料。\n"
                "  先跑：python 错字分流.py\n"
                "  （直接用 训练语料_真人作文.jsonl 会把「确认健」"
                "「再接再励」这类错字教给模型，用户就会看到错别字。）")
    elif essay_only and os.path.exists(strat):
        log("  [对策②] 检测到 骨架分层_训练语料.jsonl —— 作文主料改吃骨架分层语料")
        pats = ["训练语料_外部小说.jsonl", "distill/*.jsonl",
                "骨架分层_训练语料.jsonl", "作文范文_qa.jsonl",
                "记忆/作文范文_qa.jsonl"]
    elif essay_only:
        pats = ["训练语料_外部小说.jsonl", "distill/*.jsonl",
                "作文范文_qa.jsonl", "训练语料_3万.jsonl",
                "记忆/作文范文_qa.jsonl"]
    else:
        pats = ["记忆/teaching.jsonl", "记忆/teaching*.jsonl", "distill/*.jsonl",
                "*_qa.jsonl", "*qa*.jsonl", "数据/*.jsonl", "*.jsonl"]
    seen, rows = set(), []
    dropped_long = 0
    for pat in pats:
        for fp in glob.glob(os.path.join(ROOT, pat)):
            if "output" in fp or "checkpoint" in fp:
                continue
            # 下采样：某些文件任务错配（如 100% 小说续写），只取 1/N，
            # 否则它会把真正该学的作文主料在配比里挤没。
            keep_every = downsample.get(os.path.basename(fp), 1)
            line_no = 0
            try:
                with open(fp, "r", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        line_no += 1
                        if keep_every > 1 and (line_no % keep_every):
                            continue
                        try:
                            obj = json.loads(line)
                        except Exception:
                            continue
                        q = obj.get("q") or obj.get("instruction") or obj.get("question")
                        a = obj.get("a") or obj.get("output") or obj.get("target") \
                            or obj.get("answer")
                        if not (isinstance(q, str) and isinstance(a, str)):
                            continue
                        if len(q) + len(a) > MAX_PAIR_CHARS:
                            dropped_long += 1
                            continue
                        # output 上限从旧的 3000 放宽：分章语料的单章正文常在
                        # 2000~3000 字，旧上限会把它们卡在门外（静默丢数据最难发现）。
                        if 4 < len(q) < 1200 and 4 < len(a):
                            # 用整条指令的哈希去重。
                            # 注意：不能用 q[:40] —— 续写类语料的前 40 字
                            # （"续写一部「言情」小说，接住前文的文风与情节："）完全相同，
                            # 会把几十万条去重成个位数。
                            key = hashlib.md5(q.encode("utf-8")).hexdigest()
                            if key in seen:
                                continue
                            seen.add(key)
                            rows.append({"instruction": q, "output": a,
                                         "src": _src_of(fp)})
            except Exception as e:
                log("  跳过 %s（%s）" % (fp, e))
            if len(rows) > 500000:
                break
    if dropped_long:
        log("  丢掉 %d 条超长样本（>%d 字，截断后学不到东西，只烧算力）"
            % (dropped_long, MAX_PAIR_CHARS))
    return rows


def _mix_ratio(rows, ratio, max_samples):
    """作文/小说按比例分层采样（用户要 50/50 或 60/40）。

    · ratio=0 → 保持原行为：全量洗牌取前 N
    · ratio=0.6 → 作文六成、小说四成
    · 作文语料不足配比时**过采样**补足（LoRA 标准做法：重复干净样本
      远好于混入脏样本）—— 干净的作文语料只有一千多条，不够六成
    · 采样完把 src 标记删掉（别混进训练列）
    · 「精编」桶（复读 15.6%）默认排除，进不了任何一边"""
    by = {"essay": [], "novel": [], "mix": [], "old": []}
    for r in rows:
        by.setdefault(r.get("src", "mix"), []).append(r)
    es = by["essay"] + by["mix"]          # 未标记的杂项按作文处理
    no = by["novel"]
    n = min(max_samples, len(rows))
    if ratio <= 0:
        # ★ ratio=0 必须走「全量洗牌取前 N」（docstring 的原意）。
        #   旧版直接掉进下面的配比分支：n_es=0 → 作文/蒸馏/杂项全被丢光、
        #   只剩小说 —— 历次云端「蒸馏」训练因此全喂的小说（日志铁证：
        #   「语料配比：作文 0 / 小说 19800」）。精编桶（复读病根）照样排除。
        out = es + no
        random.Random(11).shuffle(out)
        out = out[:n]
        log("  语料配比：未指定 --essay-ratio → 全量混洗 %d 条"
            "（作文类 %d / 小说 %d，精编桶已排除）"
            % (len(out),
               sum(1 for r in out if r.get("src") != "novel"),
               sum(1 for r in out if r.get("src") == "novel")))
        return [{k: v for k, v in r.items() if k != "src"} for r in out]
    n_es = int(n * ratio) if (ratio > 0 and es) else 0
    if n_es > len(es) > 0:
        log("  作文语料 %d 条不足配比 %d → 过采样 ×%.1f"
            % (len(es), n_es, n_es / float(len(es))))
    n_no = max(0, min(n - n_es, len(no)))
    random.Random(7).shuffle(es)
    random.Random(9).shuffle(no)
    es_out = (es * (n_es // len(es) + 1))[:n_es] if (n_es > 0 and es) else []
    no_out = no[:n_no]
    log("  语料配比：作文 %d / 小说 %d（目标 %.0f%% 作文）"
        % (len(es_out), len(no_out), ratio * 100))
    out = es_out + no_out
    random.Random(11).shuffle(out)
    return [{k: v for k, v in r.items() if k != "src"} for r in out]


# ---------------------------------------------------------------- 主流程
def _ckpt_mismatch(ckpt_dir, model, lora_r):
    """★ 断点与当前基座的**形状级**校验。返回原因字符串；没问题返回 None。

    为什么必须做形状级校验（2026-10-14 实测踩坑）：
      原来的基座校验只读 adapter_config.json 的 base_model_name_or_path，
      读不到就整段跳过（旧 except: pass）→ 拿**别的基座的 checkpoint** 硬 load，
      报出来的错误是天书：
          copying a param with shape torch.Size([32, 1536])
            to checkpoint, shape torch.Size([1536, 64])
          [8960, 32] vs [1536, 64]
      [8960, 32] 压根不是 Qwen2.5-1.5B 的 vocab=151936 —— 是旧实验残留的断点，
      而自动重试逻辑还会带着同一个断点连炸两次，把 7 小时额度烧光。

    与其信文件名，不如信张量形状。
      LoRA 的形状：(A) = (r, in_features)  (B) = (out_features, r)
      r 与基座的 hidden/vocab 强相关 → 对不上就说明不是同一个模型。
    """
    cfg = os.path.join(ckpt_dir, "adapter_config.json")
    if os.path.exists(cfg):
        try:
            with open(cfg, encoding="utf-8") as f:
                bp = (json.load(f).get("base_model_name_or_path") or "").strip()
            cur = model.split("/")[-1].replace(".", "-")
            if bp and cur not in bp.replace(".", "-"):
                return "基座是 %s，本次要 %s" % (bp.split("/")[-1], cur)
        except Exception:
            pass                      # 读不出来就往下走形状校验
    sd = os.path.join(ckpt_dir, "adapter_model.safetensors")
    if not os.path.exists(sd):
        return None
    try:
        from safetensors.torch import load_file
        t = load_file(sd)
        a = [v.shape for k, v in t.items() if k.endswith("lora_A.weight")]
        b = [v.shape for k, v in t.items() if k.endswith("lora_B.weight")]
        if not a and not b:
            return None
        for sh in a:
            if len(sh) >= 1 and sh[0] != lora_r:
                return "LoRA rank=%s，本次 --lora_r=%s" % (sh[0], lora_r)
        for sh in b:
            if len(sh) >= 2 and sh[1] != lora_r:
                return "LoRA rank=%s，本次 --lora_r=%s" % (sh[1], lora_r)
    except Exception:
        return None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="大量",
                    choices=["标准", "大量", "海量", "魔搭36", "长文", "7B", "CPU", "AMD100"],
                    help="训练规模：标准=轻量试跑，大量=一次吃个饱（默认），"
                         "AMD100=魔搭AMD ROCm(192GB/100h额度)专用档，"
                         "CPU=本机/魔搭CPU实例档（实测 0.34 token/s，一条样本50 分钟，"
                         "只能小步验证、训不出成品），"
                         "魔搭36=按免费额度(36h/月, A10×2)精算过的档，"
                         "长文=max_len 3072 的长上下文档（治「只能接一小段」，吃额度近翻倍），"
                         "7B=配合 --model Qwen/Qwen2.5-7B-Instruct 用（A10 单卡约 20h）")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--init-adapter", default="",
                    help="增量续训：以上一轮 LoRA 成品为起点接着练（已学的不丢），"
                         "例：--init-adapter output_init/7b")
    ap.add_argument("--edit-aware", type=float, default=0.0,
                    help="改动区加权损失的倍数（如 3.0）。配合 改稿语料转换.py 产出的"
                         "【改】标记使用；0=关闭（默认）")
    ap.add_argument("--essay-ratio", type=float, default=0.0,
                    help="作文语料占比（0.5=一半，0.6=六成）；0 = 默认全量混洗")
    ap.add_argument("--epochs", type=int, default=0, help="0 = 按预设来")
    ap.add_argument("--max-samples", type=int, default=0, help="0 = 按预设来")
    ap.add_argument("--batch", type=int, default=0, help="0 = 按预设来；显存不够会自动减半")
    ap.add_argument("--accum", type=int, default=0, help="0 = 按预设来")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=0, help="0 = 按预设来")
    ap.add_argument("--max-len", type=int, default=0, help="0 = 按预设来")
    ap.add_argument("--synthetic", type=int, default=150000,
                    help="真实语料不足时，合成多少条写作训练对")
    ap.add_argument("--essay-only", dest="essay_only",
                    action="store_true", default=True,
                    help="（默认）纯作文模式：只训 3万篇千字作文 + 真题范文")
    ap.add_argument("--chat-mix", dest="essay_only", action="store_false",
                    help="混入聊天语料一起训（默认关闭）")
    ap.add_argument("--distill-only", dest="distill_only",
                    action="store_true", default=False,
                    help="蒸馏训练：只吃 distill/*.jsonl + 真人范文，排除 3 万篇模板语料")
    ap.add_argument("--out", default="output")
    ap.add_argument("--save-steps", type=int, default=400,
                    help="每多少优化步存一次 checkpoint（长跑保险）")
    ap.add_argument("--eval-steps", type=int, default=400)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--max-hours", type=float, default=0.0,
                    help="时间预算（小时），0=不限。到点自动优雅停机并存盘，"
                         "给平台额度留余量。魔搭 36h/月 的额度建议填 30。")
    ap.add_argument("--no-merge", action="store_true",
                    help="训练完不合并完整模型（省时间/省磁盘，adapter 一样能用）")
    args = ap.parse_args()

    # ★ HF 镜像兜底：国内直连 huggingface.co 不通，权重会下不下来。
    #   魔搭和 hf-mirror 都实测可达；只在HF_ENDPOINT 未被显式设置时才接管。
    #   （口径：本机 urllib 探测 huggingface.co = URLError 不通。）
    if not os.environ.get("HF_ENDPOINT") and not os.environ.get("HF_HUB_OFFLINE"):
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")

    # ★ AMD ROCm 路线（魔搭 AMD GPU 环境，ROCm 7.2.3/ 8核 / 192GB / 100h）：
    #   本脚本本来就支持 —— IS_DCU 判 torch.version.hip，会自动跳过 bitsandbytes
    #   走全精度 fp16/bf16 LoRA。这里补一块**开训前体检**，别跑几小时才发现坑。
    if args.preset == "AMD100":
        import torch as _t
        log("=" * 68)
        if getattr(_t.version, "hip", None) is None:
            log("!! 你选了 AMD100，但当前不是 ROCm 环境"
                "（torch.version.hip=None）。")
            log("   → AMD100 是给魔搭 AMD GPU 环境用的；"
                "如果实际是 CPU，改用 --preset CPU。")
        elif not _t.cuda.is_available():
            log("!! ROCm 装了但 cuda 不可用 —— 十有八九是镜像与卡不匹配。")
            log("   排查：rocm-smi 看得到卡吗？"
                "PYTORCH_ROCM_ARCH 要与你的卡架构对上。")
        else:
            _n = _t.cuda.device_count()
            try:
                _cap = _t.cuda.get_device_properties(0).total_memory / 1073741824
            except Exception:
                _cap = 0
            log("【AMD/ROCm 体检】卡数 %d，单卡显存 %.0fGB，torch %s"
                % (_n, _cap, _t.__version__))
            if _n > 1:
                log("   多卡：单进程会自动吃掉全部卡；"
                    "要用多卡请手动加 HIP_VISIBLE_DEVICES 后再跑。")
            log("   本档：batch 16 / LoRA r=64 / max_len 2048 / 20 万条 × 2 轮。")
            log("   全精度 fp16/bf16 LoRA（不走 4bit —— ROCm 上 bitsandbytes 常缺）。")
        log("=" * 68)

    # CPU 路线的诚实告警：实测 0.34 token/s，跑量之前先让用户看见代价。
    if args.preset == "CPU":
        log("=" * 68)
        log("【CPU 档】实测口径（8 线程 / 32G / torch cpu / Qwen2.5-1.5B 4bit QLoRA）：")
        log("   矩阵基准 3.1 GFLOP/s → 训练约 0.34 token/s")
        log("   一条 1000-token 样本≈ 50 分钟；2000 条 ≈ 69 天。")
        log("   → 这条命令**只能小步验证**（格式学不学得会 / 标签对不对 / 配比有没有坑），")
        log("     训不出成品。成品必须在带 GPU 的实例上训。")
        log("=" * 68)

    # 规模预设：参数不用您操心，平台算力顶上；单卡显存再小也会自动降档续训
    PRESETS = {
        #           samples   epochs  batch  accum  lora_r  max_len
        "标准": (200_000,   2,      4,     4,     16,     1024),
        "大量": (600_000,   3,      4,     8,     32,     1536),
        "海量": (1_500_000, 3,      4,     16,    64,     2048),
        # 「魔搭36」：按魔搭免费额度的**单次会话**精算的档（单卡 A10，一次约 7 小时）。
        # 口径：一条续写样本 ≈ 1000 token（前文 350 字 + 正文 550 字 + 提示）。
        #   1.5B 4bit QLoRA + 梯度检查点，单卡 A10 ≈ 2.5e3 token/s
        #   → 12 万条跑一轮 ≈ 13 小时，正好跨 2 次会话；双卡则约 6.5 小时/轮。
        #
        # 为什么从 30 万降到 12 万：
        #   ① tokenize 出来的 Arrow 表要驻留内存 —— 30 万条 ≈ 5GB
        #      （input_ids + labels 两列 × ~1000 token × 8 字节），叠加 map 期间的
        #      临时副本很容易把实例内存打爆（实测线上就是 tokenize 到 22% 崩的）；
        #   ② 30 万条一轮要 34 小时，36 小时/月的额度一个月都跑不完一轮，
        #      等于永远在"半轮"状态，反而不如把一轮缩到能在额度内跑完。
        "魔搭36": (120_000, 2,     4,     8,     32,     1536),
        # 「CPU」：真按**本机/魔搭 CPU 实例**标定的档，不是 A10 的数字换个名字。
        #★ 实测标定（本机 Intel 8 线程32G、torch 2.14.0+cpu、1.5B 4bit QLoRA）：
        #   矩阵乘基准 3.1 GFLOP/s（torch 默认只用 2 线程）
        #   训练 FLOPs ≈ 6·N·T→ **0.34 token/s**
        #   → 一条 1000 token 样本要**约 50 分钟**；一小时跑 1 条。
        #   换算：2000 条要 1650 小时 ≈ **69 天**。
        #   ★ 结论必须说死：**这台机器上不可能靠堆量训出成品**，
        #     哪怕不吃内存，时间上也不成立（A10 是它的约 7500倍）。
        #     CPU 这条路只能做一件事：**小步验证**（几百条以内，
        #     看格式学不学得会、标签对不对、配比有没有坑），确认无误再上云。
        #   用法：python 训练.py --preset CPU --distill-only --max-hours 8 --max-samples 10
        "CPU": (200,1,      1,     8,     16,     1024),
        # 「长文」：max_len 1536 → 3072，配 外部小说入库.py --ctx 900 --outlen 1800
        # 切出来的长续写对 —— 治「模型只学过 900 字，接不住长前文」。
        # 这是唯一能真正抬升「长上下文能力」的一步：光在提示词侧调预算只是把话说全，
        # 模型自己没见过 >1536 token 的样本，长程一致性就只能靠规则硬兜。
        # 代价：单条样本 token 约翻倍 → 同条数耗时 ≈ 2 倍；36h 额度装不下 12 万条，
        # 所以 samples 从 12 万降到 6 万（约 1 轮跑完）。
        # 用法：
        #   python 外部小说入库.py --ctx 900 --outlen 1800 --out 训练语料_长续写.jsonl
        #   python 训练.py --preset 长文 --max-hours 30
        "长文": (60_000, 2,      4,     8,     32,     3072),
        # 「7B」：换上 Qwen2.5-7B 基座时用（写作质量能上一个台阶）。
        # 4bit QLoRA 在 A10 24G 上：7B 量化后约 4G，加 LoRA/优化器/激活
        # （batch 2 + 梯度检查点）≈ 12~14G —— 放得下。
        # 速度约是 1.5B 的 1/3，所以 samples 砍到 6 万、只跑 1 轮：
        #   单卡 A10 约 20 小时 / 双卡约 10 小时（都在 36h 额度内）。
        # 用法：python 训练.py --model Qwen/Qwen2.5-7B-Instruct --preset 7B --max-hours 30
        "7B": (60_000, 1,      2,     8,     32,     1536),
        # 「AMD100」：按魔搭 **AMD ROCm 7.2.3 / 8核 / 192GB 显存 / 100小时额度** 定的档。
        #   为什么不用「魔搭36」：那一档是给 A10 24G 精算的（batch 4 + 4bit 量化）。
        #   在 192GB 上照抄它，等于把一张能跑 1.5B 全参的卡当 24G 用 —— 浪费 87% 显存，
        #   而且 4bit 量化在 ROCm 上还要额外开销，反而更慢。
        #   ROCm 上 bitsandbytes 常缺失 → 本脚本已自动走**全精度 fp16/bf16 LoRA**
        #   （IS_DCU 判定 torch.version.hip），显存够，无需量化。
        #   取舍：
        #     batch 4→ 16（192GB 撑得住）、LoRA r 32 → 64（容量给足，学指令更牢）
        #     max_len 1536 → 2048（你的痛点之一是接不住上下文）
        #     样本 12 万 → 20万，epochs 2（100 小时额度比 36 小时宽裕得多）
        #   用法：
        #     python 训练.py --preset AMD100 --distill-only --max-hours 90
        #     单卡（A10 那套 torchrun 写法在这里不要用 —— ROCm 多卡需另配 HIP_VISIBLE_DEVICES）
        "AMD100": (200_000, 2,   16,    4,     64,     2048),
    }
    p_samples, p_epochs, p_batch, p_accum, p_r, p_len = PRESETS[args.preset]
    if not args.max_samples:
        args.max_samples = p_samples
    if not args.epochs:
        args.epochs = p_epochs
    if not args.batch:
        args.batch = p_batch
    if not args.accum:
        args.accum = p_accum
    if not args.lora_r:
        args.lora_r = p_r
    if not args.max_len:
        args.max_len = p_len

    log("=== 竹 · 云端训练开始 ===")
    log("项目根目录：%s（已 cd，相对路径稳了）" % ROOT)
    os.makedirs(args.out, exist_ok=True)

    # ---- [1/7] 依赖 ----
    log("[1/7] 检查并安装依赖（transformers / peft / datasets / accelerate ...）")
    try:
        import torch
    except ImportError:
        log("!! 这个节点没有预装 PyTorch。海光 DCU 请先加载平台镜像里的")
        log("   DTK/PyTorch 环境（module load 或选对应镜像），再重新提交作业；")
        log("   脚本不会自动 pip 装 torch——装错 CUDA 版在 DCU 上是跑不起来的。")
        raise
    # CPU 线程：默认吃满所有核（DDP 多进程时按进程数均分），TRAIN_THREADS 可手动盖。
    # ★ 必须在 import torch 之后调 —— 旧版在 import 前就调 set_num_threads(2)，
    #   NameError 被 except 吞掉，从来没生效过（好在没生效：CPU 训练不会被压到 2 核）。
    try:
        _ws = int(os.environ.get("WORLD_SIZE") or 1)
        torch.set_num_threads(int(os.environ.get("TRAIN_THREADS")
                                  or max(1, (os.cpu_count() or 2) // _ws)))
        log("  CPU 线程数：%d（TRAIN_THREADS 环境变量可盖）"
            % torch.get_num_threads())
    except Exception:
        pass
    # 硬件阵营识别：NVIDIA 走 CUDA+4bit 量化；海光 DCU 走 DTK/ROCm，fp16 直接训
    IS_DCU = getattr(torch.version, "hip", None) is not None
    IS_CPU = not torch.cuda.is_available()
    log("  硬件阵营：%s（torch=%s）"
        % ("海光 DCU / ROCm" if IS_DCU else
           ("CPU" if IS_CPU else "NVIDIA CUDA"), torch.__version__))

    # ---- 双卡 / 多卡（DDP）环境探测 ----
    # torchrun --nproc_per_node=2 训练.py 会把 LOCAL_RANK / WORLD_SIZE 注进环境变量；
    # 直接 python 训练.py（单卡）时这俩不存在 → 自动退化成单进程，行为与以前一致。
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK") or 0)
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE") or 1)
    IS_MAIN = (LOCAL_RANK == 0)
    if WORLD_SIZE > 1:
        log("  分布式：WORLD_SIZE=%d，本进程 LOCAL_RANK=%d（每卡一进程，DDP）"
            % (WORLD_SIZE, LOCAL_RANK))
    else:
        log("  单进程模式（要开双卡：torchrun --nproc_per_node=2 训练.py ...）")

    need = {}
    for mod, pkg in [("transformers", "transformers"),
                     ("peft", "peft"), ("datasets", "datasets"),
                     ("accelerate", "accelerate"),
                     ("modelscope", "modelscope")]:
        try:
            __import__(mod)
            need[pkg] = False
        except Exception:
            need[pkg] = True
    if not IS_DCU:
        try:
            import bitsandbytes  # noqa
            need["bitsandbytes"] = False
        except Exception:
            need["bitsandbytes"] = True   # 仅 NVIDIA 需要 4bit 量化库
    todo = [p for p, ok in need.items() if ok]
    if todo:
        log("  缺这些：%s → 清华源开装" % ", ".join(todo))
        pip_install(todo)
    else:
        log("  依赖齐活，跳过安装")
    import transformers
    import peft
    from datasets import Dataset
    log("  版本：torch=%s transformers=%s peft=%s cuda=%s hip=%s" % (
        torch.__version__, transformers.__version__, peft.__version__,
        torch.version.cuda, getattr(torch.version, "hip", None)))
    if not torch.cuda.is_available():
        log("  !! 没检测到 GPU——平台作业请确认分区给了 gres=gpu:1。"
            "继续用 CPU 跑（会非常慢，仅作流程验证）。")
    n_gpu = torch.cuda.device_count()
    if n_gpu:
        log("  GPU：%s × %d" % (torch.cuda.get_device_name(0), n_gpu))

    # ---- [2/7] 基座模型 ----
    # ★★ 缓存目录必须带模型名！
    #    原来固定用 models/base —— 只要那个目录里有 config.json 就直接复用，
    #    于是「--model Qwen/Qwen2.5-7B-Instruct」会被**静默忽略**，
    #    拿旧的 1.5B 训一整轮（20 小时后才发现白跑）。实测踩过。
    _tag = args.model.replace("/", "--")
    model_dir = os.path.join(ROOT, "models", _tag)
    _legacy = os.path.join(ROOT, "models", "base")
    if os.path.exists(os.path.join(model_dir, "config.json")):
        log("[2/7] 基座已下载过，直接用：%s" % model_dir)
    elif args.model.endswith("Qwen2.5-1.5B-Instruct") \
            and os.path.exists(os.path.join(_legacy, "config.json")):
        model_dir = _legacy                      # 兼容老目录（默认 1.5B）
        log("[2/7] 复用已有基座：%s" % model_dir)
    else:
        log("[2/7] 下载基座模型 %s（魔搭直连，失败退 hf-mirror）" % args.model)
        try:
            from modelscope import snapshot_download
            p = snapshot_download(args.model,
                                  cache_dir=os.path.join(ROOT, "models"))
            model_dir = p
            log("  魔搭下载完成：%s" % p)
        except Exception as e:
            log("  魔搭失败（%s）→ 退 hf-mirror.com" % e)
            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
            from huggingface_hub import snapshot_download as hf_dl
            model_dir = hf_dl(args.model,
                              cache_dir=os.path.join(ROOT, "models"))
    log("  → 本次实际使用的基座：%s" % model_dir)

    # ---- [3/7] 数据 ----
    # ★ 文案必须和实际吃的语料一致 —— essay_only 分支的 pats 里**明确包含
    #   外部小说 30 万条**，叫「纯作文模式」纯属误导（用户会以为在只练作文）
    log("[3/7] 备数据%s"
        % ("（作文+小说混合：外部小说为主料，作文范文+蒸馏语料为辅料）"
           if args.essay_only else "（闲聊教学模式：teaching+全部 jsonl 通配）"))
    rows = collect_real_files(args.essay_only, args.distill_only)
    log("  真实语料捞到 %d 条" % len(rows))
    if len(rows) < args.max_samples and not args.essay_only:
        n_syn = min(args.synthetic, args.max_samples - len(rows))
        if n_syn > 0:
            log("  合成 %d 条写作训练对（指令→成品）" % n_syn)
            rows += build_synthetic(n_syn)
    rows = _mix_ratio(rows, args.essay_ratio, args.max_samples)
    # 验证集切分：绝不能把训练集切空。
    # 旧写法 n_eval = max(50, 1%) 在样本数 ≤ 50 时会让 train_rows = []，
    # 接着 Dataset.from_list([]) 一个列都没有 → remove_columns 直接炸，
    # 报出来还是 "Column name [...] not in the dataset"，跟真因八竿子打不着。
    # 顺带把验证集封顶 500 条：3000 条验证在双卡上每次要 1~2 分钟，
    # 一个 epoch 十几轮验证就是好几个小时 —— 36 小时的额度要花在训练上。
    n_eval = max(1, min(int(len(rows) * 0.01), 500, len(rows) // 10))
    if len(rows) - n_eval < 1:
        n_eval = 0
    train_rows, eval_rows = rows[n_eval:], rows[:n_eval]
    if not train_rows:
        log("!! 样本太少（只有 %d 条），没法切出训练集。加大 --max-samples。"
            % len(rows))
        return
    log("  训练 %d 条 / 验证 %d 条" % (len(train_rows), len(eval_rows)))

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def _mask_prompt(msgs):
        """把 system + user 部分打上 -100，只对「助手回复」算 loss。

        ★ 这是旧版漏掉的一步，也是治复读的关键一步。
        旧版直接 `labels = input_ids`（不屏蔽提示词），后果有两层：
          ① 大约一半 token 是提示词（system + 前文 350 字），模型的梯度
             有一半花在「背指令模板和前文」上 —— 真正要学的续写只吃到一半信号；
          ② 更糟的是，它等于在训练模型**照抄输入** —— 而"照抄"正是竹要治的复读病。
        所以这里必须 mask：只有 output（后续 550 字）参与 loss。"""
        full = tok.apply_chat_template(msgs, tokenize=False)
        ids = tok(full, truncation=True, max_length=args.max_len)
        prompt = tok.apply_chat_template(msgs[:-1], tokenize=False,
                                         add_generation_prompt=True)
        n = len(tok(prompt, truncation=True,
                    max_length=args.max_len)["input_ids"])
        labels = list(ids["input_ids"])
        for i in range(min(n, len(labels))):
            labels[i] = -100
        ids["labels"] = labels
        if getattr(args, "edit_aware", 0) > 0:
            # ★ 改动区加权：output 里 【改】…【/改】 区间 token 权重 ×N，其余 ×0.3。
            #   目的：让梯度集中在「怎么改」上，而不是「把原文抄一遍」。
            enc = tok(full, truncation=True, max_length=args.max_len,
                      return_offsets_mapping=True)
            offs = enc["offset_mapping"]
            spans, p = [], 0
            while True:
                a = full.find("【改】", p)
                if a < 0:
                    break
                b = full.find("【/改】", a)
                if b < 0:
                    break
                spans.append((a, b + len("【/改】")))   # 含标记本身
                p = b + len("【/改】")
            w = []
            for i, (st, en) in enumerate(offs):
                if i < n:
                    w.append(0.0)
                elif any(a <= st < b or a <= en <= b for a, b in spans):
                    w.append(float(args.edit_aware))
                else:
                    w.append(0.3)
            if len(w) == len(ids["input_ids"]):
                ids["token_weights"] = w
        return ids

    def encode(row):
        msgs = [{"role": "system",
                 "content": "你是写作助手「竹」，中文写作能力强，服从用户的开头、"
                            "结尾、结构、字数等要求，改稿时只动用户指明的地方。"},
                {"role": "user", "content": row["instruction"]},
                {"role": "assistant", "content": row["output"]}]
        return _mask_prompt(msgs)

    # num_proc=None（不是 1！）：1 也会开 multiprocess 子进程。
    # 而本进程此前已调过 torch.cuda.device_count()，CUDA 上下文已初始化 ——
    # CUDA 初始化后再 fork，子进程会直接死掉，父进程收到莫名奇妙的 EOFError。
    # 单进程 map 稍慢一点，但稳（而且在 GPU 机上内存也更省）。
    ds_train = Dataset.from_list(train_rows).map(encode, remove_columns=[
        "instruction", "output"], num_proc=None, desc="tokenize-train")
    ds_eval = Dataset.from_list(eval_rows).map(encode, remove_columns=[
        "instruction", "output"], num_proc=None, desc="tokenize-eval")

    # ---- [4/7] LoRA（NVIDIA 走 4bit QLoRA；海光 DCU/CPU 走全精度 LoRA）----
    log("[4/7] 装载基座 + 挂 LoRA")
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    if torch.cuda.is_available():
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() \
            else torch.float16
    else:
        compute_dtype = torch.float32          # CPU 训练用 fp32，稳
    use_quant = (not IS_DCU) and torch.cuda.is_available()
    amp_on = torch.cuda.is_available()
    if use_quant:
        log("  NVIDIA 卡 → 4bit QLoRA（省显存）")
        bnb = BitsAndBytesConfig(load_in_4bit=True,
                                 bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=compute_dtype,
                                 bnb_4bit_use_double_quant=True)
        # 多卡关键点：4bit + device_map="auto" 在 DDP 下会出事
        # （每个进程都想自己独占所有卡，两个进程抢同一张卡 → 显存爆 / NCCL 崩）。
        # 双卡时把本进程的权重钉死在自己的 rank 上：{"": LOCAL_RANK}。
        dev_map = {"": LOCAL_RANK} if WORLD_SIZE > 1 else "auto"
        if WORLD_SIZE > 1:
            log("  4bit 权重钉在 cuda:%d（每卡各持一份完整副本，不分片）"
                % LOCAL_RANK)
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, quantization_config=bnb,
            torch_dtype=compute_dtype, device_map=dev_map,
            trust_remote_code=True)
    else:
        why = "海光 DCU（DTK 环境不带 bitsandbytes）" if IS_DCU else "无 NVIDIA GPU"
        log("  %s → 跳过 4bit 量化与 device_map（CPU 直载更稳）" % why)
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, torch_dtype=compute_dtype,
            trust_remote_code=True)
    model.config.use_cache = False
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    if use_quant:
        model = prepare_model_for_kbit_training(model)
    # ★ MoE 兼容（2026-10-14）：MoE 基座（如 Qwen3-30B-A3B）的 FFN 层
    #   **不叫** gate_proj/up_proj/down_proj —— 那是稠密模型 MLP 的命名。
    #   MoE 里是 mlp.experts.{i}.gate_proj 之类，一层 128 个专家；
    #   peft 按名字精确匹配，找不到就抛 KeyError（实测在
    #   tuners_utils.inject_adapter 里直接炸，白等一小时下载）。
    #   所以：**只挂 attention 的四个投影**（MoE 和稠密都有，名字一致），
    #   绝不去碰专家层 —— 对 128 个专家都挂 LoRA 也不可能。
    #   若要训 MoE 的 FFN，正确做法是挂 mlp.gate（router）而不是 experts。
    _model_name = (args.model or "").lower()
    _is_moe = ("moe" in _model_name
               or "a3b" in _model_name
               or "a6b" in _model_name
               or "30b" in _model_name or "235b" in _model_name)
    if _is_moe:
        lora_targets = ["q_proj", "k_proj", "v_proj", "o_proj"]
        log("  [MoE] 基座 %s 疑似MoE：LoRA 只挂 attention"
            "（不挂 experts，否则 128 个专家一层都匹配不上）"
            % args.model.split("/")[-1])
    else:
        lora_targets = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]
    lora = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM",
        target_modules=lora_targets)
    if args.init_adapter:
        # ★ 增量续训：以上一轮的成品 LoRA 为起点接着练（而不是随机初始化）——
        #   换语料配比重训时，之前学到的能力不会丢（用户点名要求保护已训成果）
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.init_adapter,
                                          is_trainable=True)
        log("  ✓ 增量续训：从上一轮成品接着练（%s）" % args.init_adapter)
    else:
        model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable()
    try:
        model.enable_input_require_grads()
    except Exception:
        pass

    from transformers import (Trainer, TrainingArguments,
                              DataCollatorForSeq2Seq, TrainerCallback)
    class _EditCollator(DataCollatorForSeq2Seq):
        """把每个样本的 token_weights 也 pad 成 batch（改动区加权用）"""

        def __call__(self, features):
            ws = [f.pop("token_weights", None) for f in features]
            batch = super().__call__(features)
            if ws and ws[0] is not None:
                mx = batch["input_ids"].shape[1]
                pad = torch.zeros((len(ws), mx), dtype=torch.float)
                for i, w in enumerate(ws):
                    if w:
                        pad[i, :min(len(w), mx)] = torch.tensor(w)[:mx]
                batch["token_weights"] = pad
            return batch

    class _WeightedTrainer(Trainer):
        """改动区加权交叉熵：loss = Σ(w·ce) / Σw（w=0 的位置不参与）"""

        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            w = inputs.pop("token_weights", None)
            outputs = model(**inputs)
            if w is None or inputs.get("labels") is None:
                return (outputs.loss, outputs) if return_outputs else outputs.loss
            logits = outputs.logits[:, :-1, :].contiguous()
            labels = inputs["labels"][:, 1:].contiguous()
            ww = w[:, 1:].contiguous()
            lf = torch.nn.CrossEntropyLoss(reduction="none")
            # ★ 必须先转 float32 再算 CE —— 4bit QLoRA 下 logits 是 bf16，
            #   直接算交叉熵会因精度不足导致 loss 虚高、梯度噪声大
            #   （实测：漏了 .float() 时 train_loss 7.45 / grad_norm 2.2，
            #    而正常应 ~1.1 / ~0.3 → HF 官方 Trainer 也是先转 float 的）
            ls = lf(logits.view(-1, logits.size(-1)).float(), labels.view(-1))
            wf = ww.view(-1) * (labels.view(-1) != -100).float()
            loss = (ls * wf).sum() / wf.sum().clamp(min=1.0)
            return (loss, outputs) if return_outputs else loss

    collator = _EditCollator(tok, padding=True, label_pad_token_id=-100)
    TrainerCls = _WeightedTrainer if getattr(args, "edit_aware", 0) > 0 else Trainer
    if getattr(args, "edit_aware", 0) > 0:
        log("  [改动区加权] 已启用：改动区 ×%.1f / 未改区 ×0.3 / 提示区 ×0"
            % args.edit_aware)

    class TimeGuard(TrainerCallback):
        """时间预算守卫：到点让 Trainer 优雅停机（正常存盘、正常走产物流程）。

        为什么必须有它：魔搭免费 GPU 是「36 小时/月」的硬额度，跑超了平台会直接掐实例，
        而且是掐在没存盘的那一刻——进度丢了、额度也白烧了。有了它，到点主动收工，
        剩下的额度留给下一轮续训。"""

        def __init__(self, deadline):
            self.deadline = deadline

        def on_step_end(self, a, state, control, **kw):
            if self.deadline and time.time() >= self.deadline:
                if getattr(state, "is_world_process_zero", True):
                    log("  ⏰ 时间预算到点（--max-hours）→ 优雅停机，存盘收工")
                    log("     想接着练：同一目录重跑同一条命令即可（自动断点续训）")
                if control is not None:
                    control.should_training_stop = True
            return control

    def build_trainer(batch, accum, train_ds, eval_ds):
        import inspect
        want = dict(
            output_dir=args.out,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=batch,
            per_device_eval_batch_size=max(1, batch),
            gradient_accumulation_steps=accum,
            learning_rate=args.lr,
            warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        # CPU 上一个优化步要数分钟：logging_steps 太大 → 几小时看不到一行 loss。
        logging_steps=5,
        save_steps=args.save_steps,
        save_total_limit=3,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        bf16=amp_on and compute_dtype == torch.bfloat16,
        fp16=amp_on and compute_dtype == torch.float16,
        pin_memory=False,
        gradient_checkpointing=True,
            report_to=[],
            dataloader_num_workers=0,
                remove_unused_columns=False,
        )
        if WORLD_SIZE > 1:
            # LoRA 只训少量参数：DDP 默认的"找未使用参数"会把没进计算图的 LoRA
            # 参数误判成未使用 → 报错或直接挂死，必须关掉。
            want["ddp_find_unused_parameters"] = False
            want["ddp_backend"] = "nccl"
        sig = inspect.signature(TrainingArguments.__init__).parameters
        dropped = [k for k in list(want) if k not in sig]
        for k in dropped:
            want.pop(k)
        if dropped:
            log("  本版 transformers 不认这些参数（自动忽略）：%s" % ",".join(dropped))
        targs = TrainingArguments(**want)
        return TrainerCls(model=model, args=targs, train_dataset=train_ds,
                          eval_dataset=eval_ds, data_collator=collator,
                          callbacks=[TimeGuard(DEADLINE)])

    # 断点续训：找到最近的 checkpoint-*
    def latest_ckpt():
        cks = sorted(glob.glob(os.path.join(args.out, "checkpoint-*")),
                     key=lambda p: int(p.rsplit("-", 1)[-1])
                     if p.rsplit("-", 1)[-1].isdigit() else -1)
        return cks[-1] if cks else None

    # ---- [5/7] 开训（带两重保险：显存不够自动降批；崩了自动从断点重来）----
    log("[5/7] 开训（preset=%s, epochs=%d, batch=%d×%d, lora_r=%d, len=%d）"
        % (args.preset, args.epochs, args.batch, args.accum,
           args.lora_r, args.max_len))
    t0 = time.time()

    # ---- 时间预算 ----
    # 语义：--max-hours = 【本次运行】最多跑几小时，不是月额度。
    # 为什么按"每次运行"算：平台给的是「单次实例剩余时长」（魔搭 Notebook 常见 8 小时），
    # 到点实例被直接回收 —— 所以每次开实例都该给一个**小于剩余时长**的预算，
    # 让它到点主动优雅停机 + 存盘，而不是被硬杀。
    # 已训总时长另记一份（output/累计训练时长.txt），用来盯 36 小时/月 的额度。
    cum_file = os.path.join(args.out, "累计训练时长.txt")
    try:
        cum_before = float(open(cum_file, encoding="utf-8").read().strip())
    except Exception:
        cum_before = 0.0
    if args.max_hours > 0:
        DEADLINE = t0 + args.max_hours * 3600
        log("  本次运行时间预算：%.1f 小时（到点自动优雅停机存盘）" % args.max_hours)
    else:
        DEADLINE = None
        log("  没设 --max-hours：本次不设时间上限（平台掐实例时靠 checkpoint 兜底）")
    if cum_before > 0:
        log("  历史累计已训：%.1f 小时（本机记账；月额度 36h 自己盯着）" % cum_before)

    train_ds, eval_ds = ds_train, ds_eval
    best_batch, best_accum = args.batch, args.accum
    crashed = 0
    trainer = None
    while True:
        # ★★ 用 --init-adapter 时**禁止续旧断点**：
        #   旧断点是上一轮的中间状态，续它等于用旧 checkpoint 覆盖刚加载的成品；
        #   而且 Trainer 按旧步数（如 800）判定当前计划（186 步）已训完，直接退出 ——
        #   实测：0.51 分钟「训练完成」、0 步、产物被污染。新计划必须从 0 开始。
        resume = None if args.init_adapter else latest_ckpt()
        if resume:
            # ★★ 校验断点是不是同一个基座训出来的 ——
            #    1.5B 的旧断点接到 7B 上会直接 size mismatch 崩（实测踩过：
            #    换 7B 重训时 output/checkpoint-2800 还是 1.5B 的旧权重，
            #    加载即炸，而且自动重试逻辑会带着同一个断点连炸两次）。
            try:
                with open(os.path.join(resume, "adapter_config.json"),
                          encoding="utf-8") as f:
                    _bp = (json.load(f).get("base_model_name_or_path")
                           or "").strip()
                # ★ 断点的基座路径常是 .../Qwen--Qwen2.5-7B-Instruct/snapshots/master ——
                #   取 basename 只会得到 "master"（魔搭缓存目录名），于是把**同一个 7B**
                #   误判成「不同基座」、把正确断点挪走（2026-09-21 实测踩到）。
                #   改成判断模型名是否出现在路径里。
                _cur_name = args.model.split("/")[-1].replace(".", "-")
                _bp_norm = _bp.replace(".", "-")
                _bp_tail = _bp.split("/")[-1] if _bp else ""
                # ★★★ 2026-10-14 形状级校验（详见 _ckpt_mismatch 的 docstring）
                _bad = _ckpt_mismatch(resume, args.model, args.lora_r)
                if _bad:
                    _bak = os.path.join(ROOT, "output_旧断点_%s" % os.path.basename(resume))
                    os.makedirs(_bak, exist_ok=True)
                    try:
                        import shutil
                        shutil.move(resume, os.path.join(_bak, os.path.basename(resume)))
                    except Exception:
                        resume = None
                    log("⚠️ 断点 %s 与当前模型不匹配：%s" % (os.path.basename(resume), _bad))
                    log("   → 已挪到 %s，从头训（硬接必炸，且自动重试会带着它连炸两次）" % _bak)
                    resume = None
            except Exception:
                pass
            if resume:
                log("  发现断点 %s → 接着练（不重复花钱）" % resume)
        trainer = build_trainer(best_batch, best_accum, train_ds, eval_ds)
        try:
            trainer.train(resume_from_checkpoint=resume)
            break
        except RuntimeError as e:
            # ★ CPU 分配器报的是 "DefaultCPUAllocator: can't allocate memory"，
            #   不含 "out of memory" —— 旧版只认 CUDA 那句，CPU 上 OOM 永远
            #   识别不出来，自动降批保险丝形同虚设（连崩 3 次硬退）。
            _emsg = str(e).lower()
            oom = ("out of memory" in _emsg or "can't allocate memory" in _emsg
                   or "cannot allocate memory" in _emsg
                   or "defaultcpuallocator" in _emsg)
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            if oom and WORLD_SIZE > 1:
                # 双卡下绝不在进程内偷偷降批：两个进程会跑偏 → 死锁挂死。
                # 直接退出，让你用更小的批重跑（断点续训，不白烧）。
                log("  !! 双卡下爆显存：不在进程内自动降批（两进程会失步挂死）。")
                log("     请照这条重跑，会自动从断点接着练：")
                log("       torchrun --nproc_per_node=%d 训练.py ... --batch %d"
                    % (WORLD_SIZE, max(1, best_batch // 2)))
                raise
            if not oom:
                crashed += 1
                if crashed >= 3:
                    raise
                log("  训练中断（%s）→ 第 %d 次自动从断点重来" % (e, crashed))
                time.sleep(10)
                continue
            # 显存不够：先减半 batch，见底了再砍序列长度
            if best_batch > 1:
                best_batch = max(1, best_batch // 2)
                log("  !! 显存不够 → batch 降到 %d，从断点接着练" % best_batch)
            else:
                new_len = max(256, int(args.max_len * 0.7))
                if new_len == args.max_len:
                    raise
                args.max_len = new_len
                log("  !! batch 已到 1 还爆显存 → 序列长度降到 %d，重切数据"
                    % new_len)

                def encode_short(row):
                    msgs = [{"role": "system",
                             "content": "你是写作助手「竹」，中文写作能力强，"
                                        "服从用户的开头、结尾、结构、字数等要求，"
                                        "改稿时只动用户指明的地方。"},
                            {"role": "user", "content": row["instruction"]},
                            {"role": "assistant", "content": row["output"]}]
                    return _mask_prompt(msgs)

                train_ds = Dataset.from_list(train_rows).map(
                    encode_short, remove_columns=["instruction", "output"],
                    num_proc=None, desc="tokenize-train-short")
                eval_ds = Dataset.from_list(eval_rows).map(
                    encode_short, remove_columns=["instruction", "output"],
                    num_proc=None, desc="tokenize-eval-short")
    log("  训练完成，用时 %.1f 分钟" % ((time.time() - t0) / 60))

    # ---- [6/7] 产物（多卡：只让主进程写盘，否则两个进程抢同一批文件）----
    if WORLD_SIZE > 1 and not IS_MAIN:
        log("  非主进程（rank=%d）：产物统一由主进程保存，这里直接收工"
            % LOCAL_RANK)
        return
    # 累计已训时长（只由主进程写）。用来盯「36 小时/月」的额度：
    # --max-hours 只管本次运行，月额度得靠自己记账 —— 这里就是那本账。
    used_h = (time.time() - t0) / 3600.0
    try:
        with open(cum_file, "w", encoding="utf-8") as f:
            f.write("%.4f\n" % (cum_before + used_h))
            f.flush()
            os.fsync(f.fileno())
        log("  累计已训：%.2f 小时（本次 %.2f 小时；月额度 36h，自己盯着）"
            % (cum_before + used_h, used_h))
    except Exception:
        pass
    adapter_dir = os.path.join(args.out, "adapter")
    model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)
    log("  LoRA 适配器已存：%s" % adapter_dir)
    if args.no_merge:
        log("  --no-merge：跳过合并完整模型（adapter 一样能用）")
    else:
        try:
            merged_dir = os.path.join(args.out, "merged")
            model = model.merge_and_unload()
            model.save_pretrained(merged_dir)
            tok.save_pretrained(merged_dir)
            log("  合并完整模型已存：%s" % merged_dir)
        except Exception as e:
            log("  合并模型跳过（显存不够很正常，adapter 一样能用）：%s" % e)
    metrics = trainer.state.log_history if hasattr(trainer, "state") else []
    with open(os.path.join(args.out, "训练报告.txt"), "w", encoding="utf-8") as f:
        f.write("竹 · 训练报告\n%s\n" % ("=" * 40))
        f.write("基座：%s\n样本：%d（训练 %d / 验证 %d）\n"
                "轮数：%d  LoRA r=%d  lr=%g\n"
                "用时：%.1f 分钟\n"
                % (args.model, len(rows), len(train_rows), len(eval_rows),
                   args.epochs, args.lora_r, args.lr,
                   (time.time() - t0) / 60))
        f.write("loss 曲线（每 20 步）：%s\n" % json.dumps(
            [m for m in metrics if "loss" in m][-20:], ensure_ascii=False))

    # ---- [7/7] 打包 ----
    log("[7/7] 打包产物")
    zip_path = os.path.join(ROOT, "竹训练产物.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for fp in glob.glob(os.path.join(args.out, "adapter", "*")) + \
                [os.path.join(args.out, "训练报告.txt"), LOG_PATH]:
            if os.path.exists(fp):
                z.write(fp, os.path.relpath(fp, ROOT))
    log("产物包：%s（平台下载这个就够）" % zip_path)
    log("=== 训练全部完成 ===")


def write_slurm_template():
    """第一次运行时，把 Slurm 作业脚本模板写到旁边（改个分区名就能 sbatch）。

    只在「文件不存在 / 基本是空的」时候写。绝不因为文件里出现某个字符串就重写——
    旧版那条「看到 C:\\ 就重写为服务器版」的判断会误伤手改过的作业脚本：
    只要注释里提一句 Windows 路径，就会被它当成旧版模板整个覆盖掉。"""
    tpl = os.path.join(ROOT, "提交作业.slurm")
    if os.path.exists(tpl) and os.path.getsize(tpl) > 100:
        return
    with open(tpl, "w", encoding="utf-8") as f:
        f.write(
            "#!/bin/bash\n"
            "# ===== 竹 · 训练作业脚本（国家超算互联网 scnet.cn，Slurm） =====\n"
            "# 提交：把 训练.py 和本文件放同一目录，然后  sbatch 提交作业.slurm\n"
            "# 查看：squeue -u $USER   取消：scancel <作业号>\n"
            "# 分区/队列名请按平台文档改成你申请到的（例如 gpu、A100、4090 等）\n"
            "#SBATCH --job-name=zhu_train\n"
            "#SBATCH --partition=gpu            # ← 按平台实际队列名改\n"
            "#SBATCH --nodes=1\n"
            "#SBATCH --ntasks=1\n"
            "#SBATCH --cpus-per-task=8\n"
            "#SBATCH --gres=gpu:1\n"
            "#   ↑ NVIDIA 分区填 gpu:1；海光 DCU 分区通常填 dcu:1\n"
            "#   （拿不准就 scontrol show partition <分区名> 看它的资源名）\n"
            "#SBATCH --time=24:00:00\n"
            "#SBATCH --output=训练-%%j.out\n"
            "#SBATCH --error=训练-%%j.err\n"
            "\n"
            "# 关键：先 cd 到本目录再启动，别在 home 直接跑（相对路径会断）\n"
            "cd %s\n"
            "\n"
            "python 训练.py --preset 大量 --essay-only\n"
            "# 规模可选：--preset 标准（轻量试跑）/ 大量（默认，60万条×3轮）/ 海量（150万条）\n"
            "# 中断了不要紧：重新 sbatch 提交即可，脚本认得自己的断点，接着练不花冤枉钱\n"
            "\n"
            "echo \"训练作业结束：$(date)\"\n" % ROOT)
    log("已生成作业脚本模板：%s（把 partition 和 cd 路径改成你的）" % tpl)


if __name__ == "__main__":
    write_slurm_template()
    main()
