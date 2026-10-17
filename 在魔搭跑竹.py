#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
在魔搭跑竹.py —— 在魔搭 ModelScope 的 CPU 实例上把「竹」跑起来
================================================================================
为什么要有这个脚本
------------------
本机只有 7.9GB 内存，1.5B 模型 fp32 要 6.2GB，跑不动。
魔搭的 **CPU 实例免费且不限时**，内存比本机大，而且 `/mnt/workspace/zhu` 是持久盘，
**代码、1.5B 基座、训练出来的 adapter 都已经在里面** —— 不用下载任何东西。

怎么用（在魔搭 CPU 实例的终端里）
--------------------------------
    cd /mnt/workspace/zhu
    python 在魔搭跑竹.py --check          # 先干跑：只检查环境 + 加载模型，验证能跑通
    python 在魔搭跑竹.py                  # 正式开：起模型服务 + 进竹的对话界面
    python 在魔搭跑竹.py --no-model       # 只用规则引擎（不挂神经模型，秒级出稿）

进去之后直接说人话就行，比如：
    写一篇关于坚持的作文，600字
    以「那一次我长大了」为题写一篇记叙文
    帮我把这篇改一下          （先把正文贴进去）

它是怎么工作的
--------------
    [1/4] 检查依赖（torch / transformers / peft / accelerate），缺了自动装
    [2/4] 找模型：优先 output/merged（已合并的完整模型），
          否则 output/adapter + 基座（基座按 models/base → models/models/Qwen--* 顺序找）
    [3/4] 后台起「本地模型服务.py」（OpenAI 兼容，127.0.0.1:8766），轮询 /health 等它就绪
    [4/4] 起「竹.py --cli」进入对话
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
CAND = [HERE, "/mnt/workspace/zhu", os.path.join(HERE, "zhu")]
ROOT = next((d for d in CAND if os.path.exists(os.path.join(d, "竹.py"))), HERE)
os.chdir(ROOT)

SERVICE = os.path.join(ROOT, "本地模型服务.py")
SERVICE_LOG = os.path.join(ROOT, "本地模型服务.log")
HEALTH = "http://127.0.0.1:8766/health"
GREEN, YELLOW, RED, CYAN, DIM, RESET = (
    "\033[32m", "\033[33m", "\033[31m", "\033[36m", "\033[2m", "\033[0m")


def say(m, c="", step=None):
    print("%s[%s]%s %s" % (c, step or time.strftime("%H:%M:%S"), RESET, m), flush=True)


# ────────────────────────────── 0. 同步最新代码
# 为什么要有这一步：魔搭发布包只放了"训练用"的文件子集，运行时文件（例如
# 本地模型服务.py）可能不在里面 —— 上一版就踩过：模型服务起不来，竹退回兜底路径，
# 输出变成"有道词典释义"那种东西。让脚本自己拉一遍，就不会再缺文件。
TOKEN = os.environ.get("MS_TOKEN", "ms-3cdacbf0-3908-48f4-ad7b-a8d9538f8332")
REPO = "bg20130613/zhu-train"


def sync_code():
    say("同步最新代码…(0/5)")
    try:
        from modelscope.hub.api import HubApi
        from modelscope import snapshot_download
        HubApi().login(TOKEN)                      # 仓库私有，必须先登录
        snapshot_download(REPO, local_dir=ROOT,
                          allow_patterns=["*.py", "*.json", "*.sh", "*.md",
                                          "记忆/*"])
        need = ["本地模型服务.py", "竹.py", "写作引擎.py", "写作素材库.py"]
        miss = [n for n in need if not os.path.exists(os.path.join(ROOT, n))]
        if miss:
            say("  同步完仍缺：%s（检查仓库里有没有传上去）" % miss, RED)
            return False
        say("  运行时文件齐了（本地模型服务.py / 竹.py / 写作引擎.py / 写作素材库.py）", GREEN)
        return True
    except Exception as e:
        say("  同步失败：%s" % str(e)[:160], YELLOW)
        say("  （不影响：本地已有的文件照样能用）", DIM)
        return True


# ────────────────────────────────── 1. 依赖
def ensure_deps():
    say("检查依赖…", step="1/5")
    miss = []
    for mod, pkg in [("torch", "torch"), ("transformers", "transformers"),
                     ("peft", "peft"), ("accelerate", "accelerate")]:
        try:
            __import__(mod)
        except Exception:
            miss.append(pkg)
    if not miss:
        import torch, transformers
        say("  都在（torch %s / transformers %s）" % (torch.__version__,
                                                  transformers.__version__), GREEN)
        return True
    say("  缺：%s —— 自动安装（约需几分钟）" % ", ".join(miss), YELLOW)
    cmd = [sys.executable, "-m", "pip", "install", "-q",
           "-i", "https://pypi.tuna.tsinghua.edu.cn/simple"] + miss
    r = subprocess.run(cmd)
    if r.returncode != 0:
        say("  装失败。手动跑：pip install %s -i https://pypi.tuna.tsinghua.edu.cn/simple"
            % " ".join(miss), RED)
        return False
    say("  装好了", GREEN)
    return True


# ────────────────────────────────── 2. 找模型
def find_artifacts():
    say("找训练产物…", step="2/5")
    for out in ("output", "output_long", "output_local"):
        m = os.path.join(ROOT, out, "merged")
        cfgp = os.path.join(m, "config.json")
        if os.path.exists(cfgp):
            # ★ 坑：4bit QLoRA 训练出来的 merged，config.json 里会带 quantization_config。
            # 加载它需要 bitsandbytes，而 **CPU 上 4bit 基本不可用**（服务会直接 ImportError 崩）。
            # 所以带量化配置的 merged 一律跳过，改走 adapter + 全精度基座。
            quant = False
            try:
                with open(cfgp, encoding="utf-8") as f:
                    _c = json.load(f)
                quant = bool(_c.get("quantization_config"))
            except Exception:
                pass
            if quant:
                say("  跳过 %s：它带 4bit 量化配置，CPU 上加载不了（需要 bitsandbytes）"
                    % os.path.relpath(m, ROOT), YELLOW)
            else:
                size = sum(os.path.getsize(os.path.join(m, f))
                           for f in os.listdir(m) if os.path.isfile(os.path.join(m, f)))
                say("  用已合并的完整模型：%s（%.0f MB）" % (m, size / 1048576), GREEN)
                return ("merged", m)

    for out in ("output", "output_long", "output_local"):
        a = os.path.join(ROOT, out, "adapter")
        if os.path.exists(os.path.join(a, "adapter_config.json")):
            base = find_base()
            if not base:
                say("  找到 adapter 但**找不到基座模型**", RED)
                return (None, None)
            say("  用 adapter + 基座：", GREEN)
            say("    adapter = %s" % a, DIM)
            say("    基座    = %s" % base, DIM)
            return ("adapter", (a, base))

    say("  没找到任何训练产物（output*/merged 或 output*/adapter）", RED)
    return (None, None)


def find_base():
    """跟 本地模型服务.py 的 find_base 保持一致的查找顺序。
    ⚠️ 注意：它最后那条 sorted(glob('models/models/Qwen--*')) 是**字母序**，
    'Qwen--Qwen2.5-0.5B' 会排在 '1.5B' 前面 —— 所以本脚本优先看 models/base。"""
    b = os.path.join(ROOT, "models", "base")
    if os.path.exists(os.path.join(b, "config.json")):
        return b
    cands = sorted(glob.glob(os.path.join(ROOT, "models", "models", "Qwen--*",
                                          "snapshots", "*")))
    for c in cands:
        if os.path.exists(os.path.join(c, "config.json")):
            # 确认权重真的在（半成品 .incomplete 不算）
            w = glob.glob(os.path.join(c, "*.safetensors")) + \
                glob.glob(os.path.join(c, "*.bin"))
            if w:
                return c
    return None


# ────────────────────────────────── 3. 起模型服务
def service_alive():
    try:
        with urllib.request.urlopen(HEALTH, timeout=2) as r:
            return b'"ok"' in r.read()
    except Exception:
        return False


def start_service(timeout_min=15):
    say("起本地模型服务（后台）…", step="3/5")
    if service_alive():
        say("  已经在跑了，直接用", GREEN)
        return True
    if not os.path.exists(SERVICE):
        say("  找不到 本地模型服务.py", RED)
        return False
    logf = open(SERVICE_LOG, "w", encoding="utf-8")
    proc = subprocess.Popen([sys.executable, "-u", SERVICE], stdout=logf,
                            stderr=subprocess.STDOUT, cwd=ROOT)
    say("  日志：%s" % SERVICE_LOG, DIM)
    say("  CPU 上加载 1.5B 模型比较慢（首次要 1~5 分钟），耐心等…", DIM)
    t0 = time.time()
    last = 0
    while time.time() - t0 < timeout_min * 60:
        if service_alive():
            say("  就绪（用时 %.0f 秒）" % (time.time() - t0), GREEN)
            return True
        # 进程已经死了就别再等 —— 之前这个漏了，白白等 400 秒
        if proc.poll() is not None:
            say("  模型服务进程已退出（退出码 %s），不再等待。日志末尾：" % proc.returncode, RED)
            try:
                with open(SERVICE_LOG, encoding="utf-8", errors="replace") as f:
                    for line in f.readlines()[-15:]:
                        print("    " + line.rstrip())
            except Exception:
                pass
            return False
        el = time.time() - t0
        if el - last >= 20:
            last = el
            tail = ""
            try:
                with open(SERVICE_LOG, encoding="utf-8", errors="replace") as f:
                    tail = "".join(f.readlines()[-3:]).strip().replace("\n", " | ")
            except Exception:
                pass
            say("  …等了 %.0f 秒 %s" % (el, tail[-160:]), DIM)
        time.sleep(3)
    say("  超时 %d 分钟还没起来。看日志：%s" % (timeout_min, SERVICE_LOG), RED)
    try:
        with open(SERVICE_LOG, encoding="utf-8", errors="replace") as f:
            print("".join(f.readlines()[-25:]))
    except Exception:
        pass
    return False


# ────────────────────────────────── 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="只检查环境并预热模型，不进对话")
    ap.add_argument("--no-model", action="store_true",
                    help="不起神经模型，只用规则引擎（秒级出稿）")
    ap.add_argument("--timeout", type=float, default=15, help="等模型就绪的最长分钟数")
    args = ap.parse_args()

    say("竹 · 魔搭 CPU 实例启动器", CYAN)
    say("工作目录：%s" % ROOT, DIM)
    say("Python %s ｜ %s" % (sys.version.split()[0], sys.executable), DIM)
    print()

    if not sync_code():
        say("  同步失败：缺运行时文件，先人工确认一下", YELLOW)

    if not ensure_deps():
        return 1

    kind, path = (None, None)
    if not args.no_model:
        kind, path = find_artifacts()
        if not kind:
            say("（没模型也能用规则引擎，加 --no-model 试试）", YELLOW)

    if args.check:
        say("[3/4] --check：跳过起服务（依赖和产物都验过了）", GREEN)
        say("[4/4] --check 结束。要真跑，去掉 --check 再来一次。", GREEN)
        return 0

    if kind and not start_service(args.timeout):
        say("模型服务没起来 —— 仍可继续（竹会退回规则引擎）", YELLOW)

    say("进竹的对话界面…", step="4/5")
    say("直接说人话，例如：  写一篇关于坚持的作文，600字", CYAN)
    say("退出：输入 退出 / exit / Ctrl+C", DIM)
    print()
    time.sleep(1)

    zhu = os.path.join(ROOT, "竹.py")
    if not os.path.exists(zhu):
        say("找不到 竹.py，没法进入对话", RED)
        return 1
    try:
        # --cli = 老式黑窗口聊天；在服务器上没有浏览器，网页版用不了，所以走 cli
        return subprocess.call([sys.executable, "-u", zhu, "--cli"], cwd=ROOT)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n已退出（模型服务还在后台跑着，不想要就 kill 掉）")
        sys.exit(0)
