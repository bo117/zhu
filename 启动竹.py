# -*- coding: utf-8 -*-
"""双击我 = 竹上班。
会自动弹出网页聊天界面（127.0.0.1:8765，浏览器自己会打开）。
训练过模型的话，本地模型服务（127.0.0.1:8766）会一起在后台点火。
想用老式黑窗口文字聊天，把下面那行末尾加个参数： --cli"""
import os
import subprocess
import sys
import urllib.request

os.chdir(os.path.dirname(os.path.abspath(__file__)))


def _model_ready():
    """训练产物存在吗（merged 完整模型或 LoRA 适配器都算）"""
    for out in ("output", "output_long", "output_local"):
        if os.path.exists(os.path.join(out, "merged", "config.json")):
            return True
        if os.path.exists(os.path.join(out, "adapter", "adapter_config.json")):
            return True
    return False


def _service_alive():
    try:
        with urllib.request.urlopen("http://127.0.0.1:8766/health",
                                    timeout=1.5) as r:
            return b'"ok": true' in r.read() or b'"ok":true' in r.read()
    except Exception:
        return False


if _model_ready() and not _service_alive():
    venv_py = os.path.join(".train_venv", "Scripts", "python.exe")
    if os.path.exists(venv_py):
        subprocess.Popen([venv_py, "本地模型服务.py"],
                         creationflags=subprocess.CREATE_NEW_CONSOLE)
        print("本地模型服务点火中（首次加载模型要十几秒）……")

subprocess.call([sys.executable, "竹.py"])
input("竹下班了，按回车关掉这个窗口。")
