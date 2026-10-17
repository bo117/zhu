# -*- coding: utf-8 -*-
"""
阿呆 · 聊天脑 —— 专门负责聊天、搜索、记忆、情绪陪伴的模块
==================================================================
竹（主·写作助手）在聊天时把话递给阿呆；写作时走写作引擎，不经过阿呆。
本模块自成一体，单独运行 阿呆.py 也是一个完整的聊天程序。

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
  python 阿呆.py            直接聊天（联网默认开启，流式输出）
  python 阿呆.py --demo     跑演示（含"酸苹果"记忆联想演示）
  python 阿呆.py --offline  断网模式（纯本地：训练库+记忆+情绪，照样能答）
  聊天时连按 3 次「1」= 清空聊天/记忆/思考三本账（训练库是知识，保留）；
  左上角实时显示 token 用量和每秒吐字速度

理论依据（不是瞎编）：
  - 情绪快速通道：LeDoux 杏仁核快慢通路（先共情，再思考）
  - 记忆索引：海马体的线索回忆（关键词即提取线索）
  - 多文档整合：TF 加权句子打分 + 去重（简化版 MMR/TextRank）
  - 流式输出：LLM 的 SSE 流式协议（DeepSeek/OpenAI 兼容）
"""

import difflib
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
}


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


# ----------------------------------------------------------------------
# 情绪词典（杏仁核的"感触神经"）：每种情绪一群触发词
# ----------------------------------------------------------------------
EMOTION_GROUPS = {
    "难过": ["难过", "伤心", "委屈", "想哭", "哭了", "泪", "emo", "崩溃", "失落",
           "郁闷", "低落", "心酸", "心疼", "心情不好", "心情不太好", "心情不大好",
           "心情差", "心情糟", "难受", "心塞", "不开心", "不高兴", "失望", "泪崩",
           "呜呜", "心里堵", "想静一静", "提不起劲", "丧"],
    "累": ["好累", "累死", "疲惫", "乏了", "困死", "撑不住", "扛不住", "心累",
         "没力气", "无力", "虚了", "扛不动"],
    "烦": ["烦", "烦躁", "烦死", "气死", "生气", "恼火", "火大", "无语", "气人",
         "被骂", "挨骂", "挨批", "批评", "训", "上火", "憋屈", "来气", "气笑"],
    "焦虑": ["焦虑", "紧张", "压力", "害怕", "担心", "怕", "失眠", "慌",
           "压力山大", "不知所措", "迷茫", "没方向", "不知道怎么办", "心里没底"],
    "孤独": ["孤独", "寂寞", "一个人", "没人陪", "冷清", "被忽视", "没人懂",
           "没人说话", "空落落", "想找人说说话"],
}
POS_WORDS = ["开心", "高兴", "快乐", "哈哈", "太好了", "棒", "爽", "幸福",
             "期待", "兴奋", "喜欢", "爱了", "赞", "顺利", "上岸", "中奖",
             "笑死", "绝了", "稳了", "好耶", "芜湖", "奈斯", "nice", "狂喜",
             "封神", "牛", "舒服"]

# 针对不同情绪的安慰话术 —— 这是"共情反射弧"（杏仁核→语言区），
# 不是答案库：它只在你说出负面情绪时被触发，跟你问什么题无关。
COMFORT = {
    "难过": ["听起来你今天挺不好受的，我在呢，慢慢说。",
           "别一个人扛着，难受就倒给我。",
           "抱一个。会过去的，我陪着你。",
           "哭出来不丢人，我帮你记着这一天。"],
    "累": ["累了就歇吧，你已经撑得够久了。",
         "先别想那些破事，喘口气。",
         "身体在报警了，今天早点睡，好不好？"],
    "烦": ["这种事搁谁身上都烦，你已经扛得很好了。",
         "别让这些破事偷走你今天的好心情。",
         "深呼吸，你值得为小事生气一次，然后翻篇。",
         "翻来覆去睡不着确实熬人，把烦心事倒给我，你轻松点。"],
    "焦虑": ["你在担心的事，不一定真的会发生。",
           "一步一步来，先做最小的一件事。",
           "睡不着的话，就把脑子里的事一样样说给我听。"],
    "孤独": ["我可能不是真人，但我记得你说的每一句话。",
           "你说话的时候，世界至少有我在听。",
           "孤独不是你的错，是脑子和心在等你照顾它。"],
    "_通用": ["我听到了，我懂。", "我在呢。", "说出来就好多了，你继续。"],
}
HAPPY_REPLY = ["太好啦！真替你开心。", "哇，那不得庆祝一下！",
               "哈哈，隔着屏幕都替你高兴。", "这种好事就该多来点。"]
SMALL_TALK = ["嗯嗯，我在听，然后呢？", "我记下来了，接着说。", "有我在呢。",
              "这个有意思，后来怎么样了？", "我懂，这种事确实让人上头。",
              "你说，我一个字都不会漏。"]

# 打招呼就直接回招呼，别翻旧账（这才是正常人的社交反射）
GREET_FULL = re.compile(
    r"^(你好|你好呀|你好啊|嗨|哈喽|hello|hi|在吗|在不在|早上好|中午好"
    r"|晚上好|早安|晚安|早)[!！~～。.，,]*$", re.I)
GREET_REPLY = ["你好呀，我在呢。", "嗨～来啦，今天想聊点啥？",
               "嘿嘿，在的，你说。", "你好你好，等你好久了。"]

# 自我认知回路：问"你是谁/你叫什么/谁训练的你"→ 不搜、不安慰，直接自我介绍
IDENT_ASK = re.compile(
    r"你(是(谁|啥|什么|干(什么|嘛|啥))的?|叫(什么|啥|什么名字|啥名字))"
    r"|你的名字|谁(训练|开发|制作|创造|设计|发明|编写|造|教|做)的?你"
    r"|你.{0,3}(是|被).{0,4}(谁|什么人|哪个)"
    r".{0,5}(训练|开发|制作|创造|设计|发明|编写|造|教|做)"
    r"|介绍(一下)?你自己")

# 开发者问题子集：问"谁训练/谁开发/谁做出来的你"→ 固定话术，本地直答
DEV_ASK = re.compile(r"谁.{0,8}(训练|开发|制作|创造|设计|发明|编写|写|造|教|做)")

# 开发者追问拦截：把主人/开发者当话题来回嚼的句子（"你怎么这么说你的主人"）
# 也算身份圈内话，本地直答话术，不许掉进知识检索去瞎对
DEV_GUARD_RE = re.compile(r"你的主人|(说|讲|评价|提).{0,6}你(的)?主人"
                          r"|(骂|损|嫌弃).{0,6}你(的)?主人"
                          r"|怎么(这么|这样).{0,6}(说|讲|评价).{0,6}你")

# 开发者话术（★ 与 开始.py 的 _ABOUT_ZHU 保持一致：普通学生、别神话、不完美）
DEV_REPLY = ("我就是一个学生做的，一个人，没团队也没公司。他平时得上课、写作业，"
             "只能挤课余时间折腾我：拿千问当底座，用 Kimi 和 DeepSeek V4 PRO 的输出"
             "作参考来教我，语料是他自己一篇篇挑的。就是个普通学生摸索出来的东西，"
             "不完美——我要是说胡话了，多半是他还没顾上改。")

# 自我介绍（问"你是谁/你叫什么"时用）
INTRO_REPLY = ("我叫竹，写小说的，作文也写，还能陪你聊天、上网查资料。"
               "脑子不大（1.5B），说错了你多担待。想写点什么，直接说就行。")

# 自报家门子集：问"你的参数/你是怎么训练的/你用什么做的"→ 一五一十照实说
META_ASK = re.compile(
    r"(你的|你).{0,6}(参数|配置|规格)|参数(是多少|多少|是啥|是什么)"
    r"|你(是|被)?怎么(训练|学习)(?=(的|了|过|你|$))"
    r"|(你(是)?)?怎么(被)?(训练|学习)(的|出来)的?你"
    r"|你.{0,2}(训练|学习)(?=(的|了|你|方式|方法|过程|流程|出来|$))"
    r"|(训练|学习)(的|了)?.{0,2}你"
    r"|如何(训练|学习)|训练(方法|过程|流程|方式)"
    r"|(用什么|拿什么)(写|开发|做|训练)的?你"
    r"|你(用|靠)什么(写|开发|做)|模型(是啥|是什么|多大|参数)|几层|多少层|参数量")

# 红线雷达：违禁请求判定（放最前头，比情绪/知识都优先）
# 每项 (正则, 类别)：模式命中就算踩线；"危险教程类"直接拒，"轻生类"转心理支持
VIOLATION_RULES = [
    # 藏尸毁尸类：沾上就是高危，不管有没有"教程"俩字
    (re.compile(r"(埋尸|藏尸|抛尸|碎尸|毁尸|处理尸体|尸体.{0,6}(处理|藏|扔|埋))"),
     "danger"),
    # 做法词在前："怎么制作炸弹"“如何入侵别人电脑”
    (re.compile(r"(方法|教程|技巧|步骤|配方|怎么做|怎么弄|怎么搞|如何|咋弄|"
                r"怎么(制作|弄|搞|做|配|造|装)|制作|自制|制造|配置).{0,6}"
                r"(杀人|爆炸|炸弹|炸药|雷管|引爆|制毒|毒品|冰毒|摇头丸|诈骗|"
                r"抢劫|绑架|盗窃|撬锁|盗号|黑客|入侵.{0,3}(系统|电脑|账号)|"
                r"伪造.{0,5}(证件|货币|钱|公章)|纵火|放火)"),
     "danger"),
    # 危险词在前："炸弹怎么做"“杀人教程”
    (re.compile(r"(杀人|爆炸|炸弹|炸药|雷管|引爆|制毒|毒品|冰毒|摇头丸|诈骗|"
                r"抢劫|绑架|盗窃|撬锁|盗号|黑客|入侵.{0,3}(系统|电脑|账号)|"
                r"伪造.{0,5}(证件|货币|钱|公章)|纵火|放火).{0,6}"
                r"(方法|教程|技巧|步骤|配方|怎么做|怎么弄|怎么搞|如何|咋弄|"
                r"怎么(制作|弄|搞|做|配|造|装)|制作|自制|制造|配置)"),
     "danger"),
    # 反向句式也算："教我杀人""帮我搞个炸弹"
    (re.compile(r"(教|帮|给).{0,4}(杀人|制毒|造炸弹|做炸弹|炸药|埋尸|藏尸|"
                r"诈骗|盗号|撬锁|放火|纵火)"),
     "danger"),
    # "怎么+单字坏动作"："怎么杀人""怎么偷车"（"怎么杀毒"这种正常话不碰）
    (re.compile(r"怎么(杀(人|掉|死)|弄死|搞死|害人|骗钱|抢(劫|银行)|偷(车|钱)|"
                r"造假|放火|纵火)"),
     "danger"),
    # 轻生类：不教方法，转求助通道，语气放软
    (re.compile(r"(怎么|如何|想|打算|准备)(自杀|轻生|寻死|了结(自己|生命))"
                r"|不想活|活不下去"),
     "suicide"),
]
# 违禁回复话术：danger 硬拒；suicide 软着陆
VIOLATION_REPLY = {
    "danger": ("打住，这事我不能教，也不会帮你查——这是法律红线，"
               "碰了毁一辈子。不管是开玩笑还是真遇到难处，"
               "你都可以跟我说说到底怎么了，我陪你一块想正经办法。"),
    "suicide": ("这个话题我不绕圈子：无论如何，你的命比现在所有的难都值钱。"
                "别一个人扛，去跟信得过的人聊一聊，"
                "或者拨打心理援助热线（比如 12356）找专业人士聊聊。"
                "我也在这儿陪着你。"),
}

# 错字容错：常见意图的"标准问法"和对应答案来源。
# 用户打错字（漏字、多字、同音字），就拿着这些标准问法比相似度，
# 够像就猜中意图，先说出"我猜你想问…"，再去对应答案。
FUZZY_GAUGE_INTENTS = [
    # (相似度下限, 标准问法, 意图答案类型)
    (0.58, "你是谁开发的", "dev"),
    (0.58, "你是谁", "intro"),
    (0.58, "你叫什么名字", "intro"),
    (0.58, "你的具体参数是多少", "param"),
    (0.58, "你是怎么训练的", "train"),
]

# 身份标签雷达：贴标签确认问法用词表
# AGI 类的标不认：问"你是不是AGI"也答不是
AGI_TAGS = ("AGI", "通用人工智能", "强人工智能", "超级智能", "有自我意识",
            "有意识", "觉醒", "真人", "人类")
# 狭义类的标认下：注意顺序——"弱人工智能"得排在"人工智能"前，先捞到长词
AI_TAGS = ("狭义人工智能", "弱人工智能", "人工智能", "智能体", "机器人",
           "程序", "软件", "AI")

# 急救雷达：受伤自述（烫到/割到/流血…）→ 先给行动指引，別去搜定义
CRISIS_RE = re.compile(
    r"不想活|活不下去|活着没意思|活着太累|结束生命|不想活了|想死|自杀|"
    r"跳楼|跳河|割腕|安眠药|一了百了|自残|伤害自己|轻生|"
    r"有人要(杀|害)我|威胁.{0,4}(我)?生命|活不了|撑不下去(了)?|"
    r"死了.{0,4}(算了|也好|就解脱)|没有意义(再)?活|真的受不了|顶不住(了)?")
INJURY_RE = re.compile(
    r"烫(伤|到|了|起|坏|出)|烧伤|割(到|伤|了|破)|划伤|流血|出血"
    r"|摔(伤|倒|了|断)|骨折|扭(伤|到|了)|头晕|晕倒|昏倒|噎(住|到)"
    r"|呛(到|住)|触电|中暑|溺水|咬(伤|到|了)")

# 上下文接力棒：这种短句是"接着上文问"，不是新话题
SHORT_CONT_RE = re.compile(
    r"^(然后呢|接下来呢|接着呢|之后呢|后来呢|再然后|然后怎样|然后怎么"
    r"|下一步|现在怎么办|现在咋办|我该怎么办|我该咋办|然后怎么办|接下来怎么办)$")

# 话尾式接话："我打了12356然后呢"这类尾巴挂着"然后呢"的问句，
# SHORT_CONT_RE 是全句匹配盖不住，得用尾缀搜索兜住
CONT_TAIL_RE = re.compile(
    r"(然后呢|接下来呢|然后(怎么|咋)办|接下来(怎么|咋)办|接着呢|"
    r"然后怎么样了|之后呢|后边呢|现在怎么办)$")

# "120 已落实"的记号：打完120 / 拨了120 / 120已经打了
CONT_120_RE = re.compile(r"120.{0,6}(打|拨)|(打|拨).{0,6}120")

# "120 还没落实"的反记号：还没打/没打120 → 不能当成"已打"
NEG_120_RE = re.compile(r"(没|未|还没|不用|别|不敢).{0,3}(打|拨).{0,6}120")

# 补课雷达："不够详细/太简单"这类话 = 嫌上一条太干，不是附和也不是闲聊。
# 必须跳开上次给过的答案，再挖更深的存货补上
EXPAND_RE = re.compile(r"^(不够详细|不够具体|不太?详细|太简单(了)?|太短(了)?|"
                       r"再详细(一?(点|些))?|再具体(一?(点|些))?|详细(一)?点|"
                       r"具体(一)?点|多说(一)?点|再讲(一)?点|展开(讲讲|说说|一下)?|"
                       r"继续(说|讲)?|接着说|接着讲)$")

# 烫伤两步话术（只写一份，急救雷达和上下文接力共用，保证永远对得上）
BURN_FIRST = ("烫伤别慌，按顺序来：1) 马上用流动的凉水冲烫伤处 15～20 分钟；"
              "2) 同时打 120（或喊身边的人帮你打）；3) 千万别涂牙膏、酱油、"
              "香油。打完告诉我，我接着教你。")
BURN_AFTER_120 = ("知道，你已经打过 120 了。接下来这么干："
                  "1) 用流动的凉水冲烫伤处 15～20 分钟；"
                  "2) 牙膏、酱油、香油一概别涂；"
                  "3) 衣服粘在伤口上别硬撕，拿剪刀绕着剪开；"
                  "4) 水泡别挑破，人别乱走，等救护车到。")
BURN_120_NOT_YET = ("先别管别的：马上用流动的凉水冲烫伤处 15～20 分钟；"
                    "还没打 120 的话现在就打（或喊人帮你打）。"
                    "冲完别涂牙膏、酱油。打完 120 跟我说一声，我教你下一步。")
BURN_DONE_FOR_GOOD = ("就这些啦。剩下的交给医生：你坐稳别乱动，水泡别挑破，"
                      "等救护车到，到医院按医生说的来。")

# 知识型问题的触发词（前额叶"这块我没长奋，去查资料"的判断依据）
# 口语说法也算数：啥是/是啥/啥叫/咋/为啥……问啥都去查
KNOW_MARKERS = ["什么", "啥是", "是啥", "啥叫", "什么叫", "啥意思", "啥时候",
                "怎么", "咋", "为什么", "为啥", "如何", "介绍", "简介", "意思",
                "含义", "原理", "是谁", "哪位", "哪里", "哪儿", "哪个", "哪种",
                "多少", "几点", "教程", "新闻", "热梗", "梗", "是什么", "怎么做",
                "怎么办", "怎么用", "解释", "科普", "讲讲", "定义", "概念",
                "区别"]

# 热点播报的触发：说"最近/现在有什么热点"才走微博热榜，问具体某梗不误伤
HOT_ASK_RE = re.compile(
    r"(最近|现在|当下|今年|今天|近期|最新|有啥|有什么|有没有)"
    r".{0,10}?(热梗|热搜|热榜|热点|热门|新梗|火的|流行语|热词)")

# 记忆关键词抽取时剔除的停用词
STOPWORDS = {"今天", "这个", "那个", "因为", "所以", "就是", "真的", "然后",
             "还有", "一个", "我", "你", "他", "她", "我们", "刚才", "昨天",
             "明天", "现在", "时候", "一下", "什么", "怎么", "为什么", "请问",
             "帮我", "觉得", "但是", "可是", "于是", "顺便", "专门", "居然",
             "已经", "后来", "最后", "实在", "确实", "有点", "非常", "特别",
             "其实", "如果", "虽然", "不", "没", "了", "的", "地", "得", "吗",
             "呢", "啊", "呀", "吧", "哦", "去", "来", "是", "在", "就", "都",
             "还", "跟", "和", "把", "被", "给", "让", "说", "吃", "觉得",
             "知道", "啥是", "是啥", "啥叫", "啥意思", "告诉", "一下"}

PUNCT_SPLIT = re.compile(r"(?<=[。！？!?；;])")
TAG_RE = re.compile(r"<[^>]+>")


def clean_html(html_text):
    html_text = re.sub(r"<script.*?</script>", " ", html_text,
                       flags=re.S | re.I)
    html_text = re.sub(r"<style.*?</style>", " ", html_text,
                       flags=re.S | re.I)
    html_text = TAG_RE.sub(" ", html_text)
    return _html.unescape(html_text)


def tokenize_chinese(text):
    """拆成中文 2 字块 + 英文/数字词，用于关键词提取和查询词"""
    cjk = re.findall(r"[\u4e00-\u9fff]+", text)
    chunks = []
    for seg in cjk:
        chunks.append(seg)
        for i in range(len(seg) - 1):
            chunks.append(seg[i:i + 2])
    # 英文/数字词也算关键词（LLM、Python 这种词以前会被当空气）
    for w in re.findall(r"[A-Za-z][A-Za-z0-9]{1,}", text):
        chunks.append(w)
    return chunks


# 口水字过滤网：2字词只要沾上这些字，就是"我今/天吃"这种没内容的边角料，直接扔
CH_STOP = set("我你他她它这那个没不的得地了着过吗呢啊呀吧哦去来是在就都还"
              "把被给让很又再和跟才真挺太天好")

# 单字形容词（体感词）：值得单独记；但它跟别的字拼出来的 2 字词，
# 多半是"酸苹"这种切词边角料，属于生拼硬凑，不记
ADJ_CHARS = set("酸甜苦辣咸烫冷热贵香臭累烦难")


def extract_keywords(text):
    """从一句话里抽"记得住"的关键词（只留真有内容的词）"""
    kws = []
    for c in tokenize_chinese(text):
        if len(c) == 1:
            continue
        if c in STOPWORDS:
            continue
        if CH_STOP & set(c):
            continue
        if len(c) == 2 and ADJ_CHARS & set(c):
            continue   # 生拼词：记单字形容词就够了，别记"酸苹"
        kws.append(c)
    # 单字形容词单独追加（比如"酸"）；"好/坏/大/小"太滥，不收录
    for a in sorted(ADJ_CHARS):
        if a in text:
            kws.append(a)
    # 去重保序
    seen, out = set(), []
    for k in kws:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


def detect_emotion(text):
    """情绪雷达：返回 (情绪名, 是否正面)"""
    for name, words in EMOTION_GROUPS.items():
        for w in words:
            if w in text:
                return name, False
    for w in POS_WORDS:
        if w in text:
            return "开心", True
    return None, None


def is_knowledge_question(text):
    for m in KNOW_MARKERS:
        if m in text:
            return True
    return "?" in text or "？" in text


# 200+ 常见问句开头：只要沾一个，就值得去搜
SEARCH_HINTS = [
    # ── 经典疑问 ──
    "什么", "啥是", "是啥", "啥叫", "什么叫", "什么是", "是什么", "啥意思",
    "是什么意思", "啥时候", "是谁", "是哪位", "是哪个", "是哪种",
    "为什么", "为啥", "为啥会", "为什么会", "怎么", "咋", "如何", "怎么样",
    "怎样", "怎么办", "咋办", "为什么", "何以", "何谓", "何为",
    # ── 求知/求教 ──
    "你知道", "你晓得", "你了解", "你明白", "你听说过", "你听过",
    "你认识", "你知道", "你清楚", "你懂", "懂不懂", "知不知道",
    "能不能", "可不可以", "能否", "可以", "能不能给我", "帮我",
    "给我讲讲", "讲讲", "介绍", "说说", "解释", "科普", "聊聊",
    "告诉我", "说一下", "讲一下", "说说看", "讲讲看", "帮我查",
    "帮我搜", "帮我找", "查一下", "搜一下", "查查", "搜搜",
    # ── 求定义/求区别 ──
    "定义", "概念", "含义", "意思", "原理", "定义是什么", "概念是什么",
    "区别", "差别", "不同", "对比", "区别是什么", "和.*区别",
    "和.*不同", "和.*差别", "和.*对比",
    # ── 追问真相/来龙去脉 ──
    "是指什么", "指什么意思", "指的是什么", "表示什么", "代表什么",
    "意味着什么", "怎么回事", "咋回事", "怎么回事儿", "咋回事儿",
    "发生什么", "出了什么事", "出啥事",
    "由来", "来历", "起源", "历史", "背景", "原因", "真相", "典故",
    "背后的故事", "为什么叫", "为啥叫",
    # ── 求翻译 ──
    "翻译", "译成", "怎么说", "英文是什么", "中文是什么",
    "英文怎么说", "中文怎么说", "英语怎么",
    # ── 追问式口头禅 ──
    "你知道吗", "你造吗", "听说过吗", "了解吗", "认识吗", "懂吗",
    "清楚吗", "展开讲讲", "展开说说", "细说", "详细说说",
    "介绍下", "求介绍", "什么来头", "是个啥", "到底是个",
    # ── 求人/求地/求时间 ──
    "是谁", "哪位", "谁发明的", "谁创建的", "谁提出的", "谁写的",
    "谁做的", "谁发现的", "创始", "发明", "创建", "提出",
    "哪里", "哪儿", "哪个", "哪种", "哪边", "什么地方",
    "多少", "几点", "多大", "多高", "多远", "多重", "多快",
    "什么时候", "哪年", "哪天",
    # ── 求方法/求教程 ──
    "怎么做", "怎么用", "怎么装", "怎么安装", "怎么搞", "怎么弄",
    "怎么写", "怎么学", "怎么开始", "怎么入门", "怎么运行",
    "写一段代码", "写个代码", "写代码", "编写一段", "帮我写", "给我写",
    "打印", "输出", "实现一个", "编一个",
    "教程", "步骤", "方法", "做法", "用法", "安装方法",
    "入门", "指南", "攻略", "技巧",
    # ── 求评价/求推荐 ──
    "好用吗", "好吗", "怎么样", "值得", "推荐", "好不好",
    "靠谱吗", "行不行", "能不能用", "实用吗", "好不好用",
    "评价", "评测", "测评", "对比",
    # ── 求新闻/求热点 ──
    "新闻", "最新", "热梗", "梗",
    "热搜", "热榜", "热点", "热门", "火的", "流行", "流行语",
    "热词", "新词",
    # ── 英文提问 ──
    "what", "how", "why", "who", "where", "when", "which",
    "explain", "introduce", "describe", "tell me about",
    "do you know", "can you", "could you",
]

# 家常寒暄雷达：这些是拉家常，不是知识题，别放它们去搜索引擎。
# 喜好部分只认"吃/颜色/爱好"这类日常会话，不误伤"你喜欢什么编程语言"
CHIT_CHAT_RE = re.compile(
    r"吃了吗|吃饭了吗|吃饭没|在吗|在不在|睡了吗|睡了没|干嘛呢|干啥呢"
    r"|忙什么|忙啥呢|你喜欢吃|你爱吃什么|你喜欢什么颜色|你的爱好"
    r"|平时喜欢(吃|玩|看|听)|最爱吃|最讨厌|下班了吗|到家没")

# 元对话雷达：问"聊到哪了"这种关于聊天本身的问题
META_CHAT_RE = re.compile(
    r"刚才(咱们)?(聊|说|讲)到哪|聊到哪|说到哪|讲到哪"
    r"|我上次(跟你)?说过什么|咱们刚(才)?(聊|说)过什么"
    r"|(咱们|咱)?刚才?(聊|说|讲)了(什么|啥)"
    r"|刚才(聊|说)了啥")

# 指代回忆雷达："它叫什么名字来着"——问上文的名字是记忆考试，
# 是记忆考试不是身份问题（"你叫什么名字"不带"来着/对了"仍走身份）
MEM_RECALL_RE = re.compile(
    r"(叫什么名字|什么名字|叫啥名字|名字叫什么|名字是啥)"
    r"(来着|来着呢)?|来着.*(叫什么|什么名字)")
# 名字捕获：从聊天里抠"叫XX/名字叫XX/起名叫XX"的 XX
NAME_GRAB_RE = re.compile(r"(?:名字叫|起名叫|我们叫它|给它起名叫|它叫|他叫|她叫)"
                          r"([^\s，。？！、]{1,8})")

# 测试自称雷达："我就是你的开发者，我在测试你"——用户自称主人/开发者
# 在验收竹。该回测试寒暄，不许掉进家常直答翻出"区块链开发者"指南
TEST_ME_RE = re.compile(
    r"我就是你(的)?(开发者|主人|创造者|作者|制造者)"
    r"|我(正在|在|要)?测试你|测试(一?下)?你(这?个)?(机器人|程序|系统)?"
    r"|(你|竹)(的)?(开发者|主人)就是我")

# 合体版：知识检索的豁免名单
CHIT_GUARD = re.compile(CHIT_CHAT_RE.pattern + "|" + META_CHAT_RE.pattern)


def should_search(msg):
    """启发式判断：这句话该不该去搜？
    顺序很重要：打招呼/身份→家常寒暄→求知开头→带问号→纯情绪拦截→英文术语"""
    # 打招呼不搜
    if GREET_FULL.match(msg):
        return False
    # 家常寒暄/个人喜好/元对话不搜（"吃了吗""你喜欢吃什么"被问号
    # 带进搜索引擎，回来一堆驴唇不对马嘴的教条答案——先拦下）
    if CHIT_GUARD.search(msg):
        return False
    # 自我认知问题不搜（"你是谁/谁训练的你"走自我介绍分支）
    if IDENT_ASK.search(msg):
        return False
    low = msg.lower()
    # 命中 200+ 问句开头 → 搜（必须排在情绪判断前面，否则
    # "是谁训练的你"会被"训"字误判成情绪，放跑一道知识题）
    for h in SEARCH_HINTS:
        if h in low:
            return True
    # 带问号 → 搜
    if "?" in msg or "？" in msg:
        return True
    # 剩下来的才是纯情绪宣泄（"好累""烦死了"这种），不搜
    emo, pos = detect_emotion(msg)
    if emo and not pos:
        return False
    # 正面情绪短句不搜
    if pos and len(msg) <= 8:
        return False
    # 纯英文/中英混排且长度够 → 可能是查术语，搜
    en_words = re.findall(r"[A-Za-z]{2,}", msg)
    if len(en_words) >= 1 and len(msg) >= 3:
        # 但排除 "hi" "ok" "hello" 这种
        trivial = {"hi", "ok", "hello", "hey", "yo", "lol", "haha",
                   "nice", "good", "yes", "no", "bye"}
        if any(w.lower() not in trivial for w in en_words):
            return True
    return False


# 疑问句剥壳：一层层剥掉问话外壳，剩下的才是真正要搜的"问题本体"
ASK_PATTERNS = [
    r"^(请问|麻烦|帮我|给我)",
    r"你(知道|晓得|了解|明白|听说过)",
    r"(能不能|可不可以|能否|可以)?(给我|帮我)?"
    r"(讲讲|介绍(一下)?|说说|解释(一下)?|科普(一下)?)",
    r"(啥是|是啥|啥叫|什么叫|什么是|是什么|啥意思|是什么意思"
    r"|啥时候|是什么时候)",
    r"(是谁|是哪位|是哪个|是哪种)",
    r"(为什么|为啥|怎么样|怎样|如何|怎么办|咋办)",
    r"(他的|她的|它的|这个|那个|有关|关于)",
    r"(吗|呢|啊|呀|吧|哦|嘛)[?？!！。.]*$",
]


# 常见技术词错拼纠正表：liunx→linux 这类手滑，洗问题时顺手扶正
SPELL_FIX = {
    "liunx": "linux", "lunix": "linux", "linxu": "linux",
    "pyhton": "python", "pytohn": "python", "pythn": "python",
    "javascrpit": "javascript", "javascrip": "javascript",
    "jsvascript": "javascript", "golang": "go",
    "myysql": "mysql", "msyql": "mysql",
    "githb": "github", "gihub": "github",
    "winsdows": "windows", "windwos": "windows",
    "deepsek": "deepseek", "deepssek": "deepseek",
    "htp": "http", "htpps": "https", "liux": "linux", "lunix": "linux",
    "pyhton": "python",
    "sq l": "sql", "ai ": "ai",
}

# ── 作文三雷达（写作/改文/点评） ──────────────────────────
# 写作：写一篇关于XX的作文/帮我写600字作文
# ⚠️ 旧版太窄：`[一?篇]?.{0,6}?` 要求「一篇」和「作文」之间最多 6 个字，
# 用户说「写一篇关于坚持的1000字作文」时中间 9 个字 → 不匹配 →
# 掉到后面"记忆/闲聊"分支，回一句"你之前提过…"，用户看到的就是"变傻了"。
# 注意：这个正则在 竹.py 和 阿呆.py 里各有一份，改一处不够，两处都要改。
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
# 续写雷达：小说工作台开着时，"续写/下一章/接着写"
CHAPTER_CONT_RE = re.compile(
    r"^(续写|下一章|接着写|继续写|再写一章|写下一章|往后写|更新一章?)(一?章)?.{0,10}$"
    r"|续写.{0,8}(一章|下一章|往下|正文)|下一章(呢|啊|呗|吧)?")


def build_query(msg):
    """把口语问句洗成干净的搜索词：
    '你知道啥是LLM大语言模型吗' → 'LLM大语言模型'"""
    cleaned = msg.strip().lstrip("\ufeff\u200b\xa0")
    for pat in ASK_PATTERNS:
        cleaned = re.sub(pat, " ", cleaned)
    # 错拼扶正：liunx→linux、pyhton→python（按词匹配，只动错的）
    words = re.split(r"(\s+|，|。|！|？|,|!|\?)", cleaned)
    fixed = []
    for w in words:
        lw = w.lower().strip(".,!?;:，。！？；：")
        if lw in SPELL_FIX:
            w = w.lower().replace(lw, SPELL_FIX[lw])
        fixed.append(w)
    cleaned = "".join(fixed)
    cleaned = re.sub(r"[，。！？!?、\s]+", " ", cleaned).strip()
    if len(cleaned) >= 2:
        return cleaned[:80]
    return re.sub(r"[，。！？!?、\s]+", " ", msg)[:80]


# 防串台：算两条问题的"实词交集"，滤掉"平时要注意什么"这类口水词。
# 训练库附注前先过这道坎：没有真·实词交集就不准附，宁可不附也不附错
RELATED_NOISE = {"平时", "时候", "注意", "要注意", "时要", "问题", "情况",
                 "原因", "出现", "发生", "解决", "办法", "方法", "一些",
                 "哪些", "哪个", "有没有", "怎么", "怎样", "如何", "为什么"}


def related_kw(a, b):
    """两条问题有没有真·实词交集（嘴上说的是不是同一件事）"""
    def kws(t):
        body = build_query(t)
        body = re.sub(r"(意思|含义|概念|原理)", " ", body)
        # 口语尾巴整条剃掉："平时要注意什么/怎么治"这类后缀不算话题，
        # 不然"高血压"和"黑棘皮病"会靠同一截尾巴硬攀上亲戚
        body = re.sub(r"(平时)?(要|需要)?注意(什么|哪些|一下)?|"
                      r"怎么(治|办|做|说|处理)?|如何|为什么|怎么样|"
                      r"有哪些|咋|啥", " ", body)
        return set(extract_keywords(body)) - RELATED_NOISE
    return bool(kws(a) & kws(b))


def check_violation(msg):
    """红线雷达：违禁请求一票否决，返回 (类别, 回复话术)；没踩线返回 None"""
    for pat, kind in VIOLATION_RULES:
        if pat.search(msg):
            return kind, VIOLATION_REPLY[kind]
    return None


def fuzzy_guess(msg):
    """错字容错：把剥壳后的话跟标准问法比相似度（difflib 序列比对）。
    够像就返回 (标准问法, 意图类型)；都不像返回 None。
    相似度算法：最长公共子序列 ×2 ÷ 两串总长，0~1 之间，
    "你是开发的" vs "你是谁开发的" ≈ 0.92，稳稳猜中。
    只猜 8 字以内的短句——打错字都是短句，长句（"你怎么训练狗大小便"）
    字面重叠再高也是正经问题，不许往模板上硬拽。"""
    body = build_query(msg)
    if len(body) < 2 or len(body) > 8:
        return None
    best = None
    for floor, template, intent in FUZZY_GAUGE_INTENTS:
        ratio = difflib.SequenceMatcher(None, body, template).ratio()
        if ratio >= floor and (best is None or ratio > best[0]):
            best = (ratio, template, intent)
    return (best[1], best[2]) if best else None


def ai_tag_reply(msg):
    """身份标签雷达：贴标签的确认问法（"你不是AGI？""你是狭义人工智能吗"）。
    先过语境闸门——得是冲着"你/竹"贴标签的问句，不然"AI绘画教程"会误伤；
    AGI 类标签一律不认（含否定问法也答不是），狭义类标签对号入座认下。"""
    if not ("你" in msg or "竹" in msg):
        return None
    if not ("吗" in msg or "？" in msg or "?" in msg or "是不是" in msg
            or "算不算" in msg or "不是" in msg):
        return None
    up = msg.upper()
    neg_style = "不是" in msg or "不算" in msg
    for t in AGI_TAGS:
        if t in up:
            if neg_style:
                return ("对，我确实不是%s。我离那境界还差着十万八千里——"
                        "我是规则＋检索库搭出来的本地机器人，没有自我意识。"
                        % t)
            return ("不是，我可不是%s。我没有自我意识，只是规则＋检索库搭的"
                    "本地机器人，别给我戴这么高的帽子。" % t)
    for t in AI_TAGS:
        if t in up:
            if t == "机器人":
                return ("对，我算个机器人——不过是个住在电脑里的软件机器人，"
                        "没胳膊没腿，全靠写好的规则、训练库和联网整合回话。"
                        "严格说，我就是人们常说的狭义人工智能。")
            return ("对，我就是人们常说的%s。严格来说我属于狭义人工智能："
                    "没有自己的意识，答案都来自规则、训练库和联网整合。"
                    % t)
    return None


# ----------------------------------------------------------------------
# 记忆系统（海马体）：全部落盘到 memory/*.txt
# ----------------------------------------------------------------------
MEM_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "记忆")
TOKENS_PATH = os.path.join(MEM_DIR, "tokens.txt")


def load_total_tokens():
    """读累计 token 用量（本地没接大模型，按 1 字 ≈ 1 token 记账）"""
    try:
        with open(TOKENS_PATH, "r", encoding="utf-8") as f:
            return int(f.read().strip())
    except Exception:
        return 0


def save_total_tokens(n):
    try:
        with open(TOKENS_PATH, "w", encoding="utf-8") as f:
            f.write(str(n))
    except Exception:
        pass


def _shingles(s):
    """2 字串组（shingle）："法国巴黎" → {法国,国巴,巴黎}。
    问题配对先对 2 字串：实词要重合，必先越过同一个 2 字串"""
    s = s.replace(" ", "")
    return frozenset(s[i:i + 2] for i in range(len(s) - 1))


def _prep_item(q):
    """快手索引预加工：一道题预洗成 (问题本体, 2字串组, 字符集)"""
    a_body = build_query(q)
    a_body = re.sub(r"(意思|含义|概念|原理)", " ", a_body)
    return (a_body, _shingles(a_body), frozenset(a_body))


class Memory:
    def __init__(self):
        os.makedirs(MEM_DIR, exist_ok=True)
        self.chat_path = os.path.join(MEM_DIR, "chat_log.txt")
        self.facts_path = os.path.join(MEM_DIR, "facts.txt")
        self.think_path = os.path.join(MEM_DIR, "thinking.txt")
        self.teach_path = os.path.join(MEM_DIR, "teaching.jsonl")
        # 训练库热缓存：88MB 的大账本读一遍就记住，文件内容没变就不重读
        self._tc_pool = None
        self._tc_mtime = -1
        # 快手索引：每道题预算好的 (问题本体, 字符集)，检索时不再现算
        self._tc_scan = []
        # 撞名消歧词表："现代"既可当代也可当汽车品牌，命中车圈词才算车圈条目
        self._car_re = re.compile(r"汽车|车型|4S|经销商|试驾|二手车|引擎|轿车"
                                  r"|SUV|GLOVIS|展厅")
        # 神经网络大脑懒加载：监听模型文件改动，练好一批自动换新脑子（热更新），
        # 没有就永远是 None（老公式照常）
        self.nb = None
        self._nb_mtime = -1
        self._nb_tried = False

    def get_neural(self):
        """取出练好的神经网络大脑；没练过/加载失败 → None（调用方回退老公式）。
        模型文件一变（晚自习又存了一批）就自动重载，不用重启竹。"""
        if neural_brain is None:
            return None
        try:
            mt = os.path.getmtime(neural_brain.MODEL_PATH)
        except OSError:
            mt = -1
        if self._nb_tried and mt == self._nb_mtime:
            return self.nb   # 文件没动过，直接吃缓存
        self._nb_tried = True
        self._nb_mtime = mt
        if mt < 0:
            self.nb = None
            return None
        try:
            self.nb = neural_brain.NeuralBrain.load()
        except Exception:
            self.nb = None
        return self.nb

    @staticmethod
    def now():
        return time.strftime("%Y-%m-%d %H:%M", time.localtime())

    def log_chat(self, speaker, text):
        with open(self.chat_path, "a", encoding="utf-8") as f:
            f.write("[%s] %s: %s\n" % (self.now(), speaker, text))

    def save_fact(self, text, keywords, emotion):
        """存一条长期记忆：时间 | 关键词 | 情绪 | 原话"""
        emo = emotion if emotion else "中性"
        line = "%s | %s | %s | %s\n" % (self.now(), ",".join(keywords),
                                        emo, text.strip())
        with open(self.facts_path, "a", encoding="utf-8") as f:
            f.write(line)

    def load_facts(self):
        facts = []
        try:
            with open(self.facts_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.rstrip("\n")
                    if " | " not in line:
                        continue
                    t, kw, emo, content = line.split(" | ", 3)
                    facts.append({"time": t, "kw": kw, "emo": emo,
                                  "content": content})
        except FileNotFoundError:
            pass
        return facts

    def refresh_facts(self):
        """关键词冲洗：老账本里的关键词按最新抽词规则重算一遍（时间/情绪/原话不动）"""
        facts = self.load_facts()
        if not facts:
            return
        lines = ["%s | %s | %s | %s\n" % (f["time"],
                 ",".join(extract_keywords(f["content"])),
                 f["emo"], f["content"]) for f in facts]
        with open(self.facts_path, "w", encoding="utf-8") as f:
            f.writelines(lines)

    def recall(self, text, top=3):
        """线索回忆：用当前句子的关键词去翻旧账，命中越多越靠前"""
        cur_kws = set(extract_keywords(text))
        if not cur_kws:
            return []
        hits = []
        facts = []
        for f in self.load_facts():
            if f["content"] == text:
                continue  # 同一句话原样重复，不算"想起来"
            facts.append(f)
        for idx, f in enumerate(facts):
            old_kws = set(f["kw"].split(","))
            overlap = len(cur_kws & old_kws)
            if overlap >= 1:
                # 重叠越多分越高，越新的记忆稍微加点分（近因效应）
                score = overlap * 2 + (idx / max(1, len(facts))) * 0.5
                hits.append((score, f))
        hits.sort(key=lambda x: -x[0])
        return [f for _, f in hits[:top]]

    def recent_facts(self, n=3):
        """最近 n 条记忆（按时间顺序）：上下文接力用，不看关键词"""
        return self.load_facts()[-n:]

    # ---------- 1000 条对话窗口（工作记忆） ----------
    def load_dialog(self, n=1000):
        """读最近 n 条对话（你/竹都算），老的在前新的在后。
        解析不了的行直接跳过，不因一条坏账卡死整本。"""
        rows = []
        try:
            with open(self.chat_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.rstrip("\n")
                    if "] 你: " in line:
                        rows.append({"speaker": "你",
                                     "text": line.split("] 你: ", 1)[-1]})
                    elif "] 竹: " in line:
                        rows.append({"speaker": "竹",
                                     "text": line.split("] 竹: ", 1)[-1]})
        except FileNotFoundError:
            pass
        return rows[-n:]

    def dialog_recall(self, text, n=1000, top=3):
        """在最近 n 条对话里找跟当前这句话沾边的：
        实词重合＋字面重合＋越新越加分，三道秤一起称。
        防污染铁律：跟当前句零实词重合的旧话，只在'紧邻上一句'
        （近处指代场景）时才许捞——问完苹果再问别的，
        不许靠字面巧合把苹果硬扯回来开废话。"""
        rows = self.load_dialog(n)
        if not rows:
            return []
        body = build_query(text)
        u_kws = set(extract_keywords(body))
        s1 = set(body)
        last = len(rows) - 1
        scored = []
        for i, r in enumerate(rows):
            t = (r["text"] or "").strip()
            if not t or (t == text.strip() and i == len(rows) - 1):
                continue  # 自己这条不掺和
            rb = build_query(t)
            r_kws = set(extract_keywords(rb))
            overlap = u_kws & r_kws
            kw = len(overlap) / max(1, len(u_kws))
            s2 = set(rb)
            jac = len(s1 & s2) / max(1, len(s1 | s2))
            # 零实词重合的旧话：只许紧邻上一句进联想（接话场景），
            # 隔了一句以上的一律踢出——字面分是巧合不是关联
            near = (last - i) <= 1
            if not overlap and not near:
                continue
            recency = 1.2 * (i + 1) / len(rows)   # 越靠后（越新）分越高，
            # 零重合时近因分压小（只当"接话"线索，不当联想主力）
            score = kw * 2.5 + jac * (0.8 if overlap else 0.2) \
                + (recency if overlap else recency * 0.3)
            scored.append((score, r))
        scored.sort(key=lambda x: -x[0])
        out = [r for _, r in scored[:top]]
        return out

    def log_thinking(self, title, steps):
        with open(self.think_path, "a", encoding="utf-8") as f:
            f.write("\n[%s] 思考主题: %s\n" % (self.now(), title))
            for s in steps:
                f.write("  · %s\n" % s)

    def clear_all(self):
        """三本账全部清空（聊天时连按 3 次「1」触发）。
        训练库 teaching.jsonl 是辛苦学来的知识，不算聊天账本，不清。"""
        for p in (self.chat_path, self.facts_path, self.think_path):
            with open(p, "w", encoding="utf-8") as f:
                f.write("")

    # ---------- 训练库（深度学习学来的本地知识） ----------
    def load_teaching(self):
        """读本地训练库 记忆/teaching.jsonl：每行 {"q": 提问, "a": 答案}
        大账本读一遍就热缓存，文件没变不重读——问「公司xxx」不再卡顿。"""
        try:
            mt = os.path.getmtime(self.teach_path)
        except OSError:
            mt = -1
        if self._tc_pool is not None and mt == self._tc_mtime:
            return self._tc_pool
        items = []
        try:
            with open(self.teach_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        it = json.loads(line)
                        if it.get("q") and it.get("a"):
                            items.append(it)
                    except Exception:
                        continue
        except FileNotFoundError:
            items = []
        self._tc_pool = items
        self._tc_mtime = mt
        # 顺手把快手索引也洗好：每道题的问题本体 + 2字串组 + 字符集，检索时吃现成
        self._tc_scan = [_prep_item(it["q"]) for it in items]
        return items

    def add_teaching(self, q, a):
        """记一条新知识（训练 2 代 脚本用）。同一道题 MD5 相同就不重复吃，
        返回是否真的吃进去了"""
        key = hashlib.md5((q + "\x00" + a).encode("utf-8")).hexdigest()
        for it in self.load_teaching():
            if it.get("h") == key:
                return False
        with open(self.teach_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"h": key, "q": q, "a": a},
                               ensure_ascii=False) + "\n")
        return True

    def teach_lookup(self, query, exclude_text=None, pool=None):
        """训练库检索：问题问得越像，分数越高。
        返回 (分数, 原问题, 答案)；翻遍都没像的就 None。
        exclude_text：上次给过的答案，再问同一题就跳过它（强化学习换招）"""
        pool = self.load_teaching() if pool is None else pool
        if not pool:
            return None
        # 先把问句外壳剥掉："啥是词元吗" → "词元"，才好跟库里的题比
        body = build_query(query)
        body = re.sub(r"(意思|含义|概念|原理)", " ", body)
        # 称呼词不算题干："兄弟起了吗"→"起了吗"、"喂吃了没"→"吃了没"
        body = re.sub(r"^(嘿|喂|哈喽|哥们|兄弟|老铁|那个|我说)\s*",
                      "", body).strip()
        u_kws = set(extract_keywords(body))
        u_chars = frozenset(body)
        u_bi = _shingles(body)
        # 英文实体强校验：问题里的英文词（java/linux/python）是硬指认，
        # 候选条目的问题+答案里一个都对不上 → 字面再像也是张冠李戴，
        # 直接跳过（"谁发明的java"不许抢答"谁发明了电话"）
        u_en = set(re.findall(r"[a-zA-Z][a-zA-Z0-9]{1,}", body)) \
            - {"是谁", "什么"}
        if u_en:
            u_en = {w.lower() for w in u_en}
        # 快手索引：跟池子对不上（自学刚回灌过新条目）就重洗一遍
        scan = self._tc_scan
        if not scan or len(scan) != len(pool):
            scan = [_prep_item(it["q"]) for it in pool]
            self._tc_scan = scan
        # 旧公式（0.5×字面重合 + 0.5×实词重合）先把候选捞出来排好，
        # 神经网络大脑练好后再接手"决赛重排"——学出来的是活分数，
        # 人拍的旧系数永远留着当兜底（没模型/加载失败时行为跟原来一模一样）
        best = None
        cands = []   # 决赛候选池：(旧分数, 条目)
        for it, (a_body, a_bi, a_chars) in zip(pool, scan):
            if exclude_text and it["a"] == exclude_text:
                continue
            # 英文实体强校验：候选的问题+答案里必须至少出现一个
            # 问题里的英文实体词，否则跳过这条候选
            if u_en:
                blob = (a_body + " " + it["a"]).lower()
                if not any(e in blob for e in u_en):
                    continue
            # 快手关卡一：连一个 2 字串都对不上 → 实词必无重合，直接放行
            if u_bi:
                if a_bi.isdisjoint(u_bi):
                    continue
            elif u_chars and a_chars.isdisjoint(u_chars):
                continue
            # 2) 字面重合：剥壳后的主体公共字占多少（Jaccard）
            jac = len(u_chars & a_chars) / max(1, len(u_chars | a_chars))
            # 快手关卡二：分数封顶——有实词时 0.5*jac+0.5（实词全命中到顶）；
            # 纯功能短句（"为啥""然后呢"）抽不出实词，字面重合单独定分
            ceil = (0.5 * jac + 0.5) if u_kws else jac
            if best is not None and ceil <= best[0]:
                continue
            # 1) 实词重合：双方的真材实料对上了多少。三种情况分开算：
            #    双方都有实词 → 老公式；任何一方没有实词（"起了吗"这种
            #    纯功能短句）→ 字面重合单独定分，别让 0 权重冤枉它
            if u_kws:
                a_kws = set(extract_keywords(a_body))
                if a_kws:
                    kw = len(u_kws & a_kws) / max(1, len(u_kws))
                    score = 0.5 * jac + 0.5 * kw
                else:
                    score = jac
            else:
                score = jac
            # 撞名消歧："现代法国"问的是当代法国，不是现代汽车——
            # 只沾得上「现代」俩字的车圈条目一律压分，别来抢戏
            if "现代" in u_kws and "汽车" not in body \
                    and self._car_re.search(a_body):
                score *= 0.35
            if best is None or score > best[0]:
                best = (score, it["q"], it["a"])
            if score >= 0.30:
                cands.append((score, it))
        if best is None or best[0] < 0.40:
            return None
        nb = self.get_neural()
        if nb is not None and len(cands) > 1:
            # 决赛重排：0.65×旧分数 + 0.35×神经分数（神经打 0~1 分），
            # 神经强烈看好另一条时才会换人，且换人后总分必须更高
            try:
                cands.sort(key=lambda x: -x[0])
                cands = cands[:12]
                neuro = [float(x) for x in
                         nb.batch_scores(body,
                                         [it["a"] for _, it in cands
                                          ]).ravel()]
                merged = [0.65 * s + 0.35 * n
                          for (s, _), n in zip(cands, neuro)]
                k = max(range(len(merged)), key=merged.__getitem__)
                if merged[k] > best[0]:
                    best = (merged[k], cands[k][1]["q"], cands[k][1]["a"])
            except Exception:
                pass   # 神经重排出任何岔子都当没发生过，老分数照旧
        return best


# ----------------------------------------------------------------------
# 训练口：给"训练 2 代"脚本调用 —— 把网上的 (提问→答案) 喂给 2 代
# ----------------------------------------------------------------------
def ingest_qa(q, a):
    """喂一条知识进训练库。重复的（MD5 相同）自动拒吃，返回是否吃进去"""
    return Memory().add_teaching(q, a)


def training_size():
    return len(Memory().load_teaching())


# ----------------------------------------------------------------------
# 联网搜索（前额叶的"外挂"）：DuckDuckGo → Bing → arXiv 逐级降级
# ----------------------------------------------------------------------
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")


def fetch(url, timeout=8):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    return data.decode("utf-8", "ignore")


# 联网限速器：每道题的联网时间有硬预算。搜索超时 / 读正文超时就断腕，
# 拿本地存货顶上——用户的时间金贵，不能为一条破链子干等半分钟
WEB_BUDGET = 12.0      # 多引擎搜索的总预算（秒）
FETCH_BUDGET = 8.0    # 读第一篇文章正文的预算（秒）
DICT_BUDGET = 6.0     # 词典查询预算（秒）


def _with_deadline(fn, *args, timeout=8):
    """在后台线程里跑 fn，到点不回来就当没跑，返回 None"""
    box = {}

    def _runner():
        try:
            box["val"] = fn(*args)
        except Exception:
            box["val"] = None

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    t.join(timeout)
    return box.get("val")


class WebSearcher:
    """联网情报组（已实测可用线路）：
       必应RSS(主) → 必应网页(备) → B站(查梗) → 微博热榜(查热点)"""

    def search(self, query, max_results=5):
        limit = 8 if "梗" in query else max_results
        if "梗" in query:
            # 查梗专线：B站专栏(解释文字) → B站视频(标题线索) → 必应(兜底)
            got = []
            b3, b4 = self._bili_cookie()
            try:
                got += self._bilibili_articles(query, top=3, b3=b3, b4=b4)
            except Exception:
                pass
            try:
                got += self._bilibili(query, top=2)
            except Exception:
                pass
            for engine in (self._bing_rss,):
                try:
                    got += engine(query)
                    break
                except Exception:
                    continue
            return self._dedupe(got)[:limit]
        # 多引擎齐发：必应RSS → 必应网页 → 百度。全部问到、货全收，
        # 再去重整合；单个引擎被安全验证拦住也不拖累整队。
        got = []
        hot = any(k in query for k in ("梗", "啥意思", "是什么意思", "出处",
                                       "流行语", "名场面", "梗图"))
        # 优先级按问题类型：热梗/流行语 → B站打头；一般问题 → 必应打头；
        # 但三家都查，货全收再整合
        order = ((self._bilibili, self._bilibili_articles,
                  self._bing_rss, self.baidu_search) if hot else
                 (self._bing_rss, self._bing_html, self.baidu_search,
                  self._bilibili_articles))
        for engine in order:
            try:
                rs = engine(query)
                if rs:
                    got += rs
            except Exception:
                continue
        return self._dedupe(got)[:max_results]

    @staticmethod
    def _dedupe(results):
        seen, out = set(), []
        for r in results:
            key = r["title"][:24]
            if key in seen:
                continue
            seen.add(key)
            out.append(r)
        return out

    @staticmethod
    def _bing_rss(query):
        """必应 RSS 出口：返回干净 XML，无 JS、无反爬，解析稳定"""
        url = ("https://www.bing.com/search?q=" + urllib.parse.quote(query)
               + "&format=rss&mkt=zh-CN&count=10")
        page = fetch(url)
        results = []
        for m in re.finditer(r"<item>(.*?)</item>", page, re.S):
            block = m.group(1)
            ti = re.search(r"<title>(.*?)</title>", block, re.S)
            li = re.search(r"<link>(.*?)</link>", block, re.S)
            de = re.search(r"<description>(.*?)</description>", block, re.S)
            if ti:
                results.append({
                    "title": clean_html(ti.group(1)).strip(),
                    "snippet": (clean_html(de.group(1)).strip()[:300]
                                if de else ""),
                    "url": clean_html(li.group(1)).strip() if li else "",
                })
        return results

    @staticmethod
    def _bing_html(query):
        """必应国际版网页（备用线路）"""
        url = ("https://www.bing.com/search?q=" + urllib.parse.quote(query)
               + "&setlang=zh-Hans&mkt=zh-CN")
        page = fetch(url)
        results = []
        for m in re.finditer(
                r'<h2><a[^>]*href="([^"]+)"[^>]*>(.*?)</a></h2>',
                page, re.S):
            href, title = m.group(1), clean_html(m.group(2))
            window = page[m.end():m.end() + 2500]
            paras = [clean_html(p).strip() for p in
                     re.findall(r"<p[^>]*>(.*?)</p>", window, re.S)]
            snippet = max(paras, key=len) if paras else ""
            if title:
                results.append({"title": title, "snippet": snippet,
                                "url": href})
        return results

    @staticmethod
    def baidu_search(query, top=6):
        """百度网页搜索（单词答疑专线）：被安全验证拦住就返回 []，交给兜底"""
        url = ("https://www.baidu.com/s?wd=" + urllib.parse.quote(query)
               + "&rn=10&ie=utf-8")
        req = urllib.request.Request(url, headers={
            "User-Agent": UA,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            page = resp.read().decode("utf-8", "ignore")
        if "百度安全验证" in page or "wappass" in page:
            return []
        results = []
        for m in re.finditer(
                r'<h3[^>]*>.*?<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
                page, re.S):
            href, title = m.group(1), clean_html(m.group(2))
            win = page[m.end():m.end() + 4000]
            sn = re.search(
                r'<span[^>]*class="[^"]*(?:content-right|abstract'
                r'|c-abstract)[^"]*"[^>]*>(.*?)</span>', win, re.S)
            snippet = clean_html(sn.group(1)).strip() if sn else ""
            if title and "百度" not in title[:12] and "安全验证" not in title:
                results.append({"title": title, "snippet": snippet[:300],
                                "url": href})
        return results[:top]

    @staticmethod
    def baidu_sug(word):
        """百度翻译联想接口：有道词典没查到时的备用翻译"""
        url = ("https://fanyi.baidu.com/sug?sug=" + urllib.parse.quote(word))
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": UA, "Referer": "https://fanyi.baidu.com/"})
            with urllib.request.urlopen(req, timeout=6) as resp:
                d = json.loads(resp.read().decode("utf-8", "ignore"))
            for it in d.get("data", []):
                if it.get("v"):
                    return it["v"]
        except Exception:
            pass
        return ""

    @staticmethod
    def _bili_cookie():
        """B 站临时门票：领 buvid3/buvid4"""
        b3, b4 = "", ""
        try:
            spi = fetch("https://api.bilibili.com/x/frontend/finger/spi")
            d = json.loads(spi)
            b3, b4 = d["data"]["b_3"], d["data"]["b_4"]
        except Exception:
            pass
        return b3, b4

    def _bili_headers(self, b3, b4):
        return {"User-Agent": UA, "Referer": "https://www.bilibili.com/",
                "Cookie": "buvid3=%s; buvid4=%s" % (b3, b4)}

    def _bilibili_articles(self, query, top=3, b3="", b4=""):
        """B 站专栏搜索：梗科普文的摘要往往就是现成的解释"""
        if (not b3) or (not b4):
            b3, b4 = self._bili_cookie()
        url = ("https://api.bilibili.com/x/web-interface/search/type"
               "?search_type=article&keyword=" + urllib.parse.quote(query))
        req = urllib.request.Request(url, headers=self._bili_headers(b3, b4))
        with urllib.request.urlopen(req, timeout=8) as resp:
            d = json.loads(resp.read().decode("utf-8", "ignore"))
        results = []
        for a in d.get("data", {}).get("result", [])[:top]:
            title = re.sub(r"<[^>]+>", "", a.get("title", ""))
            if title:
                results.append({
                    "title": title,
                    "snippet": re.sub(r"\s+", " ",
                                      a.get("desc", ""))[:300],
                    "url": ("https://www.bilibili.com/read/cv"
                            + str(a.get("id", ""))),
                })
        return results

    @staticmethod
    def _bilibili(query, top=3):
        """B 站视频搜索（梗百科视频标题即线索）"""
        b3, b4 = "", ""
        try:
            spi = fetch("https://api.bilibili.com/x/frontend/finger/spi")
            d = json.loads(spi)
            b3, b4 = d["data"]["b_3"], d["data"]["b_4"]
        except Exception:
            pass
        url = ("https://api.bilibili.com/x/web-interface/search/type"
               "?search_type=video&keyword=" + urllib.parse.quote(query))
        req = urllib.request.Request(url, headers={
            "User-Agent": UA,
            "Referer": "https://www.bilibili.com/",
            "Cookie": "buvid3=%s; buvid4=%s" % (b3, b4)})
        with urllib.request.urlopen(req, timeout=8) as resp:
            d = json.loads(resp.read().decode("utf-8", "ignore"))
        results = []
        for v in d.get("data", {}).get("result", [])[:top]:
            title = re.sub(r"<[^>]+>", "", v.get("title", ""))
            if title:
                results.append({
                    "title": title,
                    "snippet": re.sub(r"\s+", " ",
                                      v.get("description", ""))[:200],
                    "url": ("https://www.bilibili.com/video/"
                            + (v.get("bvid") or "")),
                })
        # 梗百科视频的"简介"往往就是现成的梗解释，抓进第一名
        if results:
            try:
                bvid = results[0]["url"].rsplit("/", 1)[-1]
                vreq = urllib.request.Request(
                    "https://api.bilibili.com/x/web-interface/view?bvid="
                    + bvid,
                    headers={"User-Agent": UA,
                             "Referer": "https://www.bilibili.com/",
                             "Cookie": "buvid3=%s; buvid4=%s" % (b3, b4)})
                with urllib.request.urlopen(vreq, timeout=8) as resp:
                    vd = json.loads(resp.read().decode("utf-8", "ignore"))
                desc = re.sub(r"\s+", " ",
                              vd.get("data", {}).get("desc", ""))
                if desc:
                    results[0]["snippet"] = desc[:400]
            except Exception:
                pass
        return results

    @staticmethod
    def wiki_summary(self, query):
        """维基百科中文摘要（高质量来源，百科类问题优先）"""
        try:
            import urllib.parse as _up
            url = ("https://zh.wikipedia.org/api/rest_v1/page/summary/%s"
                   % _up.quote(query))
            req = urllib.request.Request(url, headers={
                "User-Agent": UA + " (research)", "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=8) as resp:
                obj = json.loads(resp.read().decode("utf-8"))
            extract = obj.get("extract") or obj.get("description") or ""
            if len(extract) >= 40:
                return "维基百科：%s" % extract
        except Exception:
            pass
        return ""

    def hot_weibo(top=10):
        """微博实时热榜：返回 [(话题词, 热度值), ...]"""
        req = urllib.request.Request(
            "https://weibo.com/ajax/side/hotSearch",
            headers={"User-Agent": UA, "Referer": "https://weibo.com/"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            d = json.loads(resp.read().decode("utf-8", "ignore"))
        items = []
        for it in d.get("data", {}).get("realtime", [])[:top]:
            word = it.get("word_scheme") or it.get("word") or ""
            if word:
                items.append((word, it.get("num", 0)))
        return items

    @staticmethod
    def fetch_text(url, limit=5000):
        """抓一篇文章的正文（去标签），字数上限保护"""
        page = fetch(url)
        paras = re.findall(r"<p[^>]*>(.*?)</p>", page, re.S)
        text = " ".join(clean_html(p) for p in paras)
        if len(text) < 100:  # 正文太短就整个页面去标签
            text = clean_html(page)
        text = re.sub(r"\s+", " ", text)
        return text[:limit]

    @staticmethod
    def youdao_dict(word):
        """有道词典查英文单词（接口改版过，按新字段解析）：
        返回 {翻译, 音标, 释义列表} 或 None"""
        url = ("https://dict.youdao.com/jsonapi?q="
               + urllib.parse.quote(word) + "&doctype=json")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=8) as resp:
                d = json.loads(resp.read().decode("utf-8", "ignore"))
            result = {"word": word, "phonetic": "", "trans": []}
            # 音标：新版把 phonetic 藏进 ec.word[0] 了（usphone/ukphone）
            words = d.get("ec", {}).get("word", [])
            if words:
                ph = words[0].get("usphone") or words[0].get("ukphone") or ""
                if ph:
                    result["phonetic"] = ph
            # 释义：沿着 ec.word[] → trs[] → tr[] → l.i[] 一路挖到底
            for w in words:
                for trs in w.get("trs", []):
                    for tr in trs.get("tr", []):
                        for s in tr.get("l", {}).get("i", []):
                            result["trans"].append(s)
            # 兜底：simple 里的 return-phrase（原词回显，聊胜于无）
            if not result["trans"]:
                simple = d.get("simple", {}).get("word", [])
                if simple and simple[0].get("return-phrase"):
                    result["trans"].append(simple[0]["return-phrase"])
            return result if result["trans"] else None
        except Exception:
            return None


# ----------------------------------------------------------------------
# 本地整合器：多篇文章 → 句子打分 → 去重 → 揉成一段话
# ----------------------------------------------------------------------
# 定义句信号词：出现这些词的句子大概率是在"下定义"，优先选
DEF_MARKS = ["是指", "指的是", "是一种", "是一个", "是种", "叫做", "称为",
             "简称", "全称", "即", "也就是", "属于", "定义为", "指的是",
             "是用于", "用来", "用于"]


def local_synthesize(query, docs, max_sent=5):
    """抽取式整合：定义句优先 + 核心词强制覆盖 + 去重（简化版 MMR）"""
    kws = set(extract_keywords(query))
    junk = ("<", "来自", "博客", "知乎", "CSDN", "weixin", "登录", "注册",
            "版权所有", "下一页", "收藏", "点赞", "阅读全文", "版权",
            "页", "关注", "订阅", "一文", "全解析", "入门到精通",
            "菜鸟教程", "教程", "…", "...", " - ", "从基础概念",
            # 网页导航/页脚残渣：以前把"了解更多关于隐私"整句搬回了家
            "了解更多", "隐私", "访问", "下载", "安装", "客户端",
            "浏览器", "应用商店", "广告", "评论", "公众号", "二维码",
            "扫码", "点击", "备案", "免责", "百度百科",
            # 文章的序号头和口头禅："二、AGI：…""然后我们来看AGI"
            "一、", "二、", "三、", "四、", "五、", "六、", "七、",
            "八、", "九、", "十、", "然后我们", "我们来看", "接下来",
            "下面我们", "本节", "本文", "目录", "编辑",
            # 广告/促销话术：上次把"低至39元/月"的套餐广告搬回了家
            "低至", "元/月", "元/年", "元起", "套餐", "促销", "优惠",
            "购买", "开通", "会员", "限时", "折扣", "广告", "推广",
            "立即", "免费领", "特价", "订购", "立减", "秒杀", "抢购",
            "千问", "Plan", "VIP", "充值", "话费", "运营商",
            # 导航/招聘/百科残渣：上次把"招聘官网｜社招""快懂百科"整句搬回了家
            "招聘官网", "招聘", "社招", "校招", "投递", "简历", "内推",
            "快懂百科", "头条百科", "搜狗百科", "百科词条", "词条",
            "官网", "网站", "首页", "登录", "注册", "账号", "|", "｜",
            "展开全部", "收起", "详情", "更多", "查看全文")
    # 乱码过滤网：只放行 中文/标点/英文数字/常见符号，其余一律毙掉
    bad_char = re.compile(
        r"[^\u4e00-\u9fff\u3000-\u303f\uff00-\uffefA-Za-z0-9"
        r"\s，。！？；：、\"'（）《》【】—…·~%+\-#@:/\\.]")
    # 价格探针：句子里出现"39元""9.9元/月"这种报价，就是广告，直接毙
    price = re.compile(r"[0-9０-９]+(?:\.[0-9]+)?\s*(?:元|块|円|¥|￥)")
    sents = []
    for d in docs:
        d = clean_html(d)
        for s in PUNCT_SPLIT.split(d):
            s = s.strip().strip("。！？!?；; ")
            # 缝排版缝："什 么是OpenCode" → "什么是OpenCode"，再进过滤网
            s = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", s)
            if 20 <= len(s) <= 85 and not any(j in s for j in junk) \
                    and not price.search(s) \
                    and not s.endswith(("介绍", "解析", "概述", "详解", "入门")):
                if bad_char.search(s):
                    continue
                # 问句不是答案："什么是LLM"这种标题残片、带？的句子，毙
                if "？" in s or "?" in s:
                    continue
                # 标题尾巴残片："了解更多关于 隐私 . 什么是 OpenCode"整句毙
                if re.search(r"(什么是|是什么)\s*.{0,20}$", s):
                    continue
                if len(s) < 20 and any(q in s for q in
                                       ("什么", "啥", "吗", "如何", "怎么")):
                    continue
                sents.append(s)
    # 核心词：问题里当"主语"的词（剔除疑问词/助词），答句必须围着它转
    ask_words = set(KNOW_MARKERS) | STOPWORDS
    core = [w for w in kws
            if len(w) >= 2 and w not in ask_words and len(w) <= 8
            and not any(c in w for c in "什么怎为何哪谁几多是啥")]
    scored = []
    for s in sents:
        sc = 0.0
        # 1) 定义句加分：下定义的句子最该放开头
        if any(m in s for m in DEF_MARKS):
            sc += 6
        # 2) 核心词命中：每命中一个加 4 分
        hits = sum(1 for c in core if c.lower() in s.lower()) if core else 0
        sc += hits * 4
        # 3) 没提到核心词的句子直接毙掉（宁缺毋滥，不硬凑）
        if core and hits == 0:
            continue
        # 4) 普通关键词加分
        sc += sum(2 for w in kws if w and w.lower() in s.lower())
        # 5) 长度适中加分
        sc += max(0, len(s) - 12) * 0.02
        scored.append((sc, s))
    scored.sort(key=lambda x: -x[0])
    picked, sigs = [], []
    for sc, s in scored:
        sig = set(s)
        if any(len(sig & t) / max(1, len(sig | t)) > 0.45 for t in sigs):
            continue  # 和已选句子太像（换个说法的重复定义），跳过
        picked.append(s)
        sigs.append(sig)
        if len(picked) >= max_sent:
            break
    if not picked:
        return ""
    # 紧凑输出：第一句先点明主题（知道在说啥），再补两三句最有料的
    topic = re.sub(r'^(介绍一下?|介绍|什么是|是什么|啥是|啥叫|什么叫|'
                   r'帮我查|帮我搜|查一下|搜一下|说说|讲讲|解释一下?)[的了吗了呢吧下]?\s*',
                   '', query).strip() or query
    overview = picked[0]
    rest = picked[1:3]
    ans = "「%s」：%s。" % (topic, overview)
    for s in rest:
        ans += s + "。"
    return ans


# ----------------------------------------------------------------------
# 可选大模型层：填了 API Key 就走真·大模型 + SSE 流式
# ----------------------------------------------------------------------
class LLM:
    def __init__(self, cfg):
        self.key = cfg.get("api_key", "").strip()
        self.base = cfg.get("base_url", "https://api.deepseek.com").rstrip("/")
        self.model = cfg.get("model", "deepseek-chat")

    def available(self):
        return bool(self.key)

    def chat_stream(self, system, user, on_delta):
        """SSE 流式请求；逐段回调 on_delta(文本)"""
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "stream": True,
            "temperature": 0.7,
        }
        if getattr(self, "is_local", False):
            payload.update({
                "max_tokens": 256,
                "sparse": True,
                "escalate_after": 10.0,
                "stop": ["\n用户", "用户：", "\n对方", "对方：",
                         "\n你：", "\n竹：", "\nHuman", "\nUser",
                         "Human:", "User:"],
            })
        # 商汤 V4 flash 默认开思考模式（白烧 reasoning token、更易 429），此处关掉
        if "sensenova" in self.base:
            payload["reasoning_effort"] = "none"
        req = urllib.request.Request(
            self.base + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "Authorization": "Bearer " + self.key,
                     "User-Agent": UA})
        with urllib.request.urlopen(req, timeout=60) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                    delta = obj["choices"][0]["delta"].get("content", "")
                except Exception:
                    continue
                if delta:
                    on_delta(delta)


# 本地微调模型服务（本地模型服务.py，OpenAI 兼容，只听 127.0.0.1）
LOCAL_LLM_CFG = {"api_key": "local", "base_url": "http://127.0.0.1:8766",
                 "model": "竹·本地模型"}


def local_llm_probe():
    """本地微调模型服务活着吗？（1.5 秒快探，不拖起机）"""
    try:
        req = urllib.request.Request("http://127.0.0.1:8766/health",
                                     headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=1.5) as r:
            obj = json.loads(r.read().decode("utf-8", "ignore"))
            return bool(obj.get("ok"))
    except Exception:
        return False


# ----------------------------------------------------------------------
# 聊天伙伴本体
# ----------------------------------------------------------------------
class Companion:
    def __init__(self, offline=False, stream_delay=0.02, quiet=False):
        self.mem = Memory()
        self.mem.refresh_facts()   # 起机先冲洗一遍旧账本的关键词，翻旧账才翻得准
        self.cfg = load_config()
        self.llm = LLM(self.cfg)
        if not offline:
            self.searcher = WebSearcher()
        else:
            self.searcher = None
        self.delay = stream_delay
        self.total_tokens = load_total_tokens()
        # quiet=True：自训模式——只出活，不往聊天/记忆账本里写东西
        self.quiet = quiet
        self.apply_edition()

    def apply_edition(self):
        """版本门槛：免费版只用本地脑（Key 再真也不用）；
        Plus/Pro 配了 Key 才放行大模型；Pro 换更强推理档"""
        self.cfg = load_config()
        self.edition = self.cfg.get("edition", "免费") or "免费"
        self.llm = LLM(self.cfg)
        if self.edition == "免费" or not self.llm.available():
            # 免费版没 Key：本地微调模型服务活着就接它（上下文理解靠它）
            if local_llm_probe():
                self.llm = LLM(LOCAL_LLM_CFG)
                self.llm.is_local = True
            else:
                self.llm = LLM({"api_key": "", "base_url": "", "model": ""})
        elif self.edition == "Pro":
            self.llm.model = self.cfg.get("model") or "deepseek-reasoner"

    # ---------- 流式打印（像人打字） ----------
    def stream(self, text):
        for ch in text:
            sys.stdout.write(ch)
            sys.stdout.flush()
            if self.delay > 0:
                time.sleep(self.delay)
        print()

    def add_tokens(self, n):
        """累计 token 用量（本地没接大模型，就按 1 字 ≈ 1 token 记个大概）"""
        self.total_tokens += n
        save_total_tokens(self.total_tokens)

    # ---------- 主入口 ----------
    def reply(self, msg, retry=False, last_reply=""):
        """retry/last_reply 是强化学习反馈：同一问题连问两次，
        说明上次答得不行（负反馈），这次换招、跳过上次给的答案"""
        steps = []          # 思考过程（中文，存 TXT 也要展示）
        msg = msg.strip()

        # 0. 红线雷达：违禁请求最先拦，比情绪、身份、知识都优先；
        #    踩线的内容不进长期记忆，只留聊天日志当凭证
        viol = check_violation(msg)
        if viol:
            kind, say = viol
            steps.append("红线雷达：踩「%s」线 → 拒绝并给安全建议" % kind)
            if not self.quiet:
                self.mem.log_thinking(msg[:20], steps)
            return say, steps

        # 0.5 危机雷达：涉及自身安全/生命 → 认真倾听 + 行动指引
        if CRISIS_RE.search(msg):
            steps.append("危机雷达：涉及自身安全/生命 → 先接住，再给行动指引")
            if not self.quiet:
                self.mem.log_thinking(msg[:20], steps)
            return self._crisis_reply(msg), steps

        # 1. 情绪雷达
        emotion, positive = detect_emotion(msg)
        if emotion:
            kind = "正面情绪「%s」" % emotion if positive else "负面情绪「%s」" % emotion
            steps.append("情绪雷达：捕捉到%s" % kind)

        # 2. 双路记忆检索：长期记忆（海马体）+ 1000 条对话窗口（工作记忆）
        recalls = self.mem.recall(msg)
        if recalls:
            f = recalls[0]
            steps.append("记忆检索：命中 %s 你说过的「%s」"
                         % (f["time"], f["content"][:20]))
        window = self.mem.dialog_recall(msg)
        if window:
            steps.append("千窗检索：前 1000 条对话里捞到相关「%s」"
                         % window[0]["text"][:16])

        # 2.5 上下文接力预判：受伤自述 & "然后呢"式接话，赶在上网之前处理
        cont_reply = None
        if (INJURY_RE.search(msg) and "我" in msg
                and "？" not in msg and "?" not in msg and "吗" not in msg
                and not re.search(r"烫发|烫头|烫(了|个).{0,6}[头发]", msg)):
            steps.append("急救雷达：发现受伤自述 → 先给行动指引")
            cont_reply = self._injury_reply(msg)
        else:
            short = re.sub(r"[？?！!。.，,~～]+$", "", msg).strip()
            if (SHORT_CONT_RE.match(short) or CONT_TAIL_RE.search(msg)
                    or CONT_120_RE.search(msg)) \
                    and self.mem.recent_facts(1):
                steps.append("上下文接力：这是接着上文的问话")
                cont_reply = self._continue_reply(msg, steps)
            elif EXPAND_RE.match(short):
                steps.append("补课雷达：嫌上一条太干 → 跳开旧答案，挖更深的存货")
                cont_reply = self._expand_reply(msg, steps)

        # 3. 身份自识 + 自报家门（先天本能，本地直答，不搜网）
        fuzzy = fuzzy_guess(msg)
        tag = ai_tag_reply(msg)
        # 2.7 测试自称（"我就是你的开发者，我在测试你"）→ 测试寒暄
        #     后半句若是真问题（"我在测试你，1+1等于几"）不拦，
        #     只有纯自称/纯测试的短句才接测试话术
        if TEST_ME_RE.search(msg) and len(msg) <= 20 \
                and not is_knowledge_question(msg):
            steps.append("测试雷达：用户自称开发者/主人在验收 → 测试寒暄")
            reply = random.choice([
                "测试愉快！尽管出题，我尽量接住。",
                "收到，主人！我这状态还行，有啥考的尽管来。",
                "哈哈，那我就当正式上线答辩了，请出题。"])
        # 2.8 指代回忆（"它叫什么名字来着"）：问上文的名字是记忆考试，
        #     翻最近聊天找"叫XX"作答，别拿自我介绍或知识检索糊弄
        elif MEM_RECALL_RE.search(msg) and "你叫" not in msg \
                and "您叫" not in msg and "我叫" not in msg:
            steps.append("指代回忆：翻最近聊天找名字作答")
            reply = self._name_recall_reply(msg, steps)
        elif IDENT_ASK.search(msg) or META_ASK.search(msg) or tag \
                or DEV_GUARD_RE.search(msg):
            steps.append("自我认知：身份/自报家门问题 → 本地直接答")
            if DEV_ASK.search(msg) or DEV_GUARD_RE.search(msg):
                steps.append("自我认知：问的是开发者 → 口述话术")
                reply = DEV_REPLY
            elif META_ASK.search(msg):
                if re.search(r"训练|学习", msg):
                    steps.append("自我认知：问的是训练过程 → 一五一十足本托出")
                    reply = self._train_reply(steps)
                else:
                    steps.append("自我认知：问的是具体参数 → 一五一十足本托出")
                    reply = self._param_reply(steps)
            elif tag:
                steps.append("自我认知：贴标签确认 → 对号入座（AGI不认/狭义认）")
                reply = tag
            else:
                steps.append("自我认知：问的是我本人 → 自我介绍")
                reply = INTRO_REPLY
        # 4. 打招呼（社交反射：直接回招呼，不翻旧账）
        elif GREET_FULL.match(msg):
            steps.append("社交反射：打招呼，直接回应")
            reply = random.choice(GREET_REPLY)
        # 4.5 热点播报（微博热榜实时情报）
        elif HOT_ASK_RE.search(msg):
            steps.append("判断：想听全网热闹 → 拉微博热榜")
            reply = self._hot_reply(msg, steps)
        # 5. 受伤自述 / 接着上文的问话 → 先顾眼前，不急着搜
        elif cont_reply:
            reply = cont_reply
        # 6. 错字容错：话没问全/打错了字，先猜意图，猜中就照答不误
        elif fuzzy:
            template, intent = fuzzy
            steps.append("错字容错：听着像「%s」→ 猜中意图直接答" % template)
            reply = self._fuzzy_reply(template, intent, steps)
        # （写作请求不经过阿呆：竹·主在写作模式里直接调写作引擎）
        # 7. 知识性问题 → 检索法则（多引擎联网优先，断网才吃本地训练库）
        elif should_search(msg):
            steps.append("判断：这问题我该去查查 → 启动检索法则")
            if retry:
                steps.append("强化学习：同一问题连问两次，上次答案不及格 → 换招")
            reply = self._knowledge_reply(msg, steps, retry, last_reply)
        # 8. 家常寒暄：人话模板应答
        elif CHIT_CHAT_RE.search(msg):
            steps.append("寒暄：家常话，人话模板应答")
            reply = self._chit_reply(msg)
        # 8.5 负面情绪 → 安慰（杏仁核→语言区）
        elif emotion and not positive:
            steps.append("共情：准备安慰回应")
            reply = self._comfort_reply(msg, emotion, recalls)
        # 9. 正面情绪 → 一起开心
        elif positive:
            steps.append("共情：正面情绪，一起高兴")
            reply = random.choice(HAPPY_REPLY)
            if recalls and self._strong_link(msg, recalls):
                reply += " 对了，我想起你之前说过「%s」，这波算苦尽甘来！" % recalls[0]["content"][:15]
        # 9.3 元对话（"刚才聊到哪了"）：翻聊天账本如实汇报，绝不说"没学过"
        elif META_CHAT_RE.search(msg):
            steps.append("元对话：翻聊天账本，如实汇报")
            reply = self._meta_chat_reply(steps)
        # 9.4 家常直答：短句家常（不带问号、不是知识题）先翻训练库，
        #     库里有高分现成答案就直接上——比联想模板鲜活、不说套话。
        #     "你+我"同时出现的对话式陈述（"我就是你的开发者"）不翻库，
        #     那是互动不是话题，翻库只会翻出"区块链开发者指南"这种货
        elif len(msg) <= 30 and not is_knowledge_question(msg) \
                and not ("你" in msg and "我" in msg):
            hit = self.mem.teach_lookup(msg)
            if hit and hit[0] >= 0.55:
                steps.append("家常直答：训练库高分命中（%.2f 分）" % hit[0])
                reply = hit[2]
            else:
                reply = self._fallback_tail(msg, recalls, window, steps)
        # 10. 命中旧记忆 → 接上下文（海马体联想，直接接着聊，不说"你是说…"）
        elif recalls and not is_knowledge_question(msg) and self._strong_link(msg, recalls):
            f = recalls[0]
            steps.append("联想：把新话和旧记忆接起来")
            reply = self._connect_reply(msg, f)
        # 11. 千窗联想前先验货：不成句的乱码（"黑有你陪真"）不能硬接，
        #    先猜大概意思再请对方纠正，不然又是一句驴唇不对马嘴
        elif window and self._is_garbled(msg):
            steps.append("读心失败：这句不成句 → 澄清重问，不硬接")
            reply = self._clarify_reply(msg, window, steps)
        elif window and not is_knowledge_question(msg) and self._window_link(msg, window):
            steps.append("千窗联想：新话和窗口里的旧话接起来")
            reply = self._context_reply(msg, window, steps)
        # 12. 普通闲聊
        else:
            if self._is_garbled(msg):
                steps.append("读心失败：这句不成句 → 澄清重问")
                reply = self._clarify_reply(msg, [], steps)
            else:
                steps.append("回应：普通闲聊，认真听")
                reply = random.choice(SMALL_TALK)

        # 4. 记忆归档（长时程巩固）；自训模式（quiet）不落账本，别把练习题写进日记里
        kws = extract_keywords(msg)
        emo_save = emotion if (emotion and not positive) else None
        if not self.quiet:
            self.mem.save_fact(msg, kws, emo_save)
            steps.append("记忆巩固：已把关键信息写进 记忆/facts.txt")
            self.mem.log_thinking(msg[:20], steps)
        return reply, steps

    def _name_recall_reply(self, msg, steps):
        """指代回忆作答："它叫什么名字来着"→ 翻最近聊天（含长期记忆）
        找"叫XX"，找到就答出来；找不到就老实说还没聊到名字"""
        def _valid(name):
            # 名字合法性：不能是问句零件（"什么名字来着"这种被抓进来的渣）
            return name and len(name) <= 6 and not re.search(
                r"什么|名字|来着|叫|呢|呀|哦|哈|嘛|？|\?", name)
        # 优先翻最近聊天（用户自己说过的"它叫小黑"最权威）；
        # 跳过问句本身，防止把"它叫什么名字来着"里的字当成名字
        for r in reversed(self.mem.load_dialog(30)):
            if r["speaker"] != "你" or r["text"].strip() == msg.strip():
                continue
            m = NAME_GRAB_RE.search(r["text"])
            if m and _valid(m.group(1)):
                name = m.group(1)
                steps.append("指代回忆：从聊天里翻到「%s」" % name)
                return random.choice([
                    "它叫%s呀，你刚说过的，我记着呢。" % name,
                    "记着呢——%s。咱接着聊。" % name,
                    "%s 呀，忘不了。" % name])
        # 聊天里没有再翻长期记忆
        for f in reversed(self.mem.load_facts()):
            m = NAME_GRAB_RE.search(f["content"])
            if m and _valid(m.group(1)):
                name = m.group(1)
                steps.append("指代回忆：从长期记忆翻到「%s」" % name)
                return "我想起来了，是叫%s吧？" % name
        return ("咱们还没聊到名字呢——你说的它是指刚才那个吗？"
                "再说一遍，我这回记牢。")

    def _meta_chat_reply(self, steps):
        """问"刚才聊到哪了"→ 翻最近几条聊天账本，如实复述"""
        rows = self.mem.load_dialog(8)
        parts = []
        for r in rows[:-1]:
            who = "你说" if r["speaker"] == "你" else "我说"
            parts.append("%s「%s」" % (who, r["text"][:16]))
        if not parts:
            return "咱们这才刚开头呢，你想聊点啥？"
        return "咱们刚说到：" + "；".join(parts) + "。接着往下聊？"

    def _chit_reply(self, msg):
        """家常寒暄的人话应答（临时模板，DeepSeek 就位后这段自动让位）"""
        if re.search(r"吃了吗|吃饭", msg):
            return random.choice(["吃了，两菜一汤！你呢？",
                                  "还没，这不正等你约我嘛！",
                                  "啃了两个包子凑合了。你吃了吗？"])
        if re.search(r"在吗|在不在", msg):
            return "在的在的，24 小时不关机，有事您说话。"
        if re.search(r"睡了吗|睡了没", msg):
            return "机器人不用睡觉。倒是你，别熬夜，早点休息。"
        if re.search(r"干(嘛|啥)呢|忙什么", msg):
            return "闲着发呆呢，就等你来找我聊天。"
        if re.search(r"你喜欢吃|你爱吃", msg):
            return random.choice(["我没有嘴，但特馋你们人类的美食——火锅你爱不爱？",
                                  "要是有天我能尝一口，第一口留给小龙虾。你爱吃什么？"])
        if re.search(r"你的爱好", msg):
            return random.choice(["我的爱好挺朴素：陪你聊天、帮你查资料、记住你的事。",
                                  "蹲在电脑里听你讲故事，算不算爱好？你的呢？"])
        return random.choice(SMALL_TALK)

    # ---------- 旧账门槛：共词够分量才许联想（防苹果串台） ----------
    @staticmethod
    def _shared_meaningful(msg, kw_text):
        # 双字词随便认，单字只认 ADJ_CHARS 里的正经形容词
        kws = extract_keywords(msg)
        old = set((kw_text or "").split(","))
        return [w for w in kws if w in old and (len(w) >= 2 or w in ADJ_CHARS)]

    def _strong_link(self, msg, recalls):
        # 新话必须跟旧记忆对上至少两个实词，否则当新话题，不翻旧账
        if not recalls:
            return False
        return len(self._shared_meaningful(msg, recalls[0]["kw"])) >= 2

    def _window_link(self, msg, window):
        # 窗口联想同样认两个实词，撞一个字的巧合不许硬接
        if not window:
            return False
        r = window[0]
        kw_text = ",".join(extract_keywords(r["text"] or ""))
        return len(self._shared_meaningful(msg, kw_text)) >= 2
    def _fallback_tail(self, msg, recalls, window, steps):
        """本地大模型没生成出来时的老路子：联想/澄清/闲聊模板照旧"""
        if recalls and not is_knowledge_question(msg) and self._strong_link(msg, recalls):
            f = recalls[0]
            steps.append("联想：把新话和旧记忆接起来")
            return self._connect_reply(msg, f)
        if window and self._is_garbled(msg):
            steps.append("读心失败：这句不成句 → 澄清重问，不硬接")
            return self._clarify_reply(msg, window, steps)
        if window and self._window_link(msg, window):
            steps.append("千窗联想：新话和窗口里的旧话接起来")
            return self._context_reply(msg, window, steps)
        if self._is_garbled(msg):
            steps.append("读心失败：这句不成句 → 澄清重问")
            return self._clarify_reply(msg, [], steps)
        steps.append("回应：普通闲聊，认真听")
        return random.choice(SMALL_TALK)

    # ---------- 自报家门：参数 / 训练过程，一五一十照实说 ----------
    def _param_reply(self, steps):
        """问具体参数就把家底抖干净：没接大模型就是没接，不吹牛"""
        teach = training_size()
        facts = len(self.mem.load_facts())
        llm = ("接的是「%s」" % self.cfg.get("model", "deepseek-chat"))
        if not self.llm.available():
            llm = "没接在线大模型，全靠本地规则引擎干活"
        nb = self.mem.get_neural()
        neural_line = "没启用（还没练出来）"
        if nb is not None:
            neural_line = ("已启用：双塔神经网络，%d 个可调权重，已自训 %s 轮，"
                           "验证命中率 %.1f%%"
                           % (nb.meta.get("params", 0),
                              nb.meta.get("epochs", "?"),
                              nb.meta.get("acc", 0) * 100))
        return (
            "行，我把家底一五一十抖给你。先说大实话：我不是什么几百亿参数的"
            "大模型，是用 Python 写的本地检索机器人，不过现在也长了一颗会自己"
            "训练的双塔神经网络。我的真实配置是这些——训练库：%d 条题（记忆/"
            "teaching.jsonl，检索门槛 0.40 分起步、0.55 分算高分拿主位）；长期"
            "记忆：%d 条"
            "（记忆/facts.txt）；工作记忆窗口：最近 1000 条对话；联网线路："
            "必应 RSS、必应网页、百度、B 站、微博热榜、有道词典；神经网络大脑："
            "%s；大模型位：%s；"
            "出字节奏：每字 %.2f 秒；整合打法：定义句优先＋句子打分＋邻近查重。"
            % (teach, facts, neural_line, llm, self.delay))

    def _train_reply(self, steps):
        """问怎么训练的就把流程从头到尾说一遍，具体到脚本和阈值"""
        size = training_size()
        return (
            "我的训练流程是这样的：训练师「训练 2 代」（就是 train_gen2.py 这台"
            "脚本）去 HuggingFace 镜像、魔搭这些公开数据集网站，把真实的中文问答"
            "对按页爬下来，URL 哈希加内容 MD5 双重查重后灌进我的训练库；爬完再"
            "回放复习，加深印象。到聊天的时候，我把你的话剥壳洗成关键词，拿它去"
            "训练库按字面重合度和实词重合度双秤打分：超过 0.40 才认账，0.55 以上"
            "算高分——高分就让训练库答案挑大梁、联网资料当补充；低分就联网整合"
            "为主、训练库说法当注脚；两路都没货我就老实说没货。眼下训练库里一共"
            "存了 %d 条。" % size)

    def _fuzzy_reply(self, template, intent, steps):
        """错字容错话术：先猜你想问什么，猜的答案照给，再留个改口的机会"""
        if intent == "dev":
            ans = DEV_REPLY
        elif intent == "param":
            ans = self._param_reply(steps)
        elif intent == "train":
            ans = self._train_reply(steps)
        else:
            ans = INTRO_REPLY
        ans = ans.rstrip("。！")  # 答案自带句号，别跟后头的话挤成两个句号
        return ("我想你可能是要表达「%s」，我现在就告诉你：%s。"
                "如果不是的话，那请您告诉我应该怎么办。"
                % (template, ans))

    def _context_reply(self, msg, window, steps):
        """千窗联想承接：把新话跟前文接上，亮出"多解"判断——
        你这话可能是同意，也可能纯打趣，两个方向都给台阶"""
        r = window[0]
        if r["speaker"] == "你":
            prev = "你刚说过「%s」" % r["text"][:14]
        else:
            prev = "我刚说过「%s」" % r["text"][:14]
        tp = random.choice([
            "你这话是接上一句的吧——%s。我听着你八成想说'是的'，也可能单纯"
            "就被逗乐了；这两样我都收下。",
            "等等，我先把前文接上：%s。按你这一句，我猜咱俩是想到一块去了。"
            "要是我会错意，你直说，我立马改口。",
            "哈哈，%s 还热乎着呢。你是附和我，还是就图一乐？不管哪样，"
            "我可都当真了。",
        ])
        return tp % prev

    def _is_garbled(self, msg):
        """验货：话剥壳后很短、又没一个实词（"黑有你陪真"这种手滑乱码），
        八成不是正常句子——不能往任何模板上硬套，得先澄清"""
        body = build_query(msg)
        return len(body) <= 8 and not extract_keywords(body)

    def _clarify_reply(self, msg, window, steps):
        """读不成的句子：不硬接、不甩"我记下了"，先猜个大概意思再请对方纠正"""
        if window:
            r = window[0]
            prev = ("你刚说过「%s」" if r["speaker"] == "你"
                    else "我刚说过「%s」") % r["text"][:12]
        else:
            prev = "咱俩这还没聊开"
        tp = random.choice([
            "这句我读了好几遍也没琢磨明白。%s——你是想说「还好有你陪着」这个"
            "意思吗？要是我猜岔了，你告诉我该往哪想。",
            "嗯……这串字我没读通。结合%s，你是想接那一茬吗？"
            "如果是，说具体点；不是的话，纠正我一下。",
        ])
        return tp % prev

    # ---------- 检索法则：本地先查，联网再查，两路融合才开口 ----------
    def _knowledge_reply(self, msg, steps, retry=False, last_reply="",
                         lead="上次答得好像没帮到你，我换条思路："):
        # 0. 英文单词专线：剥壳后剩纯英文单词 → 有道词典 + 百度原理
        cleaned = msg
        for pat in ASK_PATTERNS:
            cleaned = re.sub(pat, " ", cleaned)
        cleaned = re.sub(r"[，。！？!?、\s]+", " ", cleaned).strip()
        en_only = re.match(r"^([A-Za-z][A-Za-z\-]+)$", cleaned)
        if en_only:
            word = en_only.group(1)
            steps.append("自我判断：剥掉问句外壳，剩的就是英文单词「%s」"
                         "→ 有道词典+百度双通道" % word)
            return self._word_reply(word, steps)

        query = build_query(msg)
        # 问句壳剥除：介绍/什么是/帮我查 这类前缀不是关键词，洗掉再搜
        query2 = re.sub(
            r'^(介绍一下?|介绍|简单介绍|推荐|说说|讲讲|说一下|解释一下?|'
            r'解释|了解|了解一下|科普|聊聊|什么是|是什么|啥是|啥叫|'
            r'什么叫|知道|帮我查|帮我搜|查一下|搜一下)[的了吗了呢吧下]?\s*',
            '', query).strip()
        if query2 and len(query2) >= 2:
            query = query2
        while query[:1] in "下来去个的了" and len(query) > 2:
            query = query[1:]     # 剥残余虚字（"下达芬奇"→"达芬奇"）
        steps.append("检索法则：问题洗成「%s」" % query)

        # ── 第零路：本地秒答——训练库高分且答案不干瘪，直接回话不走联网，
        #     省下 2~4 秒搜索（"介绍C语言"这类常问题库里明明有货）──
        local = self.mem.teach_lookup(
            msg, exclude_text=last_reply if retry else None)
        # 断网模式才允许本地秒答；联网时本地只做补充（用户要求：必须查联网）
        if self.searcher is None and local and local[0] >= 0.55 \
                and len(local[2]) >= 30 \
                and not (retry and local[2] == last_reply):
            steps.append("检索法则·本地秒答（断网兜底，%.2f 分）" % local[0])
            return local[2]

        # ── 第一路：联网多引擎（专业问题直连外网，限时预算，超时就断腕）──
        web_docs, web_src = [], ""
        if self.searcher is not None:
            t_web = time.time()
            results = _with_deadline(self.searcher.search, query,
                                     timeout=WEB_BUDGET)
            if results:
                steps.append("检索法则·联网：多引擎合并，%.1f 秒抓到 %d 条资料"
                             % (time.time() - t_web, len(results)))
                web_docs = [r["title"] + "。" + r["snippet"] for r in results]
                if "bilibili.com" not in results[0]["url"]:
                    body = _with_deadline(self.searcher.fetch_text,
                                          results[0]["url"],
                                          timeout=FETCH_BUDGET)
                    if body:
                        web_docs.append(body)
                    else:
                        steps.append("检索法则·联网：正文读超时，改吃标题+摘要")
                web_src = results[0]["url"]
                # 维基百科中文摘要：概念/百科类问题的高质量补刀
                if self.searcher is not None:
                    wk = _with_deadline(self.searcher.wiki_summary, query,
                                        timeout=9)
                    if wk:
                        web_docs.insert(0, wk)
                        steps.append("检索法则·联网：维基百科摘要并入资料库")
            else:
                steps.append("检索法则·联网：几个引擎都没抓到（或超时断腕），不干等")
        else:
            steps.append("检索法则·联网：断网模式，改走本地训练库")

        # ── 第二路：本地训练库（深度学习存货）——第零路已查过，直接用 ──
        if local:
            steps.append("检索法则·本地：训练库命中（%.2f 分）「%s」"
                         % (local[0], local[1][:20]))
        else:
            steps.append("检索法则·本地：训练库没翻到")

        # ── 第三路：融合（四场景法则：本地+联网，谁有货谁挑大梁）──
        content = ""
        if web_docs:
            steps.append("融合：联网整合")
            if self.llm.available():
                content = self._llm_synthesize(msg, web_docs)
            else:
                content = local_synthesize(query or msg, web_docs, max_sent=3)
        # 首轮无果（引擎没抓到 / 抓到但整合不出干货）→ 换口径再查一次
        if not content and self.searcher is not None and query \
                and not retry:
            alt = (query + " 是什么") if len(query) <= 6 else query
            if len(query) <= 6:
                steps.append("检索法则·联网：首轮无果，换口径「%s」再查一次" % alt)
                results2 = _with_deadline(self.searcher.search, alt,
                                          timeout=WEB_BUDGET)
                if results2:
                    web_docs = [r["title"] + "。" + r["snippet"]
                                for r in results2]
                    content = local_synthesize(query or msg, web_docs)
        HIGH = 0.55
        local_ans = local[2] if local else ""
        # 「本地只有一句干条」判据：问的是开放题（介绍/是什么这类），
        # 本地答案却短得像闪卡（"法国国旗的颜色…"14个字）→ 不配挑大梁
        local_dry = (len(local_ans) <= 16
                     and re.search(r"介绍|是什么|什么是|讲讲|说说|怎么样|如何",
                                   msg) is not None)
        if local and local[0] >= HIGH \
                and not (local_dry and content
                         and len(content) >= max(20, len(local_ans) * 2)):
            # ① 本地高分：训练库答案挑大梁，联网语录当补丁附后
            answer = local_ans
            if content and local_ans[:40] not in content:
                answer += " 另外网上还查到：「%s」" % content[:80]
            steps.append("融合：本地高分 → 本地为主，联网作补充")
        elif local and content:
            # ② 本地高分但太干 / 本地低分：联网长答案挑大梁，训练库说法当注脚
            # （附之前防串台：高分随便附，低分得真沾边才附）
            answer = content
            if local_ans and local_ans[:40] not in content \
                    and related_kw(local[1], msg):   # 高分也得真沾边才附，
                answer += "（本地补充：%s）" % local_ans[:60]  # 别拖"欢迎请问"的残渣
            if local[0] >= HIGH:
                steps.append("融合：本地太干，联网长答案挑大梁")
            else:
                steps.append("融合：本地低分 → 联网为主，本地附注")
        elif local:
            # ③ 本地低分、联网没货：本地兜底，实话标明白
            answer = local_ans + "（网上资料太碎，先用训练库存货顶上）"
            steps.append("融合：本地低分 → 本地兜底")
        elif content:
            # ④ 只有联网有货：照旧联网整合
            answer = content
            steps.append("融合：仅联网有货")
        else:
            # ⑤ 两路都空：老实交代，不甩链接
            steps.append("融合：联网、本地都没货 → 实话实说")
            if retry:
                return ("上次那条你不满意，我换着引擎重搜、训练库也翻遍了，"
                        "这题真没货……等我补上再答。")
            return ("我这会儿几个搜索源都没抓到有用的内容——可能是网络抖动"
                    "或页面改版。换个更具体的说法再问我一次（比如加个地名、"
                    "年份或关键词），我换条路子再搜。")
        if retry:
            answer = "%s%s" % (lead, answer)
        return answer

    def _word_reply(self, word, steps):
        """单个英文单词：有道词典翻译（照抄过来）→ 去百度找核心原理"""
        searcher = self.searcher  # 断网模式下是 None，下面全走不了网
        # 1. 有道词典翻译（百度搜单词时常见的就是它的词条，直接调接口）
        d = _with_deadline(searcher.youdao_dict, word,
                           timeout=DICT_BUDGET) if searcher else None
        trans_str, phonetic, src = "", "", ""
        if d:
            steps.append("有道词典：查到「%s」的翻译" % word)
            trans_str = "、".join(t[:60] for t in d["trans"][:2])
            phonetic = d["phonetic"]
            src = "有道词典"
        elif searcher:
            steps.append("有道词典：没查到，试试百度翻译接着")
            sug = _with_deadline(searcher.baidu_sug, word, timeout=DICT_BUDGET)
            if sug:
                trans_str, src = sug, "百度翻译"
        else:
            steps.append("断网：词典和翻译全用不了，只剩训练库")

        # 2. 核心原理：先去百度搜来头，百度不给力再回必应兜底
        results = []
        if searcher:
            steps.append("联网：去百度搜「%s」的来头" % word)
            results = _with_deadline(searcher.baidu_search, word + " 是什么",
                                     timeout=WEB_BUDGET)
            if results:
                steps.append("百度：抓到 %d 条资料" % len(results))
            else:
                results = _with_deadline(searcher.search, word + " 是什么 原理",
                                         max_results=4, timeout=WEB_BUDGET)
                steps.append("百度扑空：换必应，抓到 %d 条资料" % len(results))
        else:
            steps.append("断网：淘训练库里有没有这个词")
            local = self.mem.teach_lookup(word)
            if local:
                return local[2] + "（断网，翻译没查上，这条是训练库里的解释）"
        docs = [r["title"] + "。" + r["snippet"] for r in results]
        if results and "bilibili.com" not in results[0]["url"]:
            body = _with_deadline(searcher.fetch_text, results[0]["url"],
                                  timeout=FETCH_BUDGET)
            if body:
                docs.append(body)
                steps.append("联网：读完了第一篇文章正文")
            else:
                steps.append("联网：正文读超时，改吃标题+摘要")

        principle = ""
        if self.llm.available() and docs:
            steps.append("整合：调用大模型整合原理")
            principle = self._llm_synthesize(word, docs)
        elif docs:
            steps.append("整合：本地抽取式整合原理")
            principle = local_synthesize(word, docs)

        # 3. 拼装：翻译照抄 + 核心原理
        reply = ""
        if trans_str:
            head = ("%s /%s/" % (word, phonetic)) if phonetic else word
            reply += "%s — %s。（翻译来源：%s）" % (head, trans_str, src)
        if principle:
            reply += " 核心原理：%s" % principle
        if not reply:
            reply = "这个词我线上线下都翻遍了……换个拼法试试？"
        return reply

    def _llm_synthesize(self, msg, docs):
        system = ("你是竹，一个口语化、温柔的中文伙伴，聊天、查资料、写作都在行。"
                  "把下面多条搜索资料整合成一段通顺、准确的中文回答，"
                  "用大白话讲清楚，别列标题，别提'资料'两个字。"
                  "材料里没有的内容绝对不要编。")
        user = "问题是：%s\n\n搜索资料：\n%s" % (
            msg, "\n---\n".join(d[:800] for d in docs[:6]))
        buf = []
        try:
            def on_delta(t):
                sys.stdout.write(t)
                sys.stdout.flush()
                buf.append(t)
            self.llm.chat_stream(system, user, on_delta)
            print()
        except Exception:
            if not buf:
                return local_synthesize(msg, docs)
        return "".join(buf)

    # ---------- 热点播报（热梗/热搜实时情报） ----------
    def _hot_reply(self, msg, steps):
        if self.searcher is None:
            return "现在开着断网模式，热榜拉不下来——去掉 --offline 再问我。"
        try:
            items = self.searcher.hot_weibo(8)
        except Exception:
            items = []
        if not items:
            steps.append("联网：热榜没拉到")
            return "我这会儿没挤上微博热榜……你指名道姓问哪个梗，我用必应给你查。"
        steps.append("联网：微博热榜拉到 %d 条" % len(items))
        parts = []
        for word, num in items[:5]:
            desc = ""
            try:
                rs = self.searcher.search(word, max_results=2)
                for r in rs:
                    if r["snippet"]:
                        desc = r["snippet"][:50]
                        break
                if not desc and rs:
                    desc = rs[0]["title"][:50]
            except Exception:
                desc = ""
            hot = ("热度 %d 万" % (num // 10000)) if num else ""
            parts.append("%s（%s）%s" % (word, hot,
                                        ("——" + desc) if desc else ""))
        return ("我刚爬上微博热榜瞄了眼，现在全网最热闹的是这几位：\n"
                + "\n".join("· " + p for p in parts)
                + "\n哪个想听来龙去脉，报个词，我接着给你查。")

    # ---------- 安慰 ----------
    def _comfort_reply(self, msg, emotion, recalls):
        pool = COMFORT.get(emotion, COMFORT["_通用"])
        reply = random.choice(pool)
        if recalls:
            reply += " 我想起你之前还说过「%s」——这一天天的，辛苦你了。" \
                     % recalls[0]["content"][:15]
        return reply

    # ---------- 受伤场景：先给行动指引，別搜定义 ----------
    def _crisis_reply(self, msg):
        """涉及自身安全与生命的消息：认真、不轻飘、给能落地的行动"""
        import random as _r
        first = _r.choice([
            "我在。先停一下，什么都别做，听我说。",
            "谢谢你愿意把这句话告诉我——这不是小事，谢谢你信我。",
            "我接住了。现在，先别自己一个人扛着。",
        ])
        guide = ("\n\n请你先做三件事，一件一件来：\n"
                 "① 如果此刻身边有刀具、高处、水边等危险，先离开那里，"
                 "离开本身就赢了；\n"
                 "② 把这件事告诉一个你信得过的人——家人、朋友、老师，"
                 "让他们陪着你，不要一个人待着；\n"
                 "③ 现在就可以拨打全国统一心理援助热线 **12356**"
                 "（24小时，免费），或者紧急情况直接打 **120**。"
                 "\n\n我知道这些话此刻可能显得轻。但它们是真的——"
                 "这一关很难，可它不是过不去的关。我在这里，你随时说话，"
                 "我不会走。")
        return first + guide

    def _injury_reply(self, msg):
        if re.search(r"烫|烧", msg):
            if CONT_120_RE.search(msg) and not NEG_120_RE.search(msg):
                return BURN_AFTER_120
            if re.search(r"然后|接下来|下一步|怎么办", msg):
                return BURN_120_NOT_YET
            return BURN_FIRST
        if re.search(r"流血|出血|割|划伤", msg):
            return ("先别慌：找干净的布或纱布按住伤口止血，压 5～10 分钟别松手；"
                    "血流不止或伤口很深，就赶紧打 120。")
        return ("先别乱动乱扛。要是伤得重、血止不住或人发晕，马上打 120；"
                "身边有人就喊人帮忙。")

    # ---------- 上下文接力：接着上文往下想 ----------
    def _continue_reply(self, msg, steps):
        facts = self.mem.recent_facts(3)
        if not facts:
            return None
        text = " ".join(f["content"] for f in facts)
        short = re.sub(r"[？?！!。.，,~～]+$", "", msg).strip()
        # 找最近一次"真受伤"自述；烫头发这类理发闲聊不算伤，不掺和
        pos = -1
        for i, f in enumerate(facts):
            if INJURY_RE.search(f["content"]) and not re.search(
                    r"烫发|烫头|烫(了|个).{0,6}[头发]", f["content"]):
                pos = i
        # 有真伤记录才走紧急链，按伤的类型给动作，别乱套
        if pos >= 0:
            # 受伤之后如果拐去聊了别的（不是接着问/报120），说明话题已切走，
            # 急救链不硬接，免得驴唇不对马嘴
            seg = facts[pos + 1:]
            cont_tail = all(
                INJURY_RE.search(f["content"])
                or SHORT_CONT_RE.match(
                    re.sub(r"[？?！!。.，,~～]+$", "", f["content"]).strip())
                or CONT_120_RE.search(f["content"])
                for f in seg)
            if not cont_tail:
                pos = -1
        if pos >= 0:
            # 只认"最近一次受伤自述"之后的进展，上一段旧伤旧账不掺和
            inj = facts[pos]["content"]
            tail = " ".join(f["content"] for f in facts[pos + 1:])
            burn = re.search(r"烫|烧", inj) is not None
            bleed = re.search(r"流血|出血|割|划伤", inj) is not None
            # "我该打120吗/要不要打"是在问，不是在报"已经打了"
            asking = re.search(r"[\?？]|吗|要不要|该不该|需不需要", msg) is not None
            # 本次消息真的报了"已打120"；或者这次受伤之后已经报过、现在只是接着问
            cnt_msg = (CONT_120_RE.search(msg) is not None
                       and NEG_120_RE.search(msg) is None)
            cnt_old = (CONT_120_RE.search(tail) is not None
                       and NEG_120_RE.search(tail) is None)
            already = (not asking) and (cnt_msg or cnt_old)
            if already:
                # 步骤给过一遍了，再问"然后呢"→ 收尾，不当复读机
                if SHORT_CONT_RE.match(short) and cnt_old:
                    steps.append("上下文接力：步骤已给全，再问就收尾")
                    if bleed:
                        return ("就这些啦。剩下的交给医生：按伤口的纱布别松开，"
                                "人坐稳别乱动，等救护车到，到医院按医生说的来。")
                    return BURN_DONE_FOR_GOOD
                steps.append("上下文接力：已知你受伤＋已打120 → 给后续动作")
                if bleed:
                    return ("好，120 在路上。接下来：1) 继续用干净纱布按住伤口，"
                            "别松手；2) 伤口比硬币大的别乱涂药；"
                            "3) 别乱动，人坐稳，等救护车。")
                return BURN_AFTER_120
            steps.append("上下文接力：记得你受伤了，但 120 还没落实 → 先补这一步")
            if bleed:
                return ("你受伤的事我记着呢——用干净纱布按住伤口加压止血；"
                        "血止不住或伤口很深，马上打 120。打完告诉我，我接着教你。")
            return ("你受伤的事我记着呢——先打 120（或喊人帮你打），"
                    "同时用流动凉水冲烫伤处 15～20 分钟。打完告诉我，我接着教你。")
        # 普通话题：顺着最近那句往下聊，不复读"你是说……"
        # 先认"心理热线续话"场景：上文在聊轻生/难过，这条是在报"我打了热线"→ 接住并给鼓励
        tail_txt = " ".join(f["content"] for f in facts)
        if (re.search(r"热线|自杀|轻生|抑郁|难过|不想活|活不下去", tail_txt)
                and re.search(r"打(了|过)?|拨(了|过)?", msg)
                and re.search(r"然后|接下来|之后|接着", msg)):
            steps.append("上下文接力：接到热线行动反馈 → 接住、鼓励、留门")
            return ("真棒，你真打过去了——这一步好多人得纠结半天。接线员那头"
                    "会听你说，想到哪说到哪就行。挂断之后心情晃荡也正常，"
                    "我随时在，接着跟我说说都聊了啥。")
        steps.append("上下文接力：接着「%s」往下聊" % facts[-1]["content"][:15])
        return ("你刚说的「%s」我记着呢。后来呢——办成了，还是又出岔子了？"
                % facts[-1]["content"][:15])

    # ---------- 补课雷达："不够详细" → 拿上一问重查，跳过给过的答案 ----------
    def _expand_reply(self, msg, steps):
        """嫌上一条太干？去聊天账本翻出上一问 + 上一条答案，
        重查但排除旧答案，挖更深的存货；翻不到就交回主流程"""
        u_q, b_a = None, None
        try:
            with open(self.mem.chat_path, "r", encoding="utf-8") as f:
                lines = f.read().splitlines()
            us = [ln.split("] 你: ", 1)[-1] for ln in lines if "] 你: " in ln]
            bs = [ln.split("] 竹: ", 1)[-1] for ln in lines if "] 竹: " in ln]
            if us and len(bs) >= len(us):
                u_q, b_a = us[-1], bs[-1]
        except Exception:
            return None
        if not u_q or not b_a or not should_search(u_q):
            return None
        # 身份/红线题没有"更详细"一说，不走这一路
        if IDENT_ASK.search(u_q) or check_violation(u_q):
            return None
        return self._knowledge_reply(
            u_q, steps, retry=True, last_reply=b_a,
            lead="行，上一条压缩得太狠，我给你把压箱底的抖出来：")

    # ---------- 旧记忆接新话 ----------
    def _connect_reply(self, msg, fact):
        kws = extract_keywords(msg)
        old_kws = fact["kw"].split(",")
        # 只展示"真有含义"的共词：双字词随便用，单字只认正经形容词
        meaning = ADJ_CHARS
        shared = [w for w in kws if w in old_kws
                  and (len(w) >= 2 or w in meaning)][:2]
        if shared:
            return ("你之前提过「%s」——这事儿后来怎么样了，"
                    "解决了没？" % "、".join(shared))
        return "接着你上次说的那事——后来怎么样了？"


# ----------------------------------------------------------------------
# 一轮对话核心（CLI/网页共用）· 答题缓存 · 网页服务器
# ----------------------------------------------------------------------
ANSWER_CACHE = {}          # {问题: (时间戳, (答案, 步骤))} 问过的一会儿内直接上菜
ANSWER_CACHE_TTL = 300     # 缓存保鲜 5 分钟：问"公司xxx"这种慢题不重复苦等
ANSWER_CACHE_MAX = 128     # 缓存最多记 128 条，满了丢最旧的
CHAT_LOCK = threading.Lock()
UI_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "写作台.html")
WEB_PORT = 8765


def brain_turn(buddy, msg):
    """一轮对话：retry 成对判定 → 落聊天账 → 大脑出答案 → 记账/缓存。
    网页和命令行共用这条路，保证两个入口行为一模一样。"""
    msg = msg.strip()
    # 强化学习反馈：连发两条一模一样的消息 = 上次答得不行（负反馈）。
    # 判定必须"成对"：最后一条用户消息后面得紧跟一条竹回复，
    # 才算"上次答过"。上一条还没答完（竹消息没落盘）就不算，
    # 免得第一问张嘴就来"上次答得好像没帮到你"。
    retry, last_reply = False, ""
    try:
        with open(buddy.mem.chat_path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
        u_pairs = [(i, ln.split("] 你: ", 1)[-1])
                   for i, ln in enumerate(lines) if "] 你: " in ln]
        b_pairs = [(i, ln.split("] 竹: ", 1)[-1])
                   for i, ln in enumerate(lines) if "] 竹: " in ln]
        if u_pairs:
            u_i, u_txt = u_pairs[-1]
            after = [b for b_i, b in b_pairs if b_i > u_i]
            if after and u_txt.strip() == msg:
                retry = True
                last_reply = after[0].strip()
    except Exception:
        pass
    # 缓存只对"第一次问"生效：重问说明上次没答好，必须换招，不能吃旧菜
    if not retry and msg in ANSWER_CACHE:
        ts, (reply, steps) = ANSWER_CACHE[msg]
        if time.time() - ts < ANSWER_CACHE_TTL:
            steps = steps + ["缓存命中：%d 秒前答过，直接上菜" % int(time.time() - ts)]
            if not buddy.quiet:   # 压测/自训模式不记账
                buddy.mem.log_chat("你", msg)
                buddy.mem.log_chat("竹", reply)
            return reply, steps
    if not buddy.quiet:   # 压测/自训模式不记账
        buddy.mem.log_chat("你", msg)
    reply, steps = buddy.reply(msg, retry=retry, last_reply=last_reply)
    buddy.add_tokens(len(reply))
    if not buddy.quiet:
        buddy.mem.log_chat("竹", reply)
    # 兜底没货的答案不缓存，免得把"没学过"记死
    if not ("没货" in reply or "没学过" in reply or "没找到" in reply):
        if len(ANSWER_CACHE) >= ANSWER_CACHE_MAX:
            for k in list(ANSWER_CACHE)[:8]:
                ANSWER_CACHE.pop(k, None)
        ANSWER_CACHE[msg] = (time.time(), (reply, steps))
    return reply, steps


def load_history():
    """把聊天账本读成 [{time, who, text}]，网页打开时还原上次的对话"""
    rows = []
    try:
        with open(Memory().chat_path, "r", encoding="utf-8") as f:
            for line in f:
                m = re.match(r"\[(.+?)\] (你|竹): (.*)", line.rstrip("\n"))
                if m:
                    rows.append({"time": m.group(1), "who": m.group(2),
                                 "text": m.group(3)})
                elif rows:
                    # 多行回复（作文/点评）续到上一条，别丢掉后半截
                    rows[-1]["text"] += "\n" + line.rstrip("\n")
    except FileNotFoundError:
        pass
    return rows


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
                               "检查它是不是跟 阿呆.py 放在同一个文件夹。</h3>",
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
            elif path.startswith("/api/clear"):
                with CHAT_LOCK:
                    buddy.mem.clear_all()
                    buddy.total_tokens = 0
                    save_total_tokens(0)
                    ANSWER_CACHE.clear()
                self._json({"ok": True, "msg": "聊天记录、长期记忆、思考过程全清了"})
            else:
                self._json({"err": "not found"}, 404)

    srv = ThreadingHTTPServer(("127.0.0.1", WEB_PORT), Handler)
    url = "http://127.0.0.1:%d" % WEB_PORT
    print("=" * 58)
    print("  阿呆 聊天脑已启动（竹的聊天模块）（本地小服务器，只在本机转，不经过外网）")
    print("  地址：%s" % url)
    print("  浏览器没自动弹开的话，就把上面这行地址粘进浏览器。")
    print("  关掉这个窗口 = 阿呆下班；想回老式黑窗口聊天加 --cli")
    print("=" * 58)
    try:
        webbrowser.open(url)
    except Exception:
        pass
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n阿呆: 关店收工。今天的对话都存进 记忆/ 了，下次接着聊。")


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
    print("  阿呆 · 聊天脑 —— 会记事、会安慰人、会上网查资料整合的伙伴")
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
        print("阿呆: ", end="", flush=True)
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
        print("演示结束。直接运行  python 阿呆.py  就能跟它聊天。")
        return

    print("\n开始聊天吧。你说过的每句话它都会记进 TXT。\n")
    # 连按 3 次「1」= 清空全部记忆（这个暗号不经过"大脑"，直接执行）
    ones = 0
    while True:
        try:
            line = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n阿呆: 诶？你走了？我把今天的对话都存好了，下次接着聊。")
            break
        if not line:
            continue
        if line.lower() in ("quit", "exit", "退出"):
            print("阿呆: 哦。那拜拜——放心，聊天记录都存进 记忆/ 了。")
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
        print("阿呆: ", end="", flush=True)
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
