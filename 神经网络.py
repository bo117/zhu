# -*- coding: utf-8 -*-
"""
竹的神经网络大脑（双塔语义模型）
================================================================
跟 GPT 那种"千亿参数全靠 GPU"的大模型没法比，但这是一颗真的会自己
学习的神经网络：约 200 万个可调权重（神经元），吃 记忆/teaching.jsonl
里的 5.8 万条问答，用对比学习 + Adam 优化器一轮一轮自己调参数。

结构（双塔）：
  问题塔 / 答案塔 各一只"眼睛"：
    中文 2 字块 + 英文数字词 → 哈希进 16384 槽词汇表
    → 64 维嵌入加和（词袋） → L2 归一 → 128 隐层(ReLU) → 32 维目光向量
  相似度 = 两只眼睛目光的余弦对齐（点积 ÷ 温度），再套 sigmoid 成 0~1 分

训练（InfoNCE 对比学习）：
  正样本 = 每题配它自己的答案；负样本 = 同一批里其他题的答案
  目标：让"对的答案"分数永远压过"别人的答案"
  优化器：手写 Adam，每轮全库洗牌批量过一遍

文件：
  记忆/neural_brain.npz       权重
  记忆/neural_brain.meta.json 元数据（训练轮数/命中率/参数规模）
"""
import json
import os
import re
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MEM_DIR = os.path.join(HERE, "记忆")
MODEL_PATH = os.path.join(MEM_DIR, "neural_brain.npz")
META_PATH = os.path.join(MEM_DIR, "neural_brain.meta.json")

# ---------- 超参数（神经元规模的"出厂规格"） ----------
VOCAB = 16384     # 词汇表槽位（2字块+英文词哈希进去）
EMB = 64          # 每个词的嵌入维度
HID = 128         # 隐层神经元数（每个塔一层）
OUT = 32          # 目光向量维度
TAU = 0.07        # 温度：控制分数分布的锐度
MAX_FEATS = 300   # 每条文本最多吃多少个特征（防超长答案拖慢）
L2 = 1e-6         # 权重衰减系数（防过拟合）

_CJK = re.compile(r"[\u4e00-\u9fff]+")
_LAT = re.compile(r"[a-z0-9][a-z0-9\-]{0,15}")


def param_count():
    """数一数这颗脑子有多少个可调数字（给用户看的参数规模）"""
    n = VOCAB * EMB * 2          # 两个塔各自的嵌入表
    n += 2 * (EMB * HID + HID)   # 两个塔的 64→128 全连接
    n += 2 * (HID * OUT + OUT)   # 两个塔的 128→32 全连接
    return n


def _hash_feat(s, vocab):
    """字符串 → 词汇表槽位号（自己写的哈希，不依赖 Python 随机种子）"""
    h = 0
    for ch in s:
        h = (h * 131 + ord(ch)) & 0x7fffffff
    return h % vocab


def feat_ids(text, vocab=VOCAB, max_feats=MAX_FEATS):
    """把一段中文拆成特征编号列表：中文 2 字块为主，英文/数字词为辅"""
    t = text.lower()
    ids = []
    for seg in _CJK.findall(t):
        for i in range(len(seg) - 1):
            ids.append(_hash_feat(seg[i:i + 2], vocab))
            if len(ids) >= max_feats:
                return ids
    for w in _LAT.findall(t):
        ids.append(_hash_feat(w, vocab))
        if len(ids) >= max_feats:
            break
    return ids


def _norm_rows(m):
    return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-8)


def _encode(ids_list, E, W1, b1, W2, b2):
    """一个塔的前向：特征编号 → 嵌入加和 → 归一 → 隐层 → 目光向量。
    返回 (目光向量V, 归一化词袋X, 隐层H, 原始词袋X_raw, 原始目光V_raw)。"""
    n = len(ids_list)
    X_raw = np.zeros((n, E.shape[1]), dtype=np.float32)
    for i, ids in enumerate(ids_list):
        if ids:
            X_raw[i] = E[ids].sum(axis=0)
    X = _norm_rows(X_raw)
    H = np.maximum(0.0, X @ W1 + b1)
    V_raw = H @ W2 + b2
    V = _norm_rows(V_raw)
    return V, X, H, X_raw, V_raw


def _renorm_back(d, raw, normalized=None):
    """Backpropagation through row-wise L2 normalization."""
    if normalized is None:
        normalized = _norm_rows(raw)
    norm = np.linalg.norm(raw, axis=1, keepdims=True)
    inv = np.where(norm > 1e-8, 1.0 / (norm + 1e-8), 0.0)
    return (d - normalized * (d * normalized).sum(1, keepdims=True)) * inv


class Adam:
    """手写 Adam 优化器（动量 + 自适应步长），每个参数矩阵配套一份状态"""

    def __init__(self, lr=3e-3):
        self.lr, self.t, self.m, self.v = lr, 0, {}, {}

    def step(self, key, w, dw):
        self.t += 1
        m = self.m.get(key)
        if m is None:
            self.m[key] = np.zeros_like(w)
            self.v[key] = np.zeros_like(w)
            m = self.m[key]
        v = self.v[key]
        m *= 0.9
        m += 0.1 * dw
        v *= 0.999
        v += 0.001 * (dw * dw)
        mh = m / (1 - 0.9 ** self.t)
        vh = v / (1 - 0.999 ** self.t)
        w -= self.lr * mh / (np.sqrt(vh) + 1e-8)


class NeuralBrain:
    """双塔语义模型：练成后负责给 (问题, 候选答案) 打 0~1 相关分。
    规模可配：默认 212 万权重的小脑；传 big 参数可造 1.3 亿权重的
    '大块头'（词汇表 131072 × 512 维嵌入，隐层 2048）——同一套算法，
    神经元的数量级不同。"""

    def __init__(self, vocab=VOCAB, emb=EMB, hid=HID, out=OUT,
                 tau=TAU, l2=L2, max_feats=MAX_FEATS):
        self.vocab = vocab
        self.emb = emb
        self.hid = hid
        self.out = out
        self.tau = tau
        self.l2 = l2
        self.max_feats = max_feats
        rng = np.random.default_rng(42)
        self.Eq = (rng.standard_normal((vocab, emb)) * 0.02).astype(np.float32)
        self.Ea = (rng.standard_normal((vocab, emb)) * 0.02).astype(np.float32)
        self.W1q = (rng.standard_normal((emb, hid)) * 0.1).astype(np.float32)
        self.b1q = np.zeros(hid, np.float32)
        self.W2q = (rng.standard_normal((hid, out)) * 0.1).astype(np.float32)
        self.b2q = np.zeros(out, np.float32)
        self.W1a = (rng.standard_normal((emb, hid)) * 0.1).astype(np.float32)
        self.b1a = np.zeros(hid, np.float32)
        self.W2a = (rng.standard_normal((hid, out)) * 0.1).astype(np.float32)
        self.b2a = np.zeros(out, np.float32)
        self.meta = {"epochs": 0, "seen": 0, "acc": 0.0}

    def pcount(self):
        """数一数这颗脑子有多少个可调数字"""
        n = self.vocab * self.emb * 2
        n += 2 * (self.emb * self.hid + self.hid)
        n += 2 * (self.hid * self.out + self.out)
        return n

    # ---------------- 前向 / 打分 ----------------
    def query_vec(self, text):
        """一条问题 → 目光向量"""
        V, _, _, _, _ = _encode(
            [feat_ids(text, self.vocab, self.max_feats)],
            self.Eq, self.W1q, self.b1q, self.W2q, self.b2q)
        return V[0]

    def batch_scores(self, q_text, a_texts):
        """一条问题 vs 一批候选答案 → numpy 数组（0~1 相关分，批量算图省时）"""
        if not a_texts:
            return np.zeros(0, np.float32)
        qv, _, _, _, _ = _encode(
            [feat_ids(q_text, self.vocab, self.max_feats)],
            self.Eq, self.W1q, self.b1q, self.W2q, self.b2q)
        av, _, _, _, _ = _encode(
            [feat_ids(t, self.vocab, self.max_feats) for t in a_texts],
            self.Ea, self.W1a, self.b1a, self.W2a, self.b2a)
        sim = (qv @ av.T) / self.tau          # 目光对齐（余弦 × 1/温度）
        return 1.0 / (1.0 + np.exp(-sim))     # sigmoid 压到 0~1

    def score(self, q_text, a_text):
        return float(self.batch_scores(q_text, [a_text]).ravel()[0])

    # ---------------- 训练 ----------------
    def _batch_backward(self, dQ, dA, Xq, Hq, Xa, Ha,
                        Xq_raw, Xa_raw, Vq_raw, Va_raw, opt):
        """两塔的反向传播（梯度全部手推）"""
        # 答案塔：32→128→64
        dV_a = _renorm_back(dA, Va_raw)
        dW2a = Ha.T @ dV_a
        db2a = dV_a.sum(0)
        dH_a = dV_a @ self.W2a.T * (Ha > 0)
        dX_a = dH_a @ self.W1a.T
        dx_a = _renorm_back(dX_a, Xa_raw, Xa)
        # 问题塔：32→128→64
        dV_q = _renorm_back(dQ, Vq_raw)
        dW2q = Hq.T @ dV_q
        db2q = dV_q.sum(0)
        dH_q = dV_q @ self.W2q.T * (Hq > 0)
        dX_q = dH_q @ self.W1q.T
        dx_q = _renorm_back(dX_q, Xq_raw, Xq)
        # 64→嵌入
        dW1a = Xa.T @ dH_a
        db1a = dH_a.sum(0)
        dW1q = Xq.T @ dH_q
        db1q = dH_q.sum(0)
        # 权重衰减加进梯度
        for key, dw, w in (("W2a", dW2a, self.W2a), ("W2q", dW2q, self.W2q),
                           ("W1a", dW1a, self.W1a), ("W1q", dW1q, self.W1q)):
            dw += 2 * self.l2 * w
            opt.step(key, w, dw)
        for key, db, b in (("b2a", db2a, self.b2a), ("b2q", db2q, self.b2q),
                           ("b1a", db1a, self.b1a), ("b1q", db1q, self.b1q)):
            opt.step(key, b, db)
        # 嵌入表的更新不在这里：词袋梯度要按特征编号摊回每个词，由调用方散射
        return dx_q, dx_a

    def _scatter_emb(self, dx, ids_list, E, lr):
        """词袋梯度按特征编号摊回嵌入表，直接小步下山（SGD）。"""
        for i, ids in enumerate(ids_list):
            if ids:
                np.add.at(E, ids, -lr * dx[i])

    def train(self, qa_pairs, epochs=6, batch=128, lr=3e-3, seed=42,
              quiet=False, log=None):
        """主训练循环：全库洗牌 → 分批 → 前向算分数 → 反向调权重"""
        def say(s):
            if not quiet:
                print(s)
            if log is not None:
                log(s)
        pair_ids = [(feat_ids(q, self.vocab, self.max_feats),
                     feat_ids(a, self.vocab, self.max_feats))
                    for q, a in qa_pairs]
        n = len(pair_ids)
        rng = np.random.default_rng(seed)
        opt = Adam(lr)
        t0 = time.time()
        for ep in range(1, epochs + 1):
            order = rng.permutation(n)
            total = 0.0
            nbatch = 0
            for start in range(0, n, batch):
                idx = order[start:start + batch]
                q_ids = [pair_ids[i][0] for i in idx]
                a_ids = [pair_ids[i][1] for i in idx]
                m = len(idx)
                Q, Xq, Hq, Xq_raw, Vq_raw = _encode(
                    q_ids, self.Eq, self.W1q, self.b1q, self.W2q, self.b2q)
                A, Xa, Ha, Xa_raw, Va_raw = _encode(
                    a_ids, self.Ea, self.W1a, self.b1a, self.W2a, self.b2a)
                logits = (Q @ A.T) / self.tau          # [m×m]，对角=正样本
                p = np.exp(logits - logits.max(1, keepdims=True))
                p /= p.sum(1, keepdims=True)
                loss = -np.log(p[np.arange(m), np.arange(m)] + 1e-9).mean()
                total += float(loss)
                nbatch += 1
                eye = np.eye(m, dtype=np.float32)
                dS = (p - eye) / m / self.tau       # softmax 交叉熵的反向
                dQ = dS @ A
                dA = dS.T @ Q
                dx_q, dx_a = self._batch_backward(
                    dQ, dA, Xq, Hq, Xa, Ha, Xq_raw, Xa_raw,
                    Vq_raw, Va_raw, opt)
                self._scatter_emb(dx_q, q_ids, self.Eq, lr)
                self._scatter_emb(dx_a, a_ids, self.Ea, lr)
            acc = self.eval_accuracy(qa_pairs, quiet=True)
            self.meta = {"epochs": ep, "seen": n * ep, "acc": acc,
                         "params": self.pcount(),
                         "trained_at": time.strftime("%Y-%m-%d %H:%M")}
            say("第 %d/%d 轮 → 平均损失 %.4f | 验证命中率 %.1f%% | 用时 %.1f 分"
                % (ep, epochs, total / max(1, nbatch),
                   acc * 100, (time.time() - t0) / 60))
        return self.meta

    def eval_accuracy(self, qa_pairs, sample=400, negs=4, quiet=False):
        """验证：随机抽题，1 个真答案 + 4 个假答案，看真答案能不能排第一。
        全凭瞎蒙的水平是 20%，练好的模型应远超"""
        n = len(qa_pairs)
        if n < 2:
            return 0.0
        sample = min(max(1, sample), n)
        negs = min(max(0, negs), n - 1)
        rng = np.random.default_rng(7)
        idx = rng.choice(n, sample, replace=False)
        hits = 0
        for i in idx:
            q, a = qa_pairs[i]
            candidates = np.delete(np.arange(n), i)
            neg = rng.choice(candidates, size=negs, replace=False).tolist()
            cands = [a] + [qa_pairs[j][1] for j in neg]
            sc = self.batch_scores(q, cands)
            if int(np.argmax(sc)) == 0:
                hits += 1
        acc = hits / sample
        if not quiet:
            print("验证：%d 题中 %d 题真答案排第一（%.1f%%，瞎蒙≈%.0f%%）"
                  % (sample, hits, acc * 100, 100 / (negs + 1)))
        return acc

    # ---------------- 存取 ----------------
    def save(self, path=None, meta_path=None):
        path = path or MODEL_PATH
        meta_path = meta_path or META_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, Eq=self.Eq, Ea=self.Ea,
                 W1q=self.W1q, b1q=self.b1q, W2q=self.W2q, b2q=self.b2q,
                 W1a=self.W1a, b1a=self.b1a, W2a=self.W2a, b2a=self.b2a)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.meta, f, ensure_ascii=False)

    @classmethod
    def load(cls, path=None, meta_path=None):
        """读权重文件 → 模型（按权重形状自动还原规模）；
        没练过/文件坏了 → None（调用方回退老规则）"""
        path = path or MODEL_PATH
        meta_path = meta_path or META_PATH
        if not os.path.exists(path):
            return None
        try:
            d = np.load(path)
            m = cls(vocab=int(d["Eq"].shape[0]), emb=int(d["Eq"].shape[1]),
                    hid=int(d["W1q"].shape[1]), out=int(d["W2q"].shape[1]))
            for k in ("Eq", "Ea", "W1q", "b1q", "W2q", "b2q",
                      "W1a", "b1a", "W2a", "b2a"):
                setattr(m, k, d[k])
            if os.path.exists(meta_path):
                with open(meta_path, "r", encoding="utf-8") as f:
                    m.meta = json.load(f)
            return m
        except Exception:
            return None


def load_pairs(max_items=None):
    """读训练库 记忆/teaching.jsonl → [(问题, 答案)] 列表"""
    path = os.path.join(MEM_DIR, "teaching.jsonl")
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                it = json.loads(line)
                q = (it.get("q") or "").strip()
                a = (it.get("a") or "").strip()
                if q and a:
                    pairs.append((q, a))
            except Exception:
                continue
            if max_items and len(pairs) >= max_items:
                break
    return pairs
