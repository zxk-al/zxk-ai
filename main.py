import numpy as np
import pandas as pd
import jieba
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from collections import Counter
import os
import warnings
warnings.filterwarnings("ignore")

# ====================== 1. 全局配置 ======================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VOCAB_SIZE = 8000       # 初始占位，后续动态更新
SEQ_LEN = 64            # 文本固定长度
EMB_DIM = 128           # 词向量维度
BATCH_SIZE = 64
EPOCHS = 15
CLASS_NUM = 10          # CNews 10个新闻分类

# CNews标签映射
label_map = {
    0:"体育",1:"娱乐",2:"家居",3:"房产",4:"教育",
    5:"时政",6:"财经",7:"科技",8:"时尚",9:"游戏"
}
str_to_label = {v: k for k, v in label_map.items()}

# ====================== 2. 数据加载 ======================
def load_cnews_txt(file_path):
    texts = []
    labels = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            label_str, text = line.split("\t", 1)
            texts.append(text)
            labels.append(label_str)
    return texts, labels

# 数据集路径
data_dir = r"E:\PythonProject\data\cnews"
train_path = os.path.join(data_dir, "cnews.train.txt")
val_path = os.path.join(data_dir, "cnews.val.txt")
test_path = os.path.join(data_dir, "cnews.test.txt")

# 读取数据
train_texts, train_labels_str = load_cnews_txt(train_path)
val_texts, val_labels_str = load_cnews_txt(val_path)
test_texts, test_labels_str = load_cnews_txt(test_path)

# 字符串标签 → 数字标签
train_labels = [str_to_label[label] for label in train_labels_str]
val_labels = [str_to_label[label] for label in val_labels_str]
test_labels = [str_to_label[label] for label in test_labels_str]

print("✅ CNews 数据集加载成功！")

# ====================== 3. 分词 & 构建词表(动态更新VOCAB_SIZE) ======================
def text_cut(text):
    return jieba.lcut(text)

# 只用训练集构建词表
words = []
for text in train_texts:
    words.extend(text_cut(text))

word_count = Counter(words)
# 过滤低频词
vocab = {word:i+1 for i, (word, count) in enumerate(word_count.items()) if count >= 5}
vocab["<PAD>"] = 0
# 关键：更新为真实词表大小
VOCAB_SIZE = len(vocab)
print(f"✅ 词表构建完成，词表大小：{VOCAB_SIZE}")

# 文本转为数字序列（修复版，防止词ID越界）
def text2idx(text, vocab, seq_len=64):
    words = text_cut(text)
    idx = []
    for word in words:
        word_id = vocab.get(word, 0)
        # 强制限制词ID 范围
        if word_id >= VOCAB_SIZE:
            word_id = 0
        idx.append(word_id)
    if len(idx) < seq_len:
        idx += [0] * (seq_len - len(idx))
    else:
        idx = idx[:seq_len]
    return idx

x_train = np.array([text2idx(text, vocab, SEQ_LEN) for text in train_texts])
x_val = np.array([text2idx(text, vocab, SEQ_LEN) for text in val_texts])
x_test = np.array([text2idx(text, vocab, SEQ_LEN) for text in test_texts])

print("✅ 文本序列转换完成，序列长度：", SEQ_LEN)
print(f"训练集：{len(train_texts)} 条")
print(f"验证集：{len(val_texts)} 条")
print(f"测试集：{len(test_texts)} 条")
print(f"类别数：{len(label_map)} 个")

# ====================== 4. 数据集 & 数据加载器 ======================
class NewsDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = torch.LongTensor(texts)
        self.labels = torch.LongTensor(labels)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)

train_dataset = NewsDataset(x_train, train_labels)
val_dataset = NewsDataset(x_val, val_labels)
test_dataset = NewsDataset(x_test, test_labels)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("✅ DataLoader 创建成功！")

# ====================== 5. CNN+LSTM 模型（修复版） ======================
class CNNLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, EMB_DIM, padding_idx=0)
        # 关键：使用 padding="same"，让输出长度自动和输入保持一致
        self.conv3 = nn.Conv1d(EMB_DIM, 64, kernel_size=3, padding="same")
        self.conv4 = nn.Conv1d(EMB_DIM, 64, kernel_size=4, padding="same")
        self.conv5 = nn.Conv1d(EMB_DIM, 64, kernel_size=5, padding="same")
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(64*3, 128, batch_first=True, bidirectional=False)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(128, CLASS_NUM)

    def forward(self, x):
        emb_out = self.emb(x)  # [B, L, EMB]
        emb_t = emb_out.permute(0, 2, 1)  # [B, EMB, L]

        # 三个卷积层的输出长度现在都等于输入长度 L=64
        c3 = self.relu(self.conv3(emb_t)).permute(0, 2, 1)  # [B, 64, 64]
        c4 = self.relu(self.conv4(emb_t)).permute(0, 2, 1)  # [B, 64, 64]
        c5 = self.relu(self.conv5(emb_t)).permute(0, 2, 1)  # [B, 64, 64]
        c_cat = torch.cat([c3, c4, c5], dim=-1)  # [B, 64, 192]

        lstm_out, (hn, _) = self.lstm(c_cat)
        last_h = hn.squeeze(0)  # [B, 128]

        out = self.drop(last_h)
        out = self.fc(out)
        return out

model = CNNLSTM().to(DEVICE)
loss_fn = nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters(), lr=1e-3)

# ====================== 6. 训练 & 测试函数 ======================
def train_epoch():
    model.train()
    total_loss = 0.0
    for bx, by in train_loader:
        bx, by = bx.to(DEVICE), by.to(DEVICE)
        pred = model(bx)
        loss = loss_fn(pred, by)

        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += loss.item()
    return total_loss / len(train_loader)

def test_epoch():
    model.eval()
    total_acc = 0
    total_num = 0
    with torch.no_grad():
        for bx, by in test_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            pred = model(bx)
            pred_idx = torch.argmax(pred, dim=-1)
            total_acc += (pred_idx == by).sum().item()
            total_num += bx.size(0)
    return total_acc / total_num

# 开始训练
print("\n===== 开始训练 CNN+LSTM 新闻分类 =====")
for e in range(EPOCHS):
    train_loss = train_epoch()
    test_acc = test_epoch()
    print(f"Epoch{e+1:2d} | Loss:{train_loss:.4f} | TestAcc:{test_acc:.4f}")

# ===================== 7. 热点挖掘模块（TF-IDF + DBSCAN聚类） ======================
"""
思路：
1. TF-IDF向量化新闻文本
2. DBSCAN聚类：同一簇 = 同一个热点事件
3. 统计簇样本数量，数量越大 = 热门热点
"""
print("\n===== 热点挖掘分析 =====")

# 选取测试集前2000条新闻做热点分析
sample_size = 2000
sample_texts_raw = test_texts[:sample_size]
sample_labels = test_labels[:sample_size]

# 逐条分词，拼接为空格分隔字符串
cut_texts = [" ".join(text_cut(t)) for t in sample_texts_raw]

# TF-IDF 向量化
vec = TfidfVectorizer(max_features=3000)
tfidf_data = vec.fit_transform(cut_texts)

# DBSCAN 聚类
db = DBSCAN(eps=0.8, min_samples=5)
cluster_label = db.fit_predict(tfidf_data)

# 统计每个聚类数量
cluster_cnt = Counter(cluster_label)
# -1 代表噪声、零散新闻
hot_cluster = sorted([k for k in cluster_cnt.keys() if k != -1],
                     key=lambda x: cluster_cnt[x], reverse=True)

# 输出 TOP5 热点
topN = 5
print(f"TOP {topN} 热点事件（簇内新闻条数）：")
for idx, hot_id in enumerate(hot_cluster[:topN]):
    hot_texts = [sample_texts_raw[i] for i, c in enumerate(cluster_label) if c == hot_id]
    cls_idx_list = [sample_labels[i] for i, c in enumerate(cluster_label) if c == hot_id]
    # 获取该簇主要分类
    cls_name = Counter([label_map[i] for i in cls_idx_list]).most_common(1)[0][0]

    print(f"热点{idx+1} | 所属类目：{cls_name} | 新闻条数：{cluster_cnt[hot_id]}")
    print("示例新闻：", hot_texts[0])
    print("-" * 60)

print("\n🎉 全部任务执行完毕：分类训练 + 热点挖掘完成！")

# 在热点挖掘模块的后面，加上这三行
torch.save(model.state_dict(), "cnn_lstm_model.pth")
torch.save(vocab, "vocab.pth")
with open("vocab_size.txt", "w", encoding="utf-8") as f:
    f.write(str(VOCAB_SIZE))
print("✅ 模型、词表、词表大小已保存，可用于网页端！")