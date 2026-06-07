import streamlit as st
import torch
import jieba
import numpy as np
import os
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN

# ===================== 全局配置 (和main.py保持一致) =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN = 64
EMB_DIM = 128
CLASS_NUM = 10

# 标签映射
label_map = {
    0:"体育",1:"娱乐",2:"家居",3:"房产",4:"教育",
    5:"时政",6:"财经",7:"科技",8:"时尚",9:"游戏"
}

# ===================== 1. 定义 CNNLSTM 模型 (和训练代码完全一致) =====================
class CNNLSTM(torch.nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab_size, EMB_DIM, padding_idx=0)
        self.conv3 = torch.nn.Conv1d(EMB_DIM, 64, kernel_size=3, padding="same")
        self.conv4 = torch.nn.Conv1d(EMB_DIM, 64, kernel_size=4, padding="same")
        self.conv5 = torch.nn.Conv1d(EMB_DIM, 64, kernel_size=5, padding="same")
        self.relu = torch.nn.ReLU()
        self.lstm = torch.nn.LSTM(64*3, 128, batch_first=True, bidirectional=False)
        self.drop = torch.nn.Dropout(0.3)
        self.fc = torch.nn.Linear(128, CLASS_NUM)

    def forward(self, x):
        emb_out = self.emb(x)
        emb_t = emb_out.permute(0, 2, 1)

        c3 = self.relu(self.conv3(emb_t)).permute(0, 2, 1)
        c4 = self.relu(self.conv4(emb_t)).permute(0, 2, 1)
        c5 = self.relu(self.conv5(emb_t)).permute(0, 2, 1)
        c_cat = torch.cat([c3, c4, c5], dim=-1)

        lstm_out, (hn, _) = self.lstm(c_cat)
        last_h = hn.squeeze(0)

        out = self.drop(last_h)
        out = self.fc(out)
        return out

# ===================== 2. 工具函数 =====================
def text_cut(text):
    """分词"""
    return jieba.lcut(text)

def text2idx(text, vocab, vocab_size, seq_len=64):
    """文本转数字序列（修复版：传入vocab_size参数）"""
    words = text_cut(text)
    idx = []
    for word in words:
        word_id = vocab.get(word, 0)
        if word_id >= vocab_size:
            word_id = 0
        idx.append(word_id)
    if len(idx) < seq_len:
        idx += [0] * (seq_len - len(idx))
    else:
        idx = idx[:seq_len]
    return idx

# ===================== 3. 加载模型、词表、词表大小 (缓存加速) =====================
@st.cache_resource
def load_all_resources():
    # 读取词表大小
    with open("vocab_size.txt", "r", encoding="utf-8") as f:
        vocab_size = int(f.read().strip())

    # 加载词表
    vocab = torch.load("vocab.pth", map_location="cpu")

    # 初始化模型 + 加载权重
    model = CNNLSTM(vocab_size=vocab_size).to(DEVICE)
    model.load_state_dict(torch.load("cnn_lstm_model.pth", map_location=DEVICE))
    model.eval()

    return model, vocab, vocab_size

# ===================== 4. 单条文本预测函数 =====================
def predict_text(model, vocab, vocab_size, text):
    model.eval()
    with torch.no_grad():
        # 文本转序列（传入vocab_size）
        idx_arr = text2idx(text, vocab, vocab_size, SEQ_LEN)
        x = torch.LongTensor([idx_arr]).to(DEVICE)
        # 预测
        out = model(x)
        pred_cls = torch.argmax(out, dim=-1).item()
        return label_map[pred_cls]

# ===================== 5. 热点挖掘函数 =====================
@st.cache_data
def get_hot_tops(sample_texts):
    cut_texts = [" ".join(text_cut(t)) for t in sample_texts]
    vec = TfidfVectorizer(max_features=3000)
    tfidf_data = vec.fit_transform(cut_texts)
    db = DBSCAN(eps=0.8, min_samples=5)
    cluster_label = db.fit_predict(tfidf_data)
    cluster_cnt = Counter(cluster_label)
    hot_cluster = sorted([k for k in cluster_cnt.keys() if k != -1],
                         key=lambda x: cluster_cnt[x], reverse=True)
    return cluster_label, cluster_cnt, hot_cluster

# ===================== 6. 页面主体 =====================
def main():
    st.set_page_config(page_title="CNN-LSTM 新闻分类 & 热点挖掘", layout="wide")
    st.title("📰 基于CNN-LSTM的新闻分类与热点挖掘系统")

    # 加载资源
    try:
        model, vocab, vocab_size = load_all_resources()
        st.success("✅ 模型、词表加载完成！")
    except Exception as e:
        st.error(f"❌ 资源加载失败：{str(e)}")
        st.info("请检查 cnn_lstm_model.pth / vocab.pth / vocab_size.txt 是否在当前目录")
        return

    # 分栏布局
    tab1, tab2 = st.tabs(["文本分类预测", "新闻热点挖掘"])

    # 标签页1：单文本预测
    with tab1:
        st.subheader("输入新闻内容，自动分类")
        user_text = st.text_area("请输入新闻文本：", height=200)
        if st.button("开始分类", type="primary"):
            if user_text.strip():
                res = predict_text(model, vocab, vocab_size, user_text)
                st.success(f"📌 预测分类结果：**{res}**")
            else:
                st.warning("请输入新闻文本内容！")

    # 标签页2：热点挖掘
    with tab2:
        st.subheader("测试集新闻热点挖掘 (TF-IDF + DBSCAN)")
        st.info("默认抽取部分新闻进行聚类，展示TOP5热点事件")

        # 读取测试集文本（和训练代码路径一致）
        data_dir = r"E:\PythonProject\data\cnews"
        test_path = os.path.join(data_dir, "cnews.test.txt")
        if not os.path.exists(test_path):
            st.error(f"测试文件不存在：{test_path}")
            return

        # 读取测试文本
        def load_test_data(path):
            texts = []
            labels = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    lab, txt = line.split("\t", 1)
                    texts.append(txt)
                    labels.append(lab)
            return texts, labels

        test_texts, test_labels_str = load_test_data(test_path)
        sample_size = 2000
        sample_texts = test_texts[:sample_size]

        if st.button("开始挖掘热点", type="primary"):
            with st.spinner("正在聚类分析，请稍等..."):
                cluster_label, cluster_cnt, hot_cluster = get_hot_tops(sample_texts)
                topN = 5
                st.subheader(f"TOP {topN} 热点事件")

                for idx, hot_id in enumerate(hot_cluster[:topN]):
                    hot_texts = [sample_texts[i] for i, c in enumerate(cluster_label) if c == hot_id]
                    st.markdown(f"""
                    **热点{idx+1}**
                    - 簇内新闻数量：{cluster_cnt[hot_id]} 条
                    - 示例新闻：{hot_texts[0][:150]}...
                    """)
                    st.divider()

if __name__ == "__main__":
    main()