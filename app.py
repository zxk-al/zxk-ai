import streamlit as st
import torch
import jieba
import numpy as np
import os
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN

# ===================== 全局配置 =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN = 64
EMB_DIM = 128
CLASS_NUM = 10

label_map = {
    0:"体育",1:"娱乐",2:"家居",3:"房产",4:"教育",
    5:"时政",6:"财经",7:"科技",8:"时尚",9:"游戏"
}

# 模型文件本地缓存名
MODEL_CACHE_NAME = "cnn_lstm_model.pth"

# ===================== 定义 CNNLSTM 模型 =====================
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

# ===================== 文本工具函数 =====================
def text_cut(text):
    return jieba.lcut(text)

def text2idx(text, vocab, vocab_size, seq_len=64):
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

# ===================== 热点挖掘函数 =====================
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

# ===================== 页面主体 + 紫色星空样式 =====================
def main():
    st.set_page_config(page_title="CNN-LSTM 新闻分类 & 热点挖掘", layout="wide")

    # 紫色星空CSS样式
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(to bottom, #0b0423 0%, #190b37 40%, #2a1052 70%, #1a0736 100%);
            color: #f0e6ff;
            font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #e8d8ff;
            text-shadow: 0 0 10px #9966ff, 0 0 20px #7a43b6;
        }
        .stButton>button {
            background: linear-gradient(90deg, #7a43b6 0%, #9966ff 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.6rem 1.2rem;
            font-weight: bold;
            box-shadow: 0 0 15px rgba(122, 67, 182, 0.6);
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            transform: scale(1.05);
            box-shadow: 0 0 25px rgba(153, 102, 255, 0.8);
        }
        .stTextArea textarea {
            background-color: rgba(40, 20, 70, 0.7) !important;
            color: #f0e6ff !important;
            border: 1px solid #7a43b6;
            border-radius: 8px;
        }
        .stTabs [data-baseweb="tab-list"] { gap: 10px; }
        .stTabs [data-baseweb="tab"] {
            background-color: rgba(122, 67, 182, 0.3);
            color: #e8d8ff;
            border-radius: 8px 8px 0 0;
            padding: 0.5rem 1rem;
        }
        .stTabs [aria-selected="true"] {
            background-color: #7a43b6 !important;
            color: white !important;
        }
        .stSuccess {
            background-color: rgba(0, 128, 0, 0.2);
            border: 1px solid #00ff88;
            border-radius: 8px;
        }
        .stWarning {
            background-color: rgba(255, 165, 0, 0.2);
            border: 1px solid #ffcc00;
            border-radius: 8px;
        }
        .stError {
            background-color: rgba(255, 0, 0, 0.2);
            border: 1px solid #ff4444;
            border-radius: 8px;
        }
        hr { border-color: #7a43b6; box-shadow: 0 0 5px #9966ff; }
        ::-webkit-scrollbar { width: 8px; }
        ::-webkit-scrollbar-track { background: #190b37; }
        ::-webkit-scrollbar-thumb { background: #7a43b6; border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: #9966ff; }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("📰 基于CNN-LSTM的新闻分类与热点挖掘系统")

    # -------------------- 第一步：上传大模型文件（绕开Git限制） --------------------
    st.subheader("第一步：上传模型文件 cnn_lstm_model.pth")
    uploaded_model = st.file_uploader("选择本地的 cnn_lstm_model.pth 文件", type="pth")

    # 保存上传的模型到服务端本地
    if uploaded_model is not None:
        with open(MODEL_CACHE_NAME, "wb") as f:
            f.write(uploaded_model.getbuffer())
        st.success("✅ 模型文件上传并缓存完成！")

    # 检查模型是否已存在
    if not os.path.exists(MODEL_CACHE_NAME):
        st.info("⚠️ 请先上传 cnn_lstm_model.pth 文件，再使用功能")
        return

    # -------------------- 第二步：加载词表（小文件已放在Git） --------------------
    vocab_path = "vocab.pth"
    if not os.path.exists(vocab_path):
        st.error("❌ 词表文件 vocab.pth 缺失，请检查Git仓库")
        return

    try:
        # 读取词表大小
        with open("vocab_size.txt", "r", encoding="utf-8") as f:
            vocab_size = int(f.read().strip())

        # 兼容 PyTorch 2.6+
        torch.serialization.add_safe_globals([dict, list])
        vocab = torch.load(vocab_path, map_location="cpu", weights_only=False)

        # 加载模型
        model = CNNLSTM(vocab_size=vocab_size).to(DEVICE)
        model.load_state_dict(torch.load(MODEL_CACHE_NAME, map_location=DEVICE, weights_only=False))
        model.eval()
        st.success("✅ 模型 + 词表 全部加载完成，可以开始使用！")

    except Exception as e:
        st.error(f"❌ 加载失败：{str(e)}")
        return

    # -------------------- 功能页面 --------------------
    tab1, tab2 = st.tabs(["文本分类预测", "新闻热点挖掘"])

    # 文本分类
    with tab1:
        st.subheader("输入新闻内容，自动分类")
        user_text = st.text_area("请输入新闻文本：", height=200)
        if st.button("开始分类", type="primary"):
            if user_text.strip():
                res = predict_text(model, vocab, vocab_size, user_text)
                st.success(f"📌 预测分类结果：**{res}**")
            else:
                st.warning("请输入新闻文本内容！")

    # 热点挖掘
    with tab2:
        st.subheader("测试集新闻热点挖掘 (TF-IDF + DBSCAN)")
        st.info("默认抽取部分新闻进行聚类，展示TOP5热点事件")

        data_dir = r"E:\PythonProject\data\cnews"
        test_path = os.path.join(data_dir, "cnews.test.txt")
        if not os.path.exists(test_path):
            st.error(f"测试文件不存在：{test_path}")
            return

        def load_test_data(path):
            texts, labels = [], []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    lab, txt = line.split("\t", 1)
                    texts.append(txt)
                    labels.append(lab)
            return texts, labels

        test_texts, _ = load_test_data(test_path)
        sample_texts = test_texts[:2000]

        if st.button("开始挖掘热点", type="primary"):
            with st.spinner("正在聚类分析，请稍等..."):
                cluster_label, cluster_cnt, hot_cluster = get_hot_tops(sample_texts)
                st.subheader("TOP 5 热点事件")
                for idx, hot_id in enumerate(hot_cluster[:5]):
                    hot_texts = [sample_texts[i] for i, c in enumerate(cluster_label) if c == hot_id]
                    st.markdown(f"""
                    **热点{idx+1}**
                    - 簇内新闻数量：{cluster_cnt[hot_id]} 条
                    - 示例新闻：{hot_texts[0][:150]}...
                    """)
                    st.divider()

if __name__ == "__main__":
    main()
