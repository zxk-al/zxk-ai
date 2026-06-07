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

# 新闻分类标签映射
label_map = {
    0:"体育",1:"娱乐",2:"家居",3:"房产",4:"教育",
    5:"时政",6:"财经",7:"科技",8:"时尚",9:"游戏" 
}

# 本地文件路径（和代码同目录）
MODEL_PATH = "cnn_lstm_model.pth"
VOCAB_PATH = "vocab.pth"
VOCAB_SIZE_PATH = "vocab_size.txt"

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

# 单条新闻分类（核心功能1）
def predict_text(model, vocab, vocab_size, text):
    model.eval()
    with torch.no_grad():
        idx_arr = text2idx(text, vocab, vocab_size, SEQ_LEN)
        x = torch.LongTensor([idx_arr]).to(DEVICE)
        out = model(x)
        pred_cls = torch.argmax(out, dim=-1).item()
        return label_map[pred_cls]

# 热点聚类函数（核心功能2，适配上传文件）
def cluster_hot_topics(sample_texts):
    cut_texts = [" ".join(text_cut(t)) for t in sample_texts]
    vec = TfidfVectorizer(max_features=2000)
    tfidf_data = vec.fit_transform(cut_texts)
    db = DBSCAN(eps=0.8, min_samples=2)
    labels = db.fit_predict(tfidf_data)
    counter = Counter(labels)
    hot_clusters = sorted([k for k in counter if k != -1], key=lambda x: counter[x], reverse=True)
    return labels, counter, hot_clusters

# ===================== 页面主体 =====================
def main():
    st.set_page_config(page_title="CNN-LSTM 新闻分类 & 热点挖掘系统", layout="wide")

    # 紫色星空全屏背景样式
    st.markdown(
        """
        <style>
        .stApp {
            background: radial-gradient(ellipse at top, #4a2a80 0%, #2c1654 40%, #1a0d33 70%, #0f0720 100%);
            color: #f0e6ff;
            font-family: 'Microsoft YaHei', sans-serif;
        }
        .stApp::before {
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-image: url('https://cdn.pixabay.com/photo/2016/09/08/18/00/starry-sky-1655304_1280.jpg');
            background-size: cover;
            opacity: 0.3;
            z-index: -1;
        }
        h1, h2, h3 {
            color: #ffffff;
            text-shadow: 0 0 15px #9966ff, 0 0 30px #7a43b6;
            text-align: center;
        }
        .stButton>button {
            background: rgba(122, 67, 182, 0.3);
            color: white;
            border: 1px solid rgba(153, 102, 255, 0.6);
            border-radius: 8px;
            padding: 0.8rem 1.5rem;
            font-weight: 500;
            backdrop-filter: blur(8px);
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background: rgba(153, 102, 255, 0.5);
            border-color: #ffffff;
            box-shadow: 0 0 15px rgba(153, 102, 255, 0.6);
        }
        .stTextArea textarea {
            background: rgba(40,20,70,0.6);
            color: #f0e6ff;
            border: 1px solid rgba(122, 67, 182, 0.6);
            border-radius: 8px;
            backdrop-filter: blur(4px);
        }
        .css-18e3th9 {
            padding-top: 2rem;
        }
        .tab-container {
            display: flex;
            justify-content: center;
            gap: 1rem;
            margin: 2rem 0;
        }
        .tab-button {
            background: rgba(122, 67, 182, 0.2);
            color: white;
            border: 1px solid rgba(153, 102, 255, 0.5);
            border-radius: 8px;
            padding: 0.7rem 1.2rem;
            backdrop-filter: blur(8px);
            transition: all 0.3s ease;
        }
        .tab-button:hover {
            background: rgba(153, 102, 255, 0.4);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # 主标题
    st.markdown("<h1 style='font-size: 3rem; margin-top: 3rem;'>CNN-LSTM 新闻分类 & 热点挖掘系统</h1>", unsafe_allow_html=True)

    # 功能按钮区（和图片中的两个功能按钮对应）
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        if st.button("📝 单条新闻分类识别", use_container_width=True):
            st.session_state["active_tab"] = "classify"
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        if st.button("📁 新闻文件热点挖掘", use_container_width=True):
            st.session_state["active_tab"] = "cluster"
        st.markdown("</div>", unsafe_allow_html=True)

    # 默认显示第一个功能
    if "active_tab" not in st.session_state:
        st.session_state["active_tab"] = "classify"

    # 校验本地文件是否存在
    miss_file = []
    if not os.path.exists(MODEL_PATH):
        miss_file.append(MODEL_PATH)
    if not os.path.exists(VOCAB_PATH):
        miss_file.append(VOCAB_PATH)
    if not os.path.exists(VOCAB_SIZE_PATH):
        miss_file.append(VOCAB_SIZE_PATH)

    if miss_file:
        st.error(f"❌ 缺失本地资源文件：{', '.join(miss_file)}，请将文件放置在代码同目录！")
        return

    # 直接加载本地词表、模型（兼容旧版PyTorch）
    try:
        with open(VOCAB_SIZE_PATH, "r", encoding="utf-8") as f:
            vocab_size = int(f.read().strip())

        vocab = torch.load(VOCAB_PATH, map_location="cpu")
        model = CNNLSTM(vocab_size=vocab_size).to(DEVICE)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()

        st.success("✅ 本地模型、词表加载完毕，所有功能可用")

    except Exception as e:
        st.error(f"❌ 资源加载失败：{str(e)}")
        return

    # ========== 标签1：单条新闻分类 ==========
    if st.session_state["active_tab"] == "classify":
        st.subheader("输入新闻文本，自动识别分类")
        user_text = st.text_area("请粘贴新闻内容：", height=220)
        if st.button("开始分类", type="primary"):
            if user_text.strip():
                res = predict_text(model, vocab, vocab_size, user_text)
                st.success(f"📌 识别结果：**{res}**")
            else:
                st.warning("请输入新闻文本内容！")

    # ========== 标签2：文件上传式热点挖掘 ==========
    elif st.session_state["active_tab"] == "cluster":
        st.subheader("📁 上传新闻文件进行热点挖掘")
        st.info("支持上传 .txt 文件，每行一条新闻，格式：`标签\t新闻内容` 或纯文本")

        uploaded_news_file = st.file_uploader("选择新闻文件", type="txt")

        if uploaded_news_file is not None:
            sample_texts = []
            try:
                content = uploaded_news_file.read().decode("utf-8").splitlines()
                for line in content:
                    line = line.strip()
                    if not line:
                        continue
                    # 兼容两种格式
                    if "\t" in line:
                        _, text = line.split("\t", 1)
                    else:
                        text = line
                    sample_texts.append(text)

                st.success(f"✅ 成功读取 {len(sample_texts)} 条新闻")

                # 控制样本数量，避免聚类过慢
                if len(sample_texts) > 2000:
                    st.info("⚠️ 新闻数量较多，将只取前2000条进行聚类")
                    sample_texts = sample_texts[:2000]

                # 开始挖掘热点
                if st.button("开始挖掘热点", type="primary"):
                    with st.spinner("正在聚类分析，请稍等..."):
                        cluster_label, cluster_cnt, hot_clusters = cluster_hot_topics(sample_texts)
                        topN = 5

                        st.subheader(f"🔥 TOP {topN} 热点事件（按热度从高到低）")

                        for idx, hot_id in enumerate(hot_clusters[:topN]):
                            hot_texts = [sample_texts[i] for i, c in enumerate(cluster_label) if c == hot_id]
                            st.markdown(f"""
                            **热点{idx+1}**
                            - 簇内新闻数量：{cluster_cnt[hot_id]} 条
                            - 示例新闻：{hot_texts[0][:150]}...
                            """)
                            st.divider()

            except Exception as e:
                st.error(f"❌ 文件读取失败：{str(e)}")

if __name__ == "__main__":
    main()
