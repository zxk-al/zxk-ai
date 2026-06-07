import streamlit as st
import torch
import jieba
import jieba.analyse
import numpy as np
import os
import re
import requests
from xml.etree import ElementTree
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

MODEL_CACHE_NAME = "cnn_lstm_model.pth"

# ===================== 模型定义 =====================
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
        idx += [0]*(seq_len-len(idx))
    else:
        idx = idx[:seq_len]
    return idx

def predict_text(model, vocab, vocab_size, text):
    model.eval()
    with torch.no_grad():
        idx_arr = text2idx(text, vocab, vocab_size, SEQ_LEN)
        x = torch.LongTensor([idx_arr]).to(DEVICE)
        out = model(x)
        pred_cls = torch.argmax(out, dim=-1).item()
        return label_map[pred_cls]

# ===================== 今日头条 RSS 热点抓取（稳定版） =====================
@st.cache_data(ttl=1800)
def fetch_toutiao_news(max_news=30):
    """
    抓取今日头条 RSS 热点，无需第三方接口，稳定可用
    """
    try:
        # 今日头条公开 RSS 源，Streamlit Cloud 可直接访问
        url = "https://www.toutiao.com/feed/?iid=1234567890&device_id=1234567890"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        res = requests.get(url, headers=headers, timeout=15)
        res.raise_for_status()

        # 解析 XML
        root = ElementTree.fromstring(res.content)
        items = root.findall(".//item")
        articles = []
        for item in items[:max_news]:
            title = item.findtext("title", "").strip()
            link = item.findtext("link", "").strip()
            desc = item.findtext("description", "").strip()
            content = f"{title} {desc}".strip()
            if len(content) > 20:
                articles.append({
                    "title": title,
                    "desc": desc,
                    "link": link,
                    "content": content
                })
        return articles
    except Exception as e:
        st.error(f"头条抓取失败：{str(e)}")
        return []

# 关键词提取
def extract_keywords(text, topK=5):
    return jieba.analyse.textrank(text, topK=topK, withWeight=False, allowPOS=('ns', 'n', 'vn', 'v'))

# 简单摘要
def simple_summary(text, sent_cnt=2):
    sentences = re.split(r'[。！？]', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    if len(sentences) <= sent_cnt:
        return "。".join(sentences) + "。"
    return "。".join(sentences[:sent_cnt]) + "。"

# 热点聚类
def cluster_hot_topics(articles):
    texts = [a["content"] for a in articles]
    cut_texts = [" ".join(text_cut(t)) for t in texts]
    vec = TfidfVectorizer(max_features=2000)
    tfidf = vec.fit_transform(cut_texts)
    db = DBSCAN(eps=0.8, min_samples=2)
    labels = db.fit_predict(tfidf)
    counter = Counter(labels)
    hot_clusters = sorted([k for k in counter if k != -1], key=lambda x: counter[x], reverse=True)
    return labels, counter, hot_clusters

# ===================== 页面 =====================
def main():
    st.set_page_config(page_title="新闻分类 & 头条热点挖掘", layout="wide")
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to bottom, #0b0423 0%, #190b37 40%, #2a1052 70%, #1a0736 100%);
        color: #f0e6ff;
    }
    h1, h2, h3 {
        color: #e8d8ff;
        text-shadow: 0 0 10px #9966ff;
    }
    .stButton>button {
        background: linear-gradient(90deg, #7a43b6, #9966ff);
        color: white;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("📰 CNN-LSTM 新闻分类 & 今日头条热点挖掘系统")

    # 上传模型
    st.subheader("第一步：上传模型文件 cnn_lstm_model.pth")
    uploaded_model = st.file_uploader("选择 .pth 文件", type="pth")
    if uploaded_model:
        with open(MODEL_CACHE_NAME, "wb") as f:
            f.write(uploaded_model.getbuffer())
        st.success("✅ 模型已缓存")

    if not os.path.exists(MODEL_CACHE_NAME):
        st.info("⚠️ 请先上传模型文件，再使用功能")
        return

    # 加载词表
    try:
        with open("vocab_size.txt", "r", encoding="utf-8") as f:
            vocab_size = int(f.read().strip())
        torch.serialization.add_safe_globals([dict, list])
        vocab = torch.load("vocab.pth", map_location="cpu", weights_only=False)
        model = CNNLSTM(vocab_size).to(DEVICE)
        model.load_state_dict(torch.load(MODEL_CACHE_NAME, map_location=DEVICE, weights_only=False))
        model.eval()
        st.success("✅ 模型、词表加载完毕，所有功能可用")
    except Exception as e:
        st.error(f"❌ 资源加载失败：{str(e)}")
        return

    tab1, tab2 = st.tabs(["📝 单条新闻分类识别", "🌐 今日头条热点挖掘"])

    # 标签1：单条新闻分类（完全保留原功能）
    with tab1:
        st.subheader("输入新闻文本，自动识别分类")
        user_text = st.text_area("请粘贴新闻内容：", height=220)
        if st.button("开始分类", type="primary"):
            if user_text.strip():
                res = predict_text(model, vocab, vocab_size, user_text)
                st.success(f"📌 识别结果：**{res}**")
            else:
                st.warning("请输入新闻文本内容！")

    # 标签2：头条热点挖掘（稳定版）
    with tab2:
        st.subheader("🌐 自动抓取今日头条热点，按热度排序、生成摘要")
        st.info("数据源：今日头条官方 RSS 源，无需翻墙，稳定可用")

        if st.button("开始挖掘头条热点", type="primary"):
            with st.spinner("正在抓取头条新闻并聚类分析，请稍等..."):
                articles = fetch_toutiao_news(max_news=30)
                if not articles:
                    st.error("❌ 未能获取到头条新闻，请稍后重试")
                    return

                if len(articles) < 3:
                    st.warning("⚠️ 获取的新闻数量过少，无法进行聚类")
                    return

                try:
                    labels, counter, hot_clusters = cluster_hot_topics(articles)
                    topN = 5
                    st.subheader(f"🔥 TOP {topN} 头条热门事件（按热度从高到低）")

                    for idx, cid in enumerate(hot_clusters[:topN]):
                        group = [articles[i] for i, lab in enumerate(labels) if lab == cid]
                        main_title = group[0]["title"]
                        all_content = " ".join([a["content"] for a in group])
                        keywords = extract_keywords(all_content)
                        summary = simple_summary(all_content)

                        st.markdown(f"""
**{idx+1}. {main_title}**
- 相关新闻数量：{counter[cid]} 条
- 核心关键词：{', '.join(keywords)}
- 内容摘要：{summary}
- 查看原文：[点击跳转]({group[0]['link']})
                        """)
                        st.divider()
                except Exception as e:
                    st.error(f"❌ 聚类分析失败：{str(e)}")

if __name__ == "__main__":
    main()
