import streamlit as st
import torch
import jieba
import jieba.analyse
import numpy as np
import os
import re
import requests
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN

# ===================== 全局基础配置 =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN = 64
EMB_DIM = 128
CLASS_NUM = 10

# 新闻分类标签
label_map = {
    0:"体育",1:"娱乐",2:"家居",3:"房产",4:"教育",
    5:"时政",6:"财经",7:"科技",8:"时尚",9:"游戏"
}

MODEL_CACHE_NAME = "cnn_lstm_model.pth"

# ===================== CNN-LSTM 模型（保留原有） =====================
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

# ===================== 文本工具函数（原有功能保留） =====================
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

# 单条新闻分类（核心原有功能）
def predict_text(model, vocab, vocab_size, text):
    model.eval()
    with torch.no_grad():
        idx_arr = text2idx(text, vocab, vocab_size, SEQ_LEN)
        x = torch.LongTensor([idx_arr]).to(DEVICE)
        out = model(x)
        pred_cls = torch.argmax(out, dim=-1).item()
        return label_map[pred_cls]

# 关键词提取
def extract_keywords(text, topK=5):
    return jieba.analyse.textrank(text, topK=topK, withWeight=False, allowPOS=('ns', 'n', 'vn', 'v'))

# 文本摘要
def simple_summary(text, sent_cnt=2):
    sentences = re.split(r'[。！？]', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    if len(sentences) <= sent_cnt:
        return "。".join(sentences) + "。"
    return "。".join(sentences[:sent_cnt]) + "。"

# 热点聚类分析
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

# ===================== 多平台热点抓取模块（新增爬虫/接口） =====================
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# 1. 微博热搜
@st.cache_data(ttl=1800)
def get_weibo_hot():
    url = "https://api.oioweb.cn/api/hotlist/weibo"
    try:
        res = requests.get(url, headers=HEADERS, timeout=10)
        data = res.json()
        articles = []
        if data.get("code") == 200 and data.get("data"):
            for item in data["data"]:
                title = item.get("title", "").strip()
                hot = item.get("hot", "0")
                link = item.get("url", "")
                content = f"{title} 热度：{hot}"
                articles.append({
                    "title": title, "desc": f"热度值：{hot}",
                    "link": link, "content": content, "source": "微博热搜"
                })
        return articles
    except Exception as e:
        st.warning(f"微博热搜抓取失败：{str(e)}")
        return []

# 2. 今日头条热点
@st.cache_data(ttl=1800)
def get_toutiao_hot():
    url = "https://api.oioweb.cn/api/hotlist/toutiao"
    try:
        res = requests.get(url, headers=HEADERS, timeout=10)
        data = res.json()
        articles = []
        if data.get("code") == 200 and data.get("data"):
            for item in data["data"]:
                title = item.get("title", "").strip()
                link = item.get("url", "")
                content = title
                articles.append({
                    "title": title, "desc": "",
                    "link": link, "content": content, "source": "今日头条"
                })
        return articles
    except Exception as e:
        st.warning(f"头条热点抓取失败：{str(e)}")
        return []

# 3. 知乎热榜
@st.cache_data(ttl=1800)
def get_zhihu_hot():
    url = "https://api.oioweb.cn/api/hotlist/zhihu"
    try:
        res = requests.get(url, headers=HEADERS, timeout=10)
        data = res.json()
        articles = []
        if data.get("code") == 200 and data.get("data"):
            for item in data["data"]:
                title = item.get("title", "").strip()
                hot = item.get("hot", "0")
                link = item.get("url", "")
                content = f"{title} 讨论热度：{hot}"
                articles.append({
                    "title": title, "desc": f"讨论热度：{hot}",
                    "link": link, "content": content, "source": "知乎热榜"
                })
        return articles
    except Exception as e:
        st.warning(f"知乎热榜抓取失败：{str(e)}")
        return []

# 整合全平台热点
def get_all_hot_news():
    all_arts = []
    all_arts.extend(get_weibo_hot())
    all_arts.extend(get_toutiao_hot())
    all_arts.extend(get_zhihu_hot())
    return all_arts

# ===================== 页面主体 =====================
def main():
    st.set_page_config(page_title="新闻分类 & 多平台热点分析系统", layout="wide")

    # 页面样式（保留原有紫色风格）
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to bottom, #0b0423 0%, #190b37 40%, #2a1052 70%, #1a0736 100%);
        color: #f0e6ff;
        font-family: 'Microsoft YaHei', sans-serif;
    }
    h1, h2, h3 {
        color: #e8d8ff;
        text-shadow: 0 0 10px #9966ff;
    }
    .stButton>button {
        background: linear-gradient(90deg, #7a43b6, #9966ff);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        font-weight: bold;
    }
    .stTextArea textarea {
        background: rgba(40,20,70,0.7);
        color: #f0e6ff;
        border: 1px solid #7a43b6;
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("📰 CNN-LSTM 新闻分类 & 多平台热点分析系统")

    # 模型上传与加载（原有逻辑完全保留）
    st.subheader("第一步：上传模型文件 cnn_lstm_model.pth")
    uploaded_model = st.file_uploader("选择 .pth 模型文件", type="pth")
    if uploaded_model is not None:
        with open(MODEL_CACHE_NAME, "wb") as f:
            f.write(uploaded_model.getbuffer())
        st.success("✅ 模型上传并缓存完成")

    if not os.path.exists(MODEL_CACHE_NAME):
        st.info("⚠️ 请先上传模型文件，再使用功能")
        return

    # 加载词表与模型
    try:
        with open("vocab_size.txt", "r", encoding="utf-8") as f:
            vocab_size = int(f.read().strip())
        torch.serialization.add_safe_globals([dict, list])
        vocab = torch.load("vocab.pth", map_location="cpu", weights_only=False)
        model = CNNLSTM(vocab_size=vocab_size).to(DEVICE)
        model.load_state_dict(torch.load(MODEL_CACHE_NAME, map_location=DEVICE, weights_only=False))
        model.eval()
        st.success("✅ 模型、词表加载完毕，所有功能可用")
    except Exception as e:
        st.error(f"❌ 资源加载失败：{str(e)}")
        return

    # 两大功能标签页
    tab1, tab2 = st.tabs(["📝 单条新闻分类识别", "🌐 多平台热点抓取与分析"])

    # ========== 标签1：原有单新闻分类功能（完全不变） ==========
    with tab1:
        st.subheader("输入新闻文本，自动识别新闻类别")
        user_text = st.text_area("请粘贴新闻内容：", height=220)
        if st.button("开始分类", type="primary"):
            if user_text.strip():
                res = predict_text(model, vocab, vocab_size, user_text)
                st.success(f"📌 识别结果：**{res}**")
            else:
                st.warning("请输入新闻文本内容！")

    # ========== 标签2：多平台热点抓取+分析（新增功能） ==========
    with tab2:
        st.subheader("🌐 微博/头条/知乎 全平台热点挖掘 & 智能分析")
        st.info("自动抓取三大平台热点，聚类同类事件、提取关键词、生成摘要、按热度排序")

        # 选择抓取平台
        platform = st.radio("选择抓取平台", ["全平台", "微博热搜", "今日头条", "知乎热榜"])

        if st.button("开始抓取并分析热点", type="primary"):
            with st.spinner("正在抓取数据、聚类分析、生成总结，请稍等..."):
                # 根据选择抓取对应平台
                if platform == "微博热搜":
                    articles = get_weibo_hot()
                elif platform == "今日头条":
                    articles = get_toutiao_hot()
                elif platform == "知乎热榜":
                    articles = get_zhihu_hot()
                else:
                    articles = get_all_hot_news()

                if not articles:
                    st.error("❌ 未获取到热点数据，请稍后重试")
                    return
                st.success(f"✅ 共获取到 {len(articles)} 条热点数据")

                # 聚类分析
                labels, counter, hot_clusters = cluster_hot_topics(articles)
                topN = 5
                st.subheader(f"🔥 综合 TOP {topN} 热点事件（按热度排序）")

                # 遍历输出热点详情
                for idx, cid in enumerate(hot_clusters[:topN]):
                    group = [articles[i] for i, lab in enumerate(labels) if lab == cid]
                    main_title = group[0]["title"]
                    source = group[0]["source"]
                    all_content = " ".join([a["content"] for a in group])
                    keywords = extract_keywords(all_content)
                    summary = simple_summary(all_content)

                    st.markdown(f"""
**{idx+1}. {main_title}**
- 来源平台：{source}
- 相关条目数量：{counter[cid]} 条
- 核心关键词：{', '.join(keywords)}
- 内容总结：{summary}
- 查看原文：[点击跳转]({group[0]['link']})
                    """)
                    st.divider()

if __name__ == "__main__":
    main()
