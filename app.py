import streamlit as st
import torch
import jieba
import os
import time
import json
import random
import requests
from bs4 import BeautifulSoup
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN

# ===================== 路径配置 =====================
VOCAB_PATH = "vocab.pth"
VOCAB_SIZE_PATH = "vocab_size.txt"
SAVE_MODEL_PATH = "cnn_lstm_model.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN = 64
EMB_DIM = 128
CLASS_NUM = 10

label_map = {
    0:"体育",1:"娱乐",2:"家居",3:"房产",4:"教育",
    5:"时政",6:"财经",7:"科技",8:"时尚",9:"游戏"
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
    "Accept-Language": "zh-CN,zh;q=0.9"
}

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

# ===================== 文本工具 =====================
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

def cluster_hot_topics(sample_texts):
    cut_texts = [" ".join(text_cut(t)) for t in sample_texts]
    vec = TfidfVectorizer(max_features=2000)
    tfidf_data = vec.fit_transform(cut_texts)
    db = DBSCAN(eps=0.8, min_samples=2)
    labels = db.fit_predict(tfidf_data)
    counter = Counter(labels)
    hot_clusters = sorted([k for k in counter if k!=-1], key=lambda x: counter[x], reverse=True)
    return labels, counter, hot_clusters

# ===================== 三平台爬虫（核心新增） =====================
def crawl_weibo():
    """微博热搜（API）"""
    try:
        url = "https://weibo.com/ajax/side/hotSearch"
        r = requests.get(url, headers=HEADERS, timeout=10)
        r.raise_for_status()
        data = r.json()
        hot_list = data["data"]["realtime"]
        return [item["word"] for item in hot_list if item.get("word")][:20]
    except Exception as e:
        st.error(f"微博抓取失败：{e}")
        return []

def crawl_zhihu():
    """知乎热榜（解析JSON）"""
    try:
        url = "https://www.zhihu.com/billboard"
        r = requests.get(url, headers=HEADERS, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        script = soup.find("script", id="js-initialData")
        if not script:
            return []
        js_data = json.loads(script.string)
        hot_list = js_data["initialState"]["topstory"]["hotList"]
        titles = []
        for item in hot_list:
            try:
                titles.append(item["target"]["titleArea"]["text"])
            except:
                pass
        return titles[:20]
    except Exception as e:
        st.error(f"知乎抓取失败：{e}")
        return []

def crawl_toutiao():
    """今日头条热点（静态解析）"""
    try:
        url = "https://www.toutiao.com/"
        r = requests.get(url, headers=HEADERS, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        items = soup.select("div.hot-list-item a")
        titles = []
        for a in items:
            t = a.get_text(strip=True)
            if t:
                titles.append(t)
        return titles[:20]
    except Exception as e:
        st.error(f"今日头条抓取失败：{e}")
        return []

# ===================== 页面样式（紫色星空） =====================
def set_style():
    st.markdown("""
    <style>
    .stApp {
        background: radial-gradient(ellipse at top, #4a2a80 0%, #2c1654 40%, #1a0d33 70%, #0f0720 100%);
        color: #f0e6ff;
        font-family: 'Microsoft YaHei', sans-serif;
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
    }
    .stButton>button:hover {
        background: rgba(153, 102, 255, 0.5);
        border-color: #ffffff;
        box-shadow: 0 0 15px rgba(153, 102, 255, 0.6);
    }
    .stTextArea textarea, .stTextInput>div>div>input {
        background: rgba(40,20,70,0.6);
        color: #f0e6ff;
        border: 1px solid rgba(122, 67, 182, 0.6);
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

# ===================== 主程序 =====================
def main():
    st.set_page_config(page_title="多平台热点挖掘系统", layout="wide")
    set_style()

    # 首次运行：强制上传模型
    if not os.path.exists(SAVE_MODEL_PATH):
        st.markdown("<h1>📦 系统初始化 - 上传模型文件</h1>", unsafe_allow_html=True)
        st.warning("请上传 cnn_lstm_model.pth，其余文件从Git获取")
        file = st.file_uploader("选择 pth 文件", type="pth")
        if file:
            with open(SAVE_MODEL_PATH, "wb") as f:
                f.write(file.getbuffer())
            st.success("✅ 上传成功，正在跳转...")
            time.sleep(1.2)
            st.rerun()
        return

    # 加载词表与模型（兼容旧版）
    try:
        with open(VOCAB_SIZE_PATH, "r", encoding="utf-8") as f:
            vocab_size = int(f.read().strip())
        vocab = torch.load(VOCAB_PATH, map_location="cpu")
        model = CNNLSTM(vocab_size).to(DEVICE)
        model.load_state_dict(torch.load(SAVE_MODEL_PATH, map_location=DEVICE))
        model.eval()
    except Exception as e:
        st.error(f"加载失败：{e}")
        return

    # 主界面
    st.markdown("<h1>🌌 新闻分类 & 多平台热点挖掘系统</h1>", unsafe_allow_html=True)
    st.success("✅ 模型就绪")

    tab1, tab2, tab3 = st.tabs([
        "📝 单条新闻分类",
        "📁 文件热点挖掘",
        "🌐 微博/头条/知乎 热点抓取分析"
    ])

    # 1. 单条分类
    with tab1:
        st.subheader("输入新闻文本")
        text = st.text_area("新闻内容", height=220)
        if st.button("分类") and text.strip():
            res = predict_text(model, vocab, vocab_size, text)
            st.success(f"结果：{res}")

    # 2. 文件聚类
    with tab2:
        st.subheader("上传 TXT 文件（每行一条）")
        file = st.file_uploader("选择 txt", type="txt")
        if file:
            lines = file.read().decode("utf-8").splitlines()
            texts = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                if "\t" in line:
                    _, t = line.split("\t", 1)
                else:
                    t = line
                texts.append(t)
            st.success(f"读取 {len(texts)} 条")
            if len(texts) > 2000:
                texts = texts[:2000]
            if st.button("挖掘热点"):
                with st.spinner("聚类中..."):
                    lbl, cnt, hot = cluster_hot_topics(texts)
                    st.subheader("TOP5 热点")
                    for i, cid in enumerate(hot[:5]):
                        group = [texts[j] for j, cl in enumerate(lbl) if cl == cid]
                        st.markdown(f"**热点{i+1}**（{cnt[cid]}条）：{group[0][:120]}...")

    # 3. 多平台爬虫+分析（核心）
    with tab3:
        st.subheader("微博 / 今日头条 / 知乎 实时热点抓取")
        plat = st.radio("选择平台", ["微博热搜", "今日头条热点", "知乎热榜"])
        if st.button("开始抓取并分析"):
            with st.spinner("抓取中..."):
                if plat == "微博热搜":
                    hot_list = crawl_weibo()
                elif plat == "今日头条热点":
                    hot_list = crawl_toutiao()
                else:
                    hot_list = crawl_zhihu()
            if not hot_list:
                st.error("未获取到数据")
                return
            st.success(f"共抓取 {len(hot_list)} 条热点")
            st.divider()

            # 逐条分类
            st.subheader("热点分类结果")
            for idx, title in enumerate(hot_list, 1):
                cate = predict_text(model, vocab, vocab_size, title)
                st.markdown(f"{idx}.【{cate}】{title}")

            # 整体聚类
            st.divider()
            st.subheader("平台热点聚类分析")
            lbl, cnt, hot = cluster_hot_topics(hot_list)
            for i, cid in enumerate(hot[:3]):
                st.markdown(f"**聚类热点{i+1}**：包含 {cnt[cid]} 条相关内容")

if __name__ == "__main__":
    main()
