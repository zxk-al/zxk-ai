import os
import requests

# -------------------------- 配置部分（只改这里！） --------------------------
# 1. 模型文件的保存路径（和你本地的路径保持一致）
MODEL_PATH = "cnn_lstm_model.pth"

# 2. 替换成你的百度网盘文件直链（需要解析后的直接下载链接）
MODEL_URL = "https://pan.baidu.com/s/1M-CHAS4BvRvBE4vOqkiA4Q?pwd=9527 "
# ---------------------------------------------------------------------------

# 自动下载模型文件
if not os.path.exists(MODEL_PATH):
    print(f"正在下载模型文件 {MODEL_PATH}...")
    # 创建文件夹（如果路径包含子目录）
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    # 流式下载，避免大文件占用过多内存
    response = requests.get(MODEL_URL, stream=True)
    response.raise_for_status()  # 下载失败时抛出错误

    # 写入文件
    with open(MODEL_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB 分块下载
            if chunk:
                f.write(chunk)
    print(f"模型文件 {MODEL_PATH} 下载完成！")

# 之后你就可以正常加载模型了，比如：
# model = torch.load(MODEL_PATH)
import streamlit as st
import torch
import jieba
import pandas as pd
import numpy as np
from io import StringIO
import matplotlib.pyplot as plt

# ===================== 全局配置（和训练代码保持一致） =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN = 64
EMB_DIM = 128
CLASS_NUM = 10

# 读取词表大小
with open("vocab_size.txt", "r", encoding="utf-8") as f:
    VOCAB_SIZE = int(f.read())

# 分类映射
label_map = {
    0:"体育",1:"娱乐",2:"家居",3:"房产",4:"教育",
    5:"时政",6:"财经",7:"科技",8:"时尚",9:"游戏"
}
all_categories = list(label_map.values())

# ===================== 模型定义（和main.py里的CNNLSTM完全一致） =====================
import torch.nn as nn
class CNNLSTM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, EMB_DIM, padding_idx=0)
        self.conv3 = nn.Conv1d(EMB_DIM, 64, kernel_size=3, padding="same")
        self.conv4 = nn.Conv1d(EMB_DIM, 64, kernel_size=4, padding="same")
        self.conv5 = nn.Conv1d(EMB_DIM, 64, kernel_size=5, padding="same")
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(64*3, 128, batch_first=True, bidirectional=False)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(128, CLASS_NUM)

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

# ===================== 加载模型与词表（缓存加速） =====================
@st.cache_resource
def load_model_and_vocab():
    model = CNNLSTM(VOCAB_SIZE)
    model.load_state_dict(torch.load("cnn_lstm_model.pth", map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    vocab = torch.load("vocab.pth")
    return model, vocab

model, vocab = load_model_and_vocab()

# ===================== 文本预处理 & 预测函数 =====================
def text_cut(text):
    return jieba.lcut(text)

def text2idx(text, seq_len=64):
    words = text_cut(text)
    idx = []
    for word in words:
        word_id = vocab.get(word, 0)
        if word_id >= len(vocab):
            word_id = 0
        idx.append(word_id)
    if len(idx) < seq_len:
        idx += [0] * (seq_len - len(idx))
    else:
        idx = idx[:seq_len]
    return idx

def predict_single(text):
    vec = text2idx(text, SEQ_LEN)
    tensor_x = torch.LongTensor([vec]).to(DEVICE)
    with torch.no_grad():
        output = model(tensor_x)
        pred_idx = torch.argmax(output, dim=1).item()
    return label_map[pred_idx]

def predict_batch(text_list):
    res = []
    for txt in text_list:
        res.append(predict_single(str(txt)))
    return res

# ===================== 页面全局美化配置 =====================
st.set_page_config(
    page_title="智能新闻分类系统",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 超精致自定义CSS：渐变背景、毛玻璃卡片、动态按钮、深色适配
custom_css = """
<style>
/* 全局渐变背景 */
[data-testid="stAppViewContainer"]{
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    background-attachment: fixed;
}
/* 毛玻璃卡片样式 */
.glass-card{
    background: rgba(255, 255, 255, 0.9);
    backdrop-filter: blur(10px);
    border-radius: 20px;
    padding: 2rem;
    box-shadow: 0 8px 32px rgba(31, 38, 135, 0.15);
    border: 1px solid rgba(255, 255, 255, 0.18);
    margin-bottom: 1.5rem;
}
/* 标题样式 */
h1{
    color: #ffffff;
    text-align: center;
    font-weight: 800;
    text-shadow: 0 2px 10px rgba(0,0,0,0.2);
    margin-bottom: 2rem;
}
h2, h3, h4{
    color: #2d3748;
}
/* 自定义按钮：渐变+悬浮动画 */
.stButton>button{
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 12px;
    height: 50px;
    font-size: 16px;
    font-weight: 600;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
}
.stButton>button:hover{
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
}
/* 输入框美化 */
.stTextArea>div>textarea{
    border-radius: 12px;
    border: 1px solid #e2e8f0;
    font-size: 15px;
    padding: 1rem;
}
/* 侧边栏整体背景 → 浅黄色 */
[data-testid="stSidebar"]{
    background: #fffbeb !important;
}

/* 侧边栏 Radio 文字（功能导航）加粗深色 */
input[type="radio"] + div,
input[type="radio"] + div * {
    color: #2c3e50 !important;
    font-weight: 700 !important;
    font-size: 16px !important;
}

/* 选中态更突出 */
input[type="radio"]:checked + div {
    color: #667eea !important;
    font-weight: 800 !important;
}

/* 提示框美化 */
.st-alert{
    border-radius: 12px;
    border: none;
}
/* 表格美化 */
.dataframe{
    border-radius: 12px;
    overflow: hidden;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# ===================== 主标题 =====================
st.markdown("<h1>📰 基于 CNN+LSTM 智能新闻分类系统</h1>", unsafe_allow_html=True)

# ===================== 侧边栏导航 =====================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/news.png", width=80)
    st.title("功能导航")
    page = st.radio("请选择功能模块", ["🏠 首页", "✍️ 单条新闻分类", "📁 批量文件预测", "📊 数据可视化"])
    st.divider()

    # 系统信息卡片
    st.markdown("### 📋 系统信息")
    st.info(f"""
    - 运行设备：{DEVICE}
    - 词表大小：{VOCAB_SIZE}
    - 支持分类：{CLASS_NUM} 类
    - 序列长度：{SEQ_LEN}
    """)
    st.divider()

    # 支持的分类列表
    st.markdown("### 🏷️ 支持新闻分类")
    for cat in all_categories:
        st.code(cat, language="text")

    st.divider()
    st.caption("版本：V1.0 | 深度学习分类系统")

# ===================== 页面逻辑分发 =====================
# 1. 首页
if page == "🏠 首页":
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.header("👋 欢迎使用智能新闻分类系统")
    st.write("""
    本系统基于 **CNN + LSTM 混合神经网络** 训练而成，针对中文新闻文本实现自动分类。
    可识别：体育、娱乐、家居、房产、教育、时政、财经、科技、时尚、游戏 十大类目。
    """)
    st.subheader("✨ 系统功能")
    st.markdown("""
    1. **单条新闻识别**：手动输入文本，实时获取分类结果
    2. **批量文件预测**：上传 CSV/TXT 文件，批量处理并导出结果
    3. **数据可视化**：对预测结果进行图表统计分析
    """)
    st.subheader("📖 使用指引")
    st.write("左侧侧边栏选择对应功能模块即可开始使用！")
    st.markdown('</div>', unsafe_allow_html=True)

# 2. 单条新闻分类
elif page == "✍️ 单条新闻分类":
    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("📝 输入新闻文本")
        input_text = st.text_area("请粘贴/输入新闻内容：", height=350)

        st.subheader("💡 快速示例")
        c1, c2, c3 = st.columns(3)
        with c1:
            btn_sport = st.button("体育新闻")
        with c2:
            btn_tech = st.button("科技新闻")
        with c3:
            btn_fin = st.button("财经新闻")

        # 示例填充
        if btn_sport:
            input_text = "国足积极备战国际赛事，队员训练状态良好，期待取得优异成绩。"
        if btn_tech:
            input_text = "新款智能设备搭载新一代AI模型，运算速度与能效全面提升。"
        if btn_fin:
            input_text = "各大上市公司发布季度财报，行业整体经济走势平稳向好。"

        run_btn = st.button("🚀 开始分类", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("📊 预测结果")
        if run_btn:
            if not input_text.strip():
                st.warning("⚠️ 请输入有效新闻文本！")
            else:
                with st.spinner("模型分析中，请稍候..."):
                    res = predict_single(input_text)
                st.success(f"✅ 预测分类：**{res}**")
                st.info(f"📏 文本长度：{len(input_text)} 字符")
        else:
            st.info("点击左侧【开始分类】执行预测")
        st.markdown('</div>', unsafe_allow_html=True)

# 3. 批量文件预测
elif page == "📁 批量文件预测":
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("📂 批量文本预测")
    st.write("支持格式：`CSV` / `TXT` | CSV 文件需包含 `text` 列，TXT 文件每行一条新闻")

    upload_file = st.file_uploader("上传文件", type=["csv", "txt"])
    df_result = None

    if upload_file:
        try:
            if upload_file.name.endswith(".csv"):
                df = pd.read_csv(upload_file)
            else:
                lines = upload_file.read().decode("utf-8").splitlines()
                df = pd.DataFrame({"text": lines})

            st.success("✅ 文件读取成功，预览数据：")
            st.dataframe(df.head(10), use_container_width=True)

            batch_btn = st.button("⚡ 开始批量预测", use_container_width=True)
            if batch_btn:
                with st.spinner("批量预测执行中..."):
                    text_list = df["text"].tolist()
                    pred_list = predict_batch(text_list)
                    df["预测分类"] = pred_list
                    df_result = df

                st.success("🎉 批量预测完成！")
                st.dataframe(df_result, use_container_width=True)

                # 导出文件
                out_buf = StringIO()
                df_result.to_csv(out_buf, index=False, encoding="utf-8-sig")
                st.download_button(
                    label="📥 下载预测结果 CSV",
                    data=out_buf.getvalue(),
                    file_name="新闻分类结果.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"❌ 文件解析失败：{str(e)}")
    else:
        st.info("请上传本地文件开始批量预测")
    st.markdown('</div>', unsafe_allow_html=True)

# 4. 数据可视化
elif page == "📊 数据可视化":
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("📈 预测结果可视化分析")
    st.write("上传已完成预测、包含`预测分类`列的CSV文件，生成统计图表")

    vis_file = st.file_uploader("上传结果文件", type=["csv"])
    if vis_file:
        try:
            df_vis = pd.read_csv(vis_file)
            if "预测分类" not in df_vis.columns:
                st.error("文件缺少【预测分类】列，请上传批量预测后的结果文件！")
            else:
                st.success("✅ 数据加载成功")
                count_data = df_vis["预测分类"].value_counts()

                # 双图布局
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei"]
                plt.rcParams["axes.unicode_minus"] = False

                # 柱状图
                count_data.plot(kind="bar", ax=ax1, color="#667eea")
                ax1.set_title("各分类新闻数量统计", fontsize=14)
                ax1.set_xlabel("新闻分类")
                ax1.set_ylabel("数量")
                ax1.tick_params(axis='x', rotation=45)

                # 饼图
                count_data.plot(kind="pie", ax=ax2, autopct="%1.1f%%", startangle=90,
                                colors=["#667eea", "#764ba2", "#f093fb", "#4facfe", "#00f2fe", "#43e97b", "#fa709a", "#fee140", "#fa709a", "#fee140"])
                ax2.set_title("各分类占比分布", fontsize=14)
                ax2.set_ylabel("")

                plt.tight_layout()
                st.pyplot(fig)

                st.subheader("📋 分类统计详情")
                st.dataframe(count_data, use_container_width=True)
        except Exception as e:
            st.error(f"❌ 数据分析失败：{str(e)}")
    else:
        st.info("请上传批量预测后的 CSV 结果文件")
    st.markdown('</div>', unsafe_allow_html=True)

st.divider()
st.caption("© 2026 深度学习新闻分类系统 | CNN-LSTM 模型赵晓康")