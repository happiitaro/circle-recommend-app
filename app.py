import streamlit as st
from sentence_transformers import SentenceTransformer, util
import json
import numpy as np

# モデル読み込み
model = SentenceTransformer("intfloat/multilingual-e5-large")

# サークルデータ読み込み
with open("circles.json", "r", encoding="utf-8") as f:
    circles = json.load(f)

# 事前計算したベクトル読み込み
with open("circle_embeddings.json", "r", encoding="utf-8") as f:
    circle_embs = json.load(f)
    circle_embs = np.array(circle_embs, dtype=np.float32)

# numpy に変換（高速化）
circle_embs = np.array(circle_embs)

# ---------------------------------------------------------
# ① カテゴリ代表文（ゼロショット分類用）
# ---------------------------------------------------------
category_texts = {
    "音楽": "音楽、楽器、演奏、歌、バンド、音を出す活動",
    "スポーツ": "運動、スポーツ、体を動かす、競技、試合",
    "アート": "絵、デザイン、創作、アート、工作、クラフト",
    "文化": "文化、歴史、学習、研究、読書、知識",
    "手芸": "手作り、編み物、裁縫、クラフト、手芸",
}

# カテゴリ埋め込みを事前計算
category_embeddings = {
    cat: model.encode(text)
    for cat, text in category_texts.items()
}

# ---------------------------------------------------------
# ② カテゴリ自動判定
# ---------------------------------------------------------
def detect_category(query):
    query_vec = model.encode(query)

    best_cat = None
    best_score = -1

    for cat, cat_vec in category_embeddings.items():
        score = np.dot(query_vec, cat_vec) / (
            np.linalg.norm(query_vec) * np.linalg.norm(cat_vec)
        )
        if score > best_score:
            best_score = score
            best_cat = cat

    return best_cat, best_score


# ---------------------------------------------------------
# ③ スコア補正ロジック
# ---------------------------------------------------------
def compute_score(query_embedding, circle, detected_category):
    # ベースのコサイン類似度
    score = float(
        np.dot(query_embedding, circle["embedding"]) /
        (np.linalg.norm(query_embedding) * np.linalg.norm(circle["embedding"]))
    )

    # カテゴリ補正（タグに一致したら加点）
    if detected_category in circle["tags"]:
        score += 0.25  # ← 調整可能

    return score


# ---------------------------------------------------------
# ④ 検索処理（カテゴリ補正付き）
# ---------------------------------------------------------
def search_circles(query):
    query_embedding = model.encode(query)

    # カテゴリ自動判定
    detected_category, cat_score = detect_category(query)

    scored_results = []
    for circle, emb in zip(circles, circle_embs):
        circle_data = {
            "name": circle["name"],
            "description": circle["description"],
            "tags": circle["tags"],
            "embedding": emb,
        }
        score = compute_score(query_embedding, circle_data, detected_category)
        scored_results.append((score, circle))

    scored_results.sort(reverse=True, key=lambda x: x[0])
    return scored_results[:5]


# ---------------------------------------------------------
# UI（あなたのCSSはそのまま）
# ---------------------------------------------------------
st.markdown("""
<style>
body {
    background-color: #fffdf5;
    font-family: "Rounded Mplus 1c", "Hiragino Maru Gothic ProN", sans-serif;
}
h1 {
    color: #ff8c94;
    text-align: center;
    font-size: 42px !important;
    font-weight: bold;
    margin-bottom: 20px;
}
input[type="text"] {
    border: 2px solid #ffb6c1;
    border-radius: 10px;
    padding: 10px;
    font-size: 18px;
}
div.stButton > button:first-child {
    background-color: #ffb6c1;
    color: white;
    border-radius: 10px;
    padding: 10px 20px;
    border: none;
    font-size: 20px;
    font-weight: bold;
    box-shadow: 2px 2px 4px rgba(0,0,0,0.2);
}
div.stButton > button:first-child:hover {
    background-color: #ff9aa2;
}
h3 {
    font-family: "Rounded Mplus 1c", sans-serif;
}
</style>
""", unsafe_allow_html=True)

st.title("稲城市公民館サークルおすすめAI")
st.write("サークル数:", len(circles))
st.caption("例：サッカーがしたい、絵を描きたい、友だちを作りたい など")

def get_icon(tags):
    if "スポーツ" in tags:
        return "⚽"
    if "音楽" in tags:
        return "🎵"
    if "アート" in tags:
        return "🎨"
    return "🌟"

# ---------------------------------------------------------
# ENTERキーで検索できるフォーム
# ---------------------------------------------------------
with st.form("search_form"):
    query = st.text_input("キーワードを入力してね")
    submitted = st.form_submit_button("検索")

# ---------------------------------------------------------
# 検索実行
# ---------------------------------------------------------
if submitted:
    if query.strip() == "":
        st.warning("キーワードを入力してね")
    else:
        results = search_circles(query)

        st.subheader("おすすめのサークル")

        if len(results) == 0:
            st.info("該当するサークルが見つかりませんでした")
        else:
            for score, c in results:
                st.markdown(
                    f"""
                    <div style="
                        background-color: #fff8dc;
                        padding: 15px;
                        border-radius: 12px;
                        margin-bottom: 15px;
                        border: 2px solid #f4d06f;
                        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
                    ">
                        <h3 style="color:#d17b0f;">{get_icon(c['tags'])} {c['name']}</h3>
                        <p style="margin:0 0 8px 0; color:#444;">{c['description']}</p>
                        <p style="font-size:14px; color:#888;">スコア: {score:.3f}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )