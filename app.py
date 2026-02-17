from sentence_transformers import SentenceTransformer, util

# クエリ用の軽量モデルを読み込む（これはOK）
model = SentenceTransformer("intfloat/multilingual-e5-large")

# 事前計算済みのベクトルを読み込む
with open("circle_embeddings.json", "r", encoding="utf-8") as f:
    circle_embs = json.load(f)
circle_embs = np.array(circle_embs)

# circles.json も読み込む
with open("circles.json", "r", encoding="utf-8") as f:
    circles = json.load(f)

st.markdown("""
<style>

body {
    background-color: #fffdf5;
    font-family: "Rounded Mplus 1c", "Hiragino Maru Gothic ProN", sans-serif;
}

/* タイトル */
h1 {
    color: #ff8c94;
    text-align: center;
    font-size: 42px !important;
    font-weight: bold;
    margin-bottom: 20px;
}

/* 入力欄 */
input[type="text"] {
    border: 2px solid #ffb6c1;
    border-radius: 10px;
    padding: 10px;
    font-size: 18px;
}

/* ボタン */
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

/* サブタイトル */
h3 {
    font-family: "Rounded Mplus 1c", sans-serif;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>

/* 右上メニュー（…） */
#MainMenu {visibility: hidden !important;}

/* フッター */
footer {visibility: hidden !important;}

/* 右上のツールバー全体 */
header [data-testid="stToolbar"] {display: none !important;}

/* 右下の管理バー（複数パターン） */
[data-testid="stAppStatusWidget"] {display: none !important;}
[data-testid="stStatusWidget"] {display: none !important;}
[data-testid="stStatusContainer"] {display: none !important;}
section[data-testid="stSidebar"] + div {display: none !important;}

</style>
""", unsafe_allow_html=True)
st.title("稲城市公民館サークルおすすめAI")
st.write("サークル数:", len(circles))
st.write("やりたいことを入力すると、おすすめのサークルを教えるよ")

user_input = st.text_input("やりたいことを入力してね")
st.caption("例：サッカーがしたい、絵を描きたい、友だちを作りたい など")
def get_icon(tags):
    if "スポーツ" in tags:
        return "⚽"
    if "音楽" in tags:
        return "🎵"
    if "アート" in tags:
        return "🎨"
    return "🌟"
st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #ffcc66;
    color: #8a4b00;
    border-radius: 10px;
    padding: 10px 20px;
    border: 2px solid #f4b400;
    font-size: 18px;
    font-weight: bold;
}
div.stButton > button:first-child:hover {
    background-color: #ffdd88;
}
</style>
""", unsafe_allow_html=True)

if st.button("検索"):
    query = "query: " + user_input
    query_emb = model.encode(query)

    # コサイン類似度を一括計算（高速）
    scores = util.cos_sim(query_emb, circle_embs)[0].tolist()

    # スコアとサークルをまとめてソート
    results = sorted(
        zip(scores, circles),
        key=lambda x: x[0],
        reverse=True
    )

    st.subheader("おすすめのサークル")
#    for score, c in results[:5]:
#        st.write(f"### {c['name']}")
#        st.write(c["description"])
#        st.write(f"スコア: {score:.3f}")
#        st.write("---")
    for score, c in results[:5]:
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