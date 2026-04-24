"""
AI 動物判定 Streamlit アプリ
実行: streamlit run app.py
"""

import os
import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import tensorflow as tf

# ── 設定 ────────────────────────────────────────────────────
CATEGORIES = [
    {'name': 'キリン',    'emoji': '🦒'},
    {'name': 'うさぎ',   'emoji': '🐰'},
    {'name': 'ぞう',     'emoji': '🐘'},
    {'name': 'ライオン', 'emoji': '🦁'},
    {'name': 'ワニ',     'emoji': '🐊'},
]
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'model.h5')

# ── ページ設定 ───────────────────────────────────────────────
st.set_page_config(page_title='AI 動物判定', page_icon='🎨', layout='centered')

st.markdown("""
<style>
  /* HeaderSpace / PageTitleArea */
  .block-container { padding-top: 56px !important; }
  h1 { font-size: 20pt !important; }

  /* ボタン（Primary: Success #34E87D / Secondary: ボーダー） */
  div.stButton > button[kind="primary"] {
    background: #34E87D;
    color: #0f172a;
    border: none;
    border-radius: 6px;
    height: 60px;
    font-size: 14px;
    font-weight: 700;
    width: 100%;
  }
  div.stButton > button[kind="secondary"] {
    background: #ffffff;
    color: #475569;
    border: 1.5px solid #cbd5e1;
    border-radius: 6px;
    height: 38px;
    font-size: 14px;
    font-weight: 700;
    width: 100%;
  }
  div.stButton > button:hover { opacity: 0.82; }
</style>
""", unsafe_allow_html=True)

# ── モデル読み込み ───────────────────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# ── UI ──────────────────────────────────────────────────────
st.title('🎨 AI 動物判定')
st.caption('キリン・うさぎ・ぞう・ライオン・ワニを認識します')

if model is None:
    st.error('model/model.h5 が見つかりません。先に `python train.py` を実行してください。')
    st.stop()

# 消すボタン用カウンター（キーを変えてキャンバスを強制リセット）
if 'clear_count' not in st.session_state:
    st.session_state.clear_count = 0

col_input, col_result = st.columns([1, 1], gap='large')

with col_input:
    st.markdown('<p style="font-size:12px;color:#94a3b8;">マウスまたはタッチで描いてください</p>',
                unsafe_allow_html=True)

    canvas = st_canvas(
        fill_color='#ffffff',
        stroke_width=7,
        stroke_color='#111111',
        background_color='#ffffff',
        width=280,
        height=280,
        drawing_mode='freedraw',
        key=f'canvas_{st.session_state.clear_count}',
    )

    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        clear = st.button('消す', type='secondary', use_container_width=True)
    with btn_col2:
        predict_btn = st.button('判定する', type='primary', use_container_width=True)

    if clear:
        st.session_state.clear_count += 1
        st.rerun()

# ── 判定 ────────────────────────────────────────────────────
with col_result:
    st.markdown('<p style="font-size:12px;font-weight:700;color:#94a3b8;text-transform:uppercase;letter-spacing:0.05em;">判定結果</p>',
                unsafe_allow_html=True)

    if predict_btn and canvas.image_data is not None:
        # 前処理: グレースケール → 28x28 → 反転 → 正規化
        img = Image.fromarray(canvas.image_data.astype('uint8')).convert('L')
        img = img.resize((28, 28))
        arr = (255 - np.array(img)) / 255.0
        arr = arr.reshape(1, 28, 28, 1).astype('float32')

        probs = model.predict(arr, verbose=0)[0]
        results = sorted(
            [{'name': CATEGORIES[i]['name'], 'emoji': CATEGORIES[i]['emoji'], 'prob': float(probs[i])}
             for i in range(len(CATEGORIES))],
            key=lambda x: x['prob'], reverse=True
        )
        top = results[0]

        # 1位表示
        st.markdown(f"""
        <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:6px;
                    padding:16px;text-align:center;margin-bottom:12px;">
          <div style="font-size:48px;line-height:1;">{top['emoji']}</div>
          <div style="font-size:20px;font-weight:800;color:#1e293b;margin-top:8px;">{top['name']}</div>
          <div style="font-size:12px;color:#64748b;margin-top:8px;">{top['prob']*100:.1f}% の確信度</div>
        </div>
        """, unsafe_allow_html=True)

        # バー表示
        for i, r in enumerate(results):
            pct = r['prob'] * 100
            bar_color = 'linear-gradient(90deg,#34E87D,#B752E2)' if i == 0 else '#cbd5e1'
            st.markdown(f"""
            <div style="margin-bottom:8px;">
              <div style="display:flex;justify-content:space-between;font-size:12px;color:#1e293b;margin-bottom:3px;">
                <span>{r['emoji']} {r['name']}</span>
                <span style="color:#64748b;">{pct:.1f}%</span>
              </div>
              <div style="height:8px;background:#f1f5f9;border-radius:99px;overflow:hidden;">
                <div style="height:100%;width:{pct:.1f}%;background:{bar_color};border-radius:99px;"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:6px;
                    padding:16px;text-align:center;min-height:80px;
                    display:flex;align-items:center;justify-content:center;">
          <p style="font-size:12px;color:#94a3b8;">判定するを押してください</p>
        </div>
        """, unsafe_allow_html=True)
