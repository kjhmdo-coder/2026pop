import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances

# -----------------------------
# 페이지 설정
# -----------------------------
st.set_page_config(page_title="인구 구조 쌍둥이 지역 찾기", layout="wide")
st.title("🌍 인구 구조 쌍둥이 지역 찾기")
st.markdown("궁금한 지역과 인구 구조가 가장 비슷한 지역을 찾아드립니다.")

# -----------------------------
# 데이터 로드
# -----------------------------
import os
import streamlit as st
import pandas as pd

@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "202606_202606_연령별인구현황_월간.csv")

    encodings_to_try = ["cp949", "euc-kr", "utf-8-sig", "utf-8"]

    df = None
    for enc in encodings_to_try:
        try:
            df = pd.read_csv(file_path, encoding=enc)
            break
        except UnicodeDecodeError:
            continue

    if df is None:
        st.error("모든 인코딩 시도에 실패했습니다.")
        st.stop()

    # 쉼표 제거 후 숫자 변환
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.replace(",", "", regex=False)
            df[col] = pd.to_numeric(df[col], errors="coerce")  # ✅ "ignore" → "coerce"

    return df

df = load_data()
# 지역 컬럼명 (데이터에 맞게 수정 필요)
region_col = "지역"

# 연령대 컬럼들 (지역 컬럼 제외 나머지 전부를 연령대로 가정)
age_cols = [col for col in df.columns if col != region_col]

st.subheader("📊 원본 데이터 미리보기")
st.dataframe(df, use_container_width=True)

# -----------------------------
# 인구 구조 비율로 정규화 (총인구 대비 비율)
# -----------------------------
df_ratio = df.copy()
df_ratio[age_cols] = df[age_cols].div(df[age_cols].sum(axis=1), axis=0)

# -----------------------------
# 지역 선택
# -----------------------------
st.subheader("🔍 지역 선택")
selected_region = st.selectbox("궁금한 지역을 선택하세요", df[region_col].unique())

# -----------------------------
# 유사도 계산 (유클리드 거리 기반)
# -----------------------------
scaler = StandardScaler()
scaled = scaler.fit_transform(df_ratio[age_cols])

scaled_df = pd.DataFrame(scaled, columns=age_cols)
scaled_df[region_col] = df_ratio[region_col].values

target_idx = scaled_df[scaled_df[region_col] == selected_region].index[0]
target_vector = scaled_df.loc[target_idx, age_cols].values.reshape(1, -1)

distances = euclidean_distances(scaled_df[age_cols].values, target_vector).flatten()

result_df = df[[region_col]].copy()
result_df["유사도_거리"] = distances
result_df = result_df[result_df[region_col] != selected_region]
result_df = result_df.sort_values("유사도_거리").reset_index(drop=True)

# 거리가 작을수록 유사 → 유사도 점수로 변환 (0~100)
max_dist = result_df["유사도_거리"].max()
result_df["유사도_점수"] = (1 - result_df["유사도_거리"] / max_dist) * 100

# -----------------------------
# 쌍둥이 지역 Top N 출력
# -----------------------------
st.subheader(f"🎯 '{selected_region}'과 인구 구조가 가장 비슷한 지역 Top 5")

top_n = st.slider("몇 개의 지역을 보시겠어요?", min_value=3, max_value=15, value=5)

top_matches = result_df.head(top_n)
st.dataframe(top_matches, use_container_width=True)

# -----------------------------
# 인터랙티브 시각화 - 막대그래프 (유사도 순위)
# -----------------------------
fig_bar = px.bar(
    top_matches,
    x="유사도_점수",
    y=region_col,
    orientation="h",
    color="유사도_점수",
    color_continuous_scale="Blues",
    text="유사도_점수",
    title=f"'{selected_region}'과 유사한 지역 순위",
)
fig_bar.update_traces(texttemplate='%{text:.1f}', textposition='outside')
fig_bar.update_layout(yaxis={'categoryorder': 'total ascending'}, height=500)
st.plotly_chart(fig_bar, use_container_width=True)

# -----------------------------
# 인터랙티브 시각화 - 인구 피라미드/구조 비교 (선택 지역 vs Top 매칭 지역)
# -----------------------------
st.subheader("📈 인구 구조 비교")

compare_region = st.selectbox(
    "비교할 쌍둥이 지역을 선택하세요",
    top_matches[region_col].tolist()
)

compare_df = pd.DataFrame({
    "연령대": age_cols,
    selected_region: df_ratio.loc[df_ratio[region_col] == selected_region, age_cols].values.flatten() * 100,
    compare_region: df_ratio.loc[df_ratio[region_col] == compare_region, age_cols].values.flatten() * 100,
})

fig_line = go.Figure()
fig_line.add_trace(go.Scatter(
    x=compare_df["연령대"], y=compare_df[selected_region],
    mode='lines+markers', name=selected_region,
    line=dict(width=3)
))
fig_line.add_trace(go.Scatter(
    x=compare_df["연령대"], y=compare_df[compare_region],
    mode='lines+markers', name=compare_region,
    line=dict(width=3, dash='dash')
))
fig_line.update_layout(
    title=f"'{selected_region}' vs '{compare_region}' 연령대별 인구 비율(%) 비교",
    xaxis_title="연령대",
    yaxis_title="인구 비율 (%)",
    hovermode="x unified",
    height=500
)
st.plotly_chart(fig_line, use_container_width=True)

# -----------------------------
# 전체 지역 산점도 (인구 구조 유사도 시각화 - 2차원 축소는 생략, 주요 2개 컬럼 활용 예시)
# -----------------------------
st.subheader("🗺️ 전체 지역 분포 (참고용 산점도)")

if len(age_cols) >= 2:
    fig_scatter = px.scatter(
        df_ratio,
        x=age_cols[0],
        y=age_cols[-1],
        hover_name=region_col,
        color=df_ratio[region_col].apply(lambda x: "선택 지역" if x == selected_region else "기타"),
        title="지역별 인구 구조 산점도 (예시 두 연령대 기준)",
        size_max=15
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
