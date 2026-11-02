
import pandas as pd
import plotly.express as px
import streamlit as st


# ==========================================
# 1. 페이지 기본 설정
# ==========================================

st.set_page_config(
    page_title="탐색 | 뇌졸중 예측 실습실",
    page_icon="🔎",
    layout="wide"
)

st.title("🔎 데이터 탐색")
st.write(
    "뇌졸중 데이터를 다양한 그래프와 표로 살펴봅니다. "
    "변수들의 분포와 뇌졸중 발생 사이의 관계를 탐색해 보세요."
)

st.divider()


# ==========================================
# 2. 데이터 불러오기
# ==========================================

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "greatsong/modudata/main/data/stroke.csv"
)


@st.cache_data
def load_data():

    return pd.read_csv(
        DATA_URL,
        encoding="utf-8"
    )


try:
    df = load_data()

except Exception as e:
    st.error("데이터를 불러오지 못했습니다.")
    st.error(f"오류 내용: {e}")
    st.stop()


# 뇌졸중 여부를 알아보기 쉬운 문자로 변환
df["stroke_group"] = df["stroke"].map({
    0: "뇌졸중 없음",
    1: "뇌졸중 있음"
})


# 그래프 색상
COLOR_MAP = {
    "뇌졸중 없음": "#4C9BE8",
    "뇌졸중 있음": "#E76F51"
}


# ==========================================
# 3. 나이와 평균 혈당 분포
# ==========================================

st.subheader("1️⃣ 나이와 평균 혈당의 분포")

st.write(
    "전체 사람들의 나이와 평균 혈당이 "
    "어떤 범위에 많이 분포하는지 확인합니다."
)

col1, col2 = st.columns(2)

with col1:

    fig_age = px.histogram(
        df,
        x="age",
        nbins=30,
        title="나이 분포",
        labels={
            "age": "나이",
            "count": "사람 수"
        },
        color_discrete_sequence=["#4C9BE8"]
    )

    fig_age.update_layout(
        xaxis_title="나이",
        yaxis_title="사람 수",
        bargap=0.05
    )

    st.plotly_chart(
        fig_age,
        use_container_width=True
    )


with col2:

    fig_glucose = px.histogram(
        df,
        x="avg_glucose_level",
        nbins=30,
        title="평균 혈당 분포",
        labels={
            "avg_glucose_level": "평균 혈당",
            "count": "사람 수"
        },
        color_discrete_sequence=["#2A9D8F"]
    )

    fig_glucose.update_layout(
        xaxis_title="평균 혈당",
        yaxis_title="사람 수",
        bargap=0.05
    )

    st.plotly_chart(
        fig_glucose,
        use_container_width=True
    )


st.divider()


# ==========================================
# 4. 뇌졸중 여부에 따른 상자그림
# ==========================================

st.subheader("2️⃣ 뇌졸중 여부에 따른 비교")

st.write(
    "뇌졸중을 겪은 사람과 겪지 않은 사람의 "
    "나이 및 평균 혈당 분포를 상자그림으로 비교합니다."
)

col1, col2 = st.columns(2)


with col1:

    fig_age_box = px.box(
        df,
        x="stroke_group",
        y="age",
        color="stroke_group",
        color_discrete_map=COLOR_MAP,
        title="뇌졸중 여부에 따른 나이 비교",
        labels={
            "stroke_group": "뇌졸중 여부",
            "age": "나이"
        }
    )

    fig_age_box.update_layout(
        showlegend=False,
        xaxis_title="뇌졸중 여부",
        yaxis_title="나이"
    )

    st.plotly_chart(
        fig_age_box,
        use_container_width=True
    )


with col2:

    fig_glucose_box = px.box(
        df,
        x="stroke_group",
        y="avg_glucose_level",
        color="stroke_group",
        color_discrete_map=COLOR_MAP,
        title="뇌졸중 여부에 따른 평균 혈당 비교",
        labels={
            "stroke_group": "뇌졸중 여부",
            "avg_glucose_level": "평균 혈당"
        }
    )

    fig_glucose_box.update_layout(
        showlegend=False,
        xaxis_title="뇌졸중 여부",
        yaxis_title="평균 혈당"
    )

    st.plotly_chart(
        fig_glucose_box,
        use_container_width=True
    )


# 평균값 비교표
age_mean = df.groupby("stroke_group")["age"].mean()
glucose_mean = df.groupby("stroke_group")["avg_glucose_level"].mean()

mean_table = pd.DataFrame({
    "뇌졸중 여부": ["뇌졸중 없음", "뇌졸중 있음"],
    "평균 나이": [
        age_mean.get("뇌졸중 없음"),
        age_mean.get("뇌졸중 있음")
    ],
    "평균 혈당": [
        glucose_mean.get("뇌졸중 없음"),
        glucose_mean.get("뇌졸중 있음")
    ]
})

mean_table["평균 나이"] = mean_table["평균 나이"].round(2)
mean_table["평균 혈당"] = mean_table["평균 혈당"].round(2)

st.markdown("#### 그룹별 평균값 비교")

st.dataframe(
    mean_table,
    use_container_width=True,
    hide_index=True
)


st.divider()


# ==========================================
# 5. 고혈압 유무에 따른 뇌졸중 비율
# ==========================================

st.subheader("3️⃣ 고혈압과 뇌졸중 발생 비율")

st.write(
    "고혈압이 있는 사람과 없는 사람의 "
    "뇌졸중 발생 비율을 비교합니다."
)


hypertension_table = (
    df.groupby("hypertension")["stroke"]
    .agg(["mean", "count"])
    .reset_index()
)

hypertension_table["stroke_ratio"] = (
    hypertension_table["mean"] * 100
)

hypertension_table["고혈압"] = (
    hypertension_table["hypertension"]
    .map({
        0: "고혈압 없음",
        1: "고혈압 있음"
    })
)

fig_hypertension = px.bar(
    hypertension_table,
    x="고혈압",
    y="stroke_ratio",
    text=hypertension_table["stroke_ratio"].map(
        lambda x: f"{x:.2f}%"
    ),
    title="고혈압 유무별 뇌졸중 발생 비율",
    labels={
        "고혈압": "고혈압 여부",
        "stroke_ratio": "뇌졸중 발생 비율(%)"
    },
    color="고혈압",
    color_discrete_sequence=["#4C9BE8", "#E76F51"]
)

fig_hypertension.update_traces(
    textposition="outside"
)

fig_hypertension.update_layout(
    showlegend=False,
    yaxis_title="뇌졸중 발생 비율(%)",
    xaxis_title="고혈압 여부",
    yaxis_range=[
        0,
        max(1, hypertension_table["stroke_ratio"].max() * 1.2)
    ]
)

st.plotly_chart(
    fig_hypertension,
    use_container_width=True
)


# ==========================================
# 6. 심장병 유무에 따른 뇌졸중 비율
# ==========================================

st.subheader("4️⃣ 심장병과 뇌졸중 발생 비율")

st.write(
    "심장병이 있는 사람과 없는 사람의 "
    "뇌졸중 발생 비율을 비교합니다."
)


heart_table = (
    df.groupby("heart_disease")["stroke"]
    .agg(["mean", "count"])
    .reset_index()
)

heart_table["stroke_ratio"] = (
    heart_table["mean"] * 100
)

heart_table["심장병"] = (
    heart_table["heart_disease"]
    .map({
        0: "심장병 없음",
        1: "심장병 있음"
    })
)

fig_heart = px.bar(
    heart_table,
    x="심장병",
    y="stroke_ratio",
    text=heart_table["stroke_ratio"].map(
        lambda x: f"{x:.2f}%"
    ),
    title="심장병 유무별 뇌졸중 발생 비율",
    labels={
        "심장병": "심장병 여부",
        "stroke_ratio": "뇌졸중 발생 비율(%)"
    },
    color="심장병",
    color_discrete_sequence=["#4C9BE8", "#E76F51"]
)

fig_heart.update_traces(
    textposition="outside"
)

fig_heart.update_layout(
    showlegend=False,
    yaxis_title="뇌졸중 발생 비율(%)",
    xaxis_title="심장병 여부",
    yaxis_range=[
        0,
        max(1, heart_table["stroke_ratio"].max() * 1.2)
    ]
)

st.plotly_chart(
    fig_heart,
    use_container_width=True
)


st.divider()


# ==========================================
# 7. BMI 빈 값 분석
# ==========================================

st.subheader("5️⃣ BMI가 비어 있는 사람들의 뇌졸중 비율")

st.write(
    "체질량지수(BMI)가 기록되지 않은 사람들을 따로 구분하여 "
    "전체 데이터의 뇌졸중 발생 비율과 비교합니다."
)


# BMI 결측자만 추출
missing_bmi = df[df["bmi"].isna()]

missing_bmi_count = len(missing_bmi)

missing_bmi_stroke_count = int(
    (missing_bmi["stroke"] == 1).sum()
)

missing_bmi_ratio = (
    missing_bmi_stroke_count / missing_bmi_count * 100
    if missing_bmi_count > 0
    else 0
)

total_stroke_count = int(
    (df["stroke"] == 1).sum()
)

total_stroke_ratio = (
    total_stroke_count / len(df) * 100
    if len(df) > 0
    else 0
)


bmi_table = pd.DataFrame({
    "구분": [
        "BMI 빈 값인 사람",
        "전체 사람"
    ],
    "사람 수": [
        missing_bmi_count,
        len(df)
    ],
    "뇌졸중 발생 인원": [
        missing_bmi_stroke_count,
        total_stroke_count
    ],
    "뇌졸중 발생 비율(%)": [
        round(missing_bmi_ratio, 2),
        round(total_stroke_ratio, 2)
    ]
})

st.dataframe(
    bmi_table,
    use_container_width=True,
    hide_index=True
)


st.info(
    f"BMI가 비어 있는 사람은 총 {missing_bmi_count:,}명이며, "
    f"그중 뇌졸중을 겪은 사람은 "
    f"{missing_bmi_stroke_count:,}명입니다."
)


st.divider()


# ==========================================
# 8. 흡연 상태별 인원수
# ==========================================

st.subheader("6️⃣ 흡연 상태별 사람 수")

st.write(
    "흡연 상태(smoking_status)에 어떤 값들이 있는지 "
    "확인하고 각 상태에 해당하는 사람 수를 세어 봅니다."
)


smoking_table = (
    df["smoking_status"]
    .value_counts(dropna=False)
    .reset_index()
)

smoking_table.columns = [
    "흡연 상태",
    "사람 수"
]

# 결측값 표시
smoking_table["흡연 상태"] = (
    smoking_table["흡연 상태"]
    .fillna("빈 값")
)

smoking_table["비율(%)"] = (
    smoking_table["사람 수"] / len(df) * 100
).round(2)


st.dataframe(
    smoking_table,
    use_container_width=True,
    hide_index=True
)


# ==========================================
# 9. 마무리 안내
# ==========================================

st.divider()

st.caption(
    "🧠 뇌졸중 예측 실습실 | 데이터 탐색 페이지"
)

st.caption(
    "※ 그래프와 표는 데이터의 통계적 특성을 보여줍니다. "
    "두 변수 사이의 관계만으로 인과관계를 판단할 수 없습니다."
)

