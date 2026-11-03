
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


# ==========================================
# 1. 페이지 기본 설정
# ==========================================

st.set_page_config(
    page_title="분류 모델 | 뇌졸중 예측 실습실",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 분류 모델")
st.write(
    "머신러닝의 분류 모델을 이용하여 뇌졸중 발생 여부를 예측합니다. "
    "로지스틱 회귀와 의사결정트리의 예측 결과를 비교하고 "
    "각 모델이 어떤 방식으로 판단하는지 살펴보세요."
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


# 사용할 속성
FEATURE_NAMES = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병"
}

FEATURES = list(FEATURE_NAMES.keys())

# 뇌졸중을 양성(1)으로 설정
TARGET = "stroke"

df = df.copy()

# 번호 순으로 정렬
if "id" in df.columns:
    df = df.sort_values(
        by="id"
    ).reset_index(drop=True)
else:
    df = df.reset_index(drop=True)

# 필요한 데이터만 숫자형으로 변환
for col in FEATURES + [TARGET]:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

# 타깃 결측값 제거
df = df.dropna(
    subset=[TARGET]
).reset_index(drop=True)

df[TARGET] = df[TARGET].astype(int)

# 잘못된 타깃 값 방지
df = df[df[TARGET].isin([0, 1])].reset_index(drop=True)


# ==========================================
# 3. 입력 속성 선택
# ==========================================

st.subheader("1️⃣ 예측에 사용할 속성 선택")

st.write(
    "뇌졸중 발생 여부를 예측할 때 사용할 속성을 선택하세요. "
    "처음에는 체질량지수를 제외한 네 가지 속성이 선택되어 있습니다."
)

selected_features = st.multiselect(
    "예측에 사용할 속성",
    options=FEATURES,
    default=[
        "age",
        "avg_glucose_level",
        "hypertension",
        "heart_disease"
    ],
    format_func=lambda x: FEATURE_NAMES[x]
)

if len(selected_features) < 2:
    st.warning(
        "예측에 사용할 속성을 두 개 이상 선택해야 합니다. "
        "속성을 추가로 선택해 주세요."
    )
    st.stop()


st.info(
    f"선택한 속성: "
    f"{', '.join(FEATURE_NAMES[x] for x in selected_features)}"
)


# ==========================================
# 4. 학습 데이터와 테스트 데이터 분리
# ==========================================

st.subheader("2️⃣ 학습용 데이터와 테스트용 데이터 분리")

st.write(
    "데이터를 번호 순으로 정렬한 뒤 열 명씩 묶습니다. "
    "각 묶음의 앞 세 명은 테스트용으로, "
    "나머지 일곱 명은 학습용으로 사용합니다."
)

# 10명씩 묶었을 때 앞의 3명은 테스트용
group_number = np.arange(len(df)) // 10
position_in_group = np.arange(len(df)) % 10

test_mask = position_in_group < 3

train_df = df.loc[~test_mask].copy()
test_df = df.loc[test_mask].copy()

X_train = train_df[selected_features].copy()
X_test = test_df[selected_features].copy()

y_train = train_df[TARGET].copy()
y_test = test_df[TARGET].copy()


# BMI를 선택한 경우에만 훈련용 중앙값으로 결측값 대체
bmi_median = None

if "bmi" in selected_features:

    bmi_median = X_train["bmi"].median()

    X_train["bmi"] = X_train["bmi"].fillna(
        bmi_median
    )

    X_test["bmi"] = X_test["bmi"].fillna(
        bmi_median
    )


# 다른 선택 속성에 결측값이 있는 경우 확인
if X_train.isna().any().any() or X_test.isna().any().any():
    st.error(
        "선택한 속성에 BMI 이외의 결측값이 있습니다. "
        "데이터를 확인해 주세요."
    )
    st.stop()


# 모델 입력값은 숫자형으로 유지
X_train = X_train.astype(float)
X_test = X_test.astype(float)


# 데이터 분리 결과
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "전체 데이터",
        f"{len(df):,}명"
    )

with col2:
    st.metric(
        "훈련 데이터",
        f"{len(train_df):,}명"
    )

with col3:
    st.metric(
        "테스트 데이터",
        f"{len(test_df):,}명"
    )


if len(test_df) != 1533:
    st.warning(
        f"테스트 데이터가 1,533명이 아닙니다. "
        f"현재 테스트 데이터: {len(test_df):,}명"
    )


if bmi_median is not None:
    st.caption(
        f"체질량지수 결측값은 훈련 데이터의 중앙값 "
        f"{bmi_median:.2f}으로 대체했습니다."
    )
else:
    st.caption(
        "체질량지수를 선택하지 않았으므로 BMI 결측값은 "
        "모델 학습에 사용하지 않았습니다."
    )


st.divider()


# ==========================================
# 5. 로지스틱 회귀 모델
# ==========================================

st.subheader(
    "3️⃣ 로지스틱 회귀(확률로 답하는 모델)"
)

st.write(
    "로지스틱 회귀는 입력된 속성을 바탕으로 "
    "뇌졸중이 발생할 확률을 계산합니다. "
    "확률이 0.5 이상이면 뇌졸중 발생(1), "
    "0.5 미만이면 뇌졸중 없음(0)으로 분류합니다."
)

logistic_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        max_iter=2000,
        random_state=42
    ))
])

logistic_model.fit(
    X_train,
    y_train
)

logistic_train_pred = logistic_model.predict(
    X_train
)

logistic_test_pred = logistic_model.predict(
    X_test
)

logistic_train_accuracy = accuracy_score(
    y_train,
    logistic_train_pred
)

logistic_test_accuracy = accuracy_score(
    y_test,
    logistic_test_pred
)


# ==========================================
# 6. 의사결정트리 모델
# ==========================================

st.subheader(
    "4️⃣ 의사결정트리(질문으로 답하는 모델)"
)

st.write(
    "의사결정트리는 속성에 관한 질문을 반복하며 "
    "사람들을 여러 그룹으로 나눕니다. "
    "질문을 세 번까지만 하고, "
    "한 마디에 사람이 다섯 명 미만이면 더 나누지 않습니다."
)

tree_model = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=5,
    random_state=42
)

tree_model.fit(
    X_train,
    y_train
)

tree_train_pred = tree_model.predict(
    X_train
)

tree_test_pred = tree_model.predict(
    X_test
)

tree_train_accuracy = accuracy_score(
    y_train,
    tree_train_pred
)

tree_test_accuracy = accuracy_score(
    y_test,
    tree_test_pred
)


# ==========================================
# 7. 다수 클래스 기준 모델
# ==========================================

st.subheader("5️⃣ 기준 모델과 정확도 비교")

st.write(
    "기준 모델은 입력 속성을 전혀 사용하지 않습니다. "
    "훈련 데이터에서 더 많은 쪽의 결과만 항상 답합니다. "
    "이 모델과 두 머신러닝 모델의 정확도를 비교합니다."
)

# 훈련 데이터에서 더 많은 클래스 선택
majority_class = int(
    y_train.value_counts().idxmax()
)

# 훈련 데이터와 테스트 데이터 모두 같은 답을 예측
baseline_train_pred = np.full(
    len(y_train),
    majority_class
)

baseline_test_pred = np.full(
    len(y_test),
    majority_class
)

baseline_train_accuracy = accuracy_score(
    y_train,
    baseline_train_pred
)

baseline_test_accuracy = accuracy_score(
    y_test,
    baseline_test_pred
)

majority_name = (
    "뇌졸중 있음"
    if majority_class == 1
    else "뇌졸중 없음"
)


# 정확도 카드
col1, col2, col3 = st.columns(3)

with col1:

    st.markdown(
        "#### 로지스틱 회귀"
    )

    st.metric(
        "테스트 데이터 정확도",
        f"{logistic_test_accuracy:.2%}"
    )

    st.caption(
        f"훈련 데이터 정확도: {logistic_train_accuracy:.2%}"
    )

    st.caption(
        f"테스트 데이터 정확도: {logistic_test_accuracy:.2%}"
    )


with col2:

    st.markdown(
        "#### 의사결정트리"
    )

    st.metric(
        "테스트 데이터 정확도",
        f"{tree_test_accuracy:.2%}"
    )

    st.caption(
        f"훈련 데이터 정확도: {tree_train_accuracy:.2%}"
    )

    st.caption(
        f"테스트 데이터 정확도: {tree_test_accuracy:.2%}"
    )


with col3:

    st.markdown(
        "#### 기준 모델"
    )

    st.metric(
        "테스트 데이터 정확도",
        f"{baseline_test_accuracy:.2%}"
    )

    st.caption(
        f"훈련 데이터 정확도: {baseline_train_accuracy:.2%}"
    )

    st.caption(
        f"테스트 데이터 정확도: {baseline_test_accuracy:.2%}"
    )

    st.caption(
        f"항상 '{majority_name}'으로 예측"
    )


st.info(
    "정확도는 전체 예측 중 정답을 맞힌 비율입니다. "
    "뇌졸중 데이터는 발생한 사람과 발생하지 않은 사람의 수에 "
    "차이가 있으므로 정확도만으로 모델의 성능을 판단하는 데는 "
    "한계가 있습니다."
)


st.divider()


# ==========================================
# 8. 두 속성을 이용한 분류 결과 시각화
# ==========================================

st.subheader("6️⃣ 두 속성으로 살펴보는 분류 결과")

st.write(
    "선택한 속성 가운데 두 가지를 가로축과 세로축으로 지정합니다. "
    "테스트 데이터를 실제 뇌졸중 여부에 따라 색으로 구분하고, "
    "두 모델이 공간을 어떻게 나누는지 확인합니다."
)


# 가로축과 세로축 선택
axis_col1, axis_col2 = st.columns(2)

with axis_col1:

    x_feature = st.selectbox(
        "가로축 속성",
        options=selected_features,
        index=0,
        format_func=lambda x: FEATURE_NAMES[x],
        key="x_feature"
    )

with axis_col2:

    y_options = [
        x for x in selected_features
        if x != x_feature
    ]

    default_y_index = 0

    if len(selected_features) > 2:
        default_y_index = min(
            1,
            len(y_options) - 1
        )

    y_feature = st.selectbox(
        "세로축 속성",
        options=y_options,
        index=default_y_index,
        format_func=lambda x: FEATURE_NAMES[x],
        key="y_feature"
    )


# 두 축 이외의 속성은 테스트 데이터 중앙값으로 고정
fixed_values = {}

for feature in selected_features:

    if feature not in [x_feature, y_feature]:

        fixed_values[feature] = float(
            X_test[feature].median()
        )


st.markdown("#### 그래프에서 고정한 나머지 속성")

if fixed_values:

    fixed_text = ", ".join(
        f"{FEATURE_NAMES[key]}: {value:.2f}"
        for key, value in fixed_values.items()
    )

    st.write(
        f"두 축에 사용하지 않은 속성은 테스트 데이터의 "
        f"중앙값으로 고정했습니다. {fixed_text}"
    )

else:

    st.write(
        "선택한 속성이 두 개이므로 고정할 나머지 속성이 없습니다."
    )


# ==========================================
# 9. 예측 영역 계산
# ==========================================

# 테스트 데이터의 두 축 범위 확인
x_min = float(X_test[x_feature].min())
x_max = float(X_test[x_feature].max())

y_min = float(X_test[y_feature].min())
y_max = float(X_test[y_feature].max())

# 그래프 여백
x_padding = max(
    (x_max - x_min) * 0.08,
    0.1
)

y_padding = max(
    (y_max - y_min) * 0.08,
    0.1
)

x_grid = np.linspace(
    x_min - x_padding,
    x_max + x_padding,
    150
)

y_grid = np.linspace(
    y_min - y_padding,
    y_max + y_padding,
    150
)

xx, yy = np.meshgrid(
    x_grid,
    y_grid
)

# 모든 입력 속성을 준비하고 나머지는 중앙값으로 고정
grid_data = pd.DataFrame(
    index=np.arange(xx.size),
    columns=selected_features,
    dtype=float
)

for feature in selected_features:

    if feature == x_feature:
        grid_data[feature] = xx.ravel()

    elif feature == y_feature:
        grid_data[feature] = yy.ravel()

    else:
        grid_data[feature] = fixed_values[feature]


# 로지스틱 회귀의 확률 계산
grid_probability = logistic_model.predict_proba(
    grid_data
)[:, 1].reshape(xx.shape)


# 의사결정트리의 분류 결과 계산
grid_tree_prediction = tree_model.predict(
    grid_data
).reshape(xx.shape)


# ==========================================
# 10. Plotly 산점도
# ==========================================

plot_df = pd.DataFrame({
    x_feature: X_test[x_feature].values,
    y_feature: X_test[y_feature].values,
    "stroke_group": y_test.map({
        0: "뇌졸중 없음",
        1: "뇌졸중 있음"
    }).values
})

COLOR_MAP = {
    "뇌졸중 없음": "#4C9BE8",
    "뇌졸중 있음": "#E76F51"
}


fig = go.Figure()


# 의사결정트리의 분류 영역
fig.add_trace(
    go.Heatmap(
        x=x_grid,
        y=y_grid,
        z=grid_tree_prediction,
        colorscale=[
            [0, "rgba(76,155,232,0.20)"],
            [1, "rgba(231,111,81,0.20)"]
        ],
        zmin=0,
        zmax=1,
        showscale=False,
        hoverinfo="skip",
        name="의사결정트리 분류 영역"
    )
)


# 로지스틱 회귀의 0.5 확률 경계선
prob_min = float(grid_probability.min())
prob_max = float(grid_probability.max())

boundary_exists = (
    prob_min <= 0.5 <= prob_max
)

if boundary_exists:

    fig.add_trace(
        go.Contour(
            x=x_grid,
            y=y_grid,
            z=grid_probability,
            contours=dict(
                start=0.5,
                end=0.5,
                size=0.001,
                coloring="none",
                showlabels=False
            ),
            line=dict(
                color="#222222",
                width=3
            ),
            showscale=False,
            hoverinfo="skip",
            name="로지스틱 회귀 0.5 경계선"
        )
    )


# 실제 테스트 데이터 점
for group_name in [
    "뇌졸중 없음",
    "뇌졸중 있음"
]:

    group_data = plot_df[
        plot_df["stroke_group"] == group_name
    ]

    fig.add_trace(
        go.Scatter(
            x=group_data[x_feature],
            y=group_data[y_feature],
            mode="markers",
            name=group_name,
            marker=dict(
                color=COLOR_MAP[group_name],
                size=6,
                opacity=0.75,
                line=dict(
                    width=0.4,
                    color="white"
                )
            ),
            hovertemplate=(
                f"{FEATURE_NAMES[x_feature]}: %{{x:.2f}}<br>"
                f"{FEATURE_NAMES[y_feature]}: %{{y:.2f}}"
                "<extra>%{fullData.name}</extra>"
            )
        )
    )


fig.update_layout(
    title=(
        "테스트 데이터와 두 분류 모델의 예측 영역"
    ),
    xaxis_title=FEATURE_NAMES[x_feature],
    yaxis_title=FEATURE_NAMES[y_feature],
    template="plotly_white",
    height=650,
    legend_title="실제 뇌졸중 여부",
    hovermode="closest"
)

fig.update_xaxes(
    range=[
        x_min - x_padding,
        x_max + x_padding
    ]
)

fig.update_yaxes(
    range=[
        y_min - y_padding,
        y_max + y_padding
    ]
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# 경계선 존재 여부 안내
if boundary_exists:

    st.success(
        "로지스틱 회귀의 뇌졸중 발생 확률 0.5 경계선이 "
        "현재 그래프 범위 안에 있습니다."
    )

else:

    st.warning(
        "로지스틱 회귀의 뇌졸중 발생 확률 0.5 경계선은 "
        "현재 그래프 범위 안에 나타나지 않습니다. "
        "현재 범위에서 계산한 확률은 "
        f"{prob_min:.3f}~{prob_max:.3f}입니다."
    )


st.caption(
    "※ 옅은 파란색과 주황색 영역은 의사결정트리의 예측 결과이며, "
    "점의 색은 실제 뇌졸중 여부를 의미합니다. "
    "검은 선은 로지스틱 회귀가 뇌졸중 발생 확률 0.5를 기준으로 "
    "분류하는 경계선입니다."
)


st.divider()


# ==========================================
# 11. 의사결정트리 가지 그림
# ==========================================

st.subheader("7️⃣ 의사결정트리의 질문 구조")

st.write(
    "의사결정트리가 학습 데이터에 어떤 질문을 던졌는지 "
    "가지 그림으로 확인합니다. "
    "각 마디에는 해당 조건에 도달한 훈련용 사람 수와 "
    "그중 뇌졸중을 겪은 사람 수 및 비율이 표시됩니다."
)


tree_structure = tree_model.tree_

# 각 훈련 데이터가 거치는 마디 확인
decision_path = tree_model.decision_path(
    X_train
)

dot_lines = [
    "digraph DecisionTree {",
    'graph [rankdir=TB, bgcolor="white", nodesep=0.45, ranksep=0.65];',
    'node [shape=box, style="filled,rounded", fontname="Malgun Gothic", fontsize=11, margin=0.16];',
    'edge [fontname="Malgun Gothic", fontsize=10];'
]

leaf_count = 0
negative_leaf_count = 0
used_features = set()


# 모든 마디 순회
for node_id in range(tree_structure.node_count):

    # 해당 마디에 도달한 훈련 데이터
    reached_mask = (
        decision_path[:, node_id]
        .toarray()
        .ravel()
        .astype(bool)
    )

    node_y = y_train.iloc[
        np.flatnonzero(reached_mask)
    ]

    total_count = len(node_y)

    positive_count = int(
        (node_y == 1).sum()
    )

    positive_ratio = (
        positive_count / total_count * 100
        if total_count > 0
        else 0
    )

    # 리프 노드인지 확인
    is_leaf = (
        tree_structure.children_left[node_id]
        == tree_structure.children_right[node_id]
    )

    if is_leaf:

        leaf_count += 1

        # 해당 리프의 실제 예측 클래스
        predicted_class = int(
            tree_model.classes_[
                np.argmax(
                    tree_structure.value[node_id][0]
                )
            ]
        )

        if predicted_class == 0:

            negative_leaf_count += 1

            fill_color = "#DCEEFF"
            answer_text = "답: 뇌졸중 없음"

        else:

            fill_color = "#FFE2D8"
            answer_text = "답: 뇌졸중 있음"

        node_label = (
            f"마디 {node_id}\\n"
            f"사람 수: {total_count}명\\n"
            f"뇌졸중: {positive_count}명\\n"
            f"발생 비율: {positive_ratio:.1f}%\\n"
            f"{answer_text}"
        )

    else:

        feature_index = tree_structure.feature[node_id]

        feature_name = selected_features[
            feature_index
        ]

        used_features.add(
            feature_name
        )

        threshold = tree_structure.threshold[node_id]

        fill_color = "#F0F0F0"

        node_label = (
            f"마디 {node_id}\\n"
            f"질문: {FEATURE_NAMES[feature_name]} "
            f"≤ {threshold:.2f}?\\n"
            f"사람 수: {total_count}명\\n"
            f"뇌졸중: {positive_count}명\\n"
            f"발생 비율: {positive_ratio:.1f}%"
        )

    # DOT 노드 추가
    dot_lines.append(
        f'{node_id} [label="{node_label}", fillcolor="{fill_color}"];'
    )


# 가지 연결
for node_id in range(tree_structure.node_count):

    left_child = tree_structure.children_left[node_id]
    right_child = tree_structure.children_right[node_id]

    is_leaf = (
        left_child == right_child
    )

    if is_leaf:
        continue

    threshold = tree_structure.threshold[node_id]

    # 왼쪽: 질문에 예 (조건 만족)
    dot_lines.append(
        f'{node_id} -> {left_child} '
        f'[label="예 (≤ {threshold:.2f})"];'
    )

    # 오른쪽: 질문에 아니요 (조건 불만족)
    dot_lines.append(
        f'{node_id} -> {right_child} '
        f'[label="아니요 (> {threshold:.2f})"];'
    )


dot_lines.append("}")

dot_string = "\n".join(dot_lines)


# Streamlit 내장 Graphviz 시각화
st.graphviz_chart(
    dot_string,
    use_container_width=True
)


# ==========================================
# 12. 의사결정트리 구조 분석
# ==========================================

st.markdown("#### 의사결정트리 구조 분석")

used_feature_names = [
    FEATURE_NAMES[feature]
    for feature in selected_features
    if feature in used_features
]

if used_feature_names:

    used_text = ", ".join(
        used_feature_names
    )

else:

    used_text = "없음"


st.write(
    f"- **답을 내리는 마디(리프 노드):** "
    f"{leaf_count}개"
)

st.write(
    f"- **뇌졸중 없음으로 답하는 마디:** "
    f"{negative_leaf_count}개"
)

st.write(
    f"- **실제로 질문에 사용한 속성:** "
    f"{used_text}"
)


# ==========================================
# 13. 마무리
# ==========================================

st.divider()

st.subheader("8️⃣ 분류 모델 비교 정리")

st.write(
    f"로지스틱 회귀의 테스트 정확도는 "
    f"**{logistic_test_accuracy:.2%}**이고, "
    f"의사결정트리의 테스트 정확도는 "
    f"**{tree_test_accuracy:.2%}**입니다."
)

st.write(
    f"입력을 사용하지 않는 기준 모델의 테스트 정확도는 "
    f"**{baseline_test_accuracy:.2%}**입니다."
)

if logistic_test_accuracy > tree_test_accuracy:

    st.write(
        "이번 실험에서는 로지스틱 회귀의 테스트 정확도가 "
        "의사결정트리보다 높았습니다."
    )

elif logistic_test_accuracy < tree_test_accuracy:

    st.write(
        "이번 실험에서는 의사결정트리의 테스트 정확도가 "
        "로지스틱 회귀보다 높았습니다."
    )

else:

    st.write(
        "이번 실험에서는 두 모델의 테스트 정확도가 같습니다."
    )


st.caption(
    "🧠 뇌졸중 예측 실습실 | 분류 모델 페이지"
)

st.caption(
    "※ 이 실습은 머신러닝의 분류 원리를 이해하기 위한 교육용 "
    "분석입니다. 실제 뇌졸중 진단이나 의료적 판단에 사용할 수 없습니다."
)
