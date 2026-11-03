import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# --------------------------------------------------
# 기본 설정
# --------------------------------------------------

st.set_page_config(
    page_title="분류 모델",
    page_icon="🤖",
    layout="wide"
)

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"

FEATURE_NAMES = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병"
}

FEATURES = list(FEATURE_NAMES.keys())

RANDOM_STATE = 42


# --------------------------------------------------
# 데이터 불러오기
# --------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL)


try:
    df = load_data()
except Exception as e:
    st.error(f"데이터를 불러오지 못했습니다: {e}")
    st.stop()


st.title("🤖 분류 모델")
st.write("인공지능 분류 모델을 활용하여 뇌졸중 발생 여부를 예측합니다.")

st.divider()


# --------------------------------------------------
# 특성 선택
# --------------------------------------------------

st.subheader("1. 예측에 사용할 특성 선택")

selected_features = st.multiselect(
    "분석에 사용할 항목을 선택하세요.",
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
    st.warning("분석을 진행하려면 최소 2개 이상의 특성을 선택해야 합니다.")
    st.stop()


# --------------------------------------------------
# 데이터 전처리
# --------------------------------------------------

df = df.copy()

if "id" in df.columns:
    df = df.sort_values("id").reset_index(drop=True)

df["stroke"] = pd.to_numeric(df["stroke"], errors="coerce")
df = df.dropna(subset=["stroke"])

X = df[selected_features].copy()

for col in selected_features:
    X[col] = pd.to_numeric(X[col], errors="coerce")

y = df["stroke"].astype(int)

if not set(y.unique()).issubset({0, 1}) or len(y.unique()) != 2:
    st.error("뇌졸중 여부(stroke)는 0과 1로 구성되어야 합니다.")
    st.stop()

if not np.isfinite(X.to_numpy(dtype=float)).all() and not X.isna().any().any():
    st.error("특성 데이터에 처리할 수 없는 값이 있습니다.")
    st.stop()


# --------------------------------------------------
# 학습 데이터와 테스트 데이터 분리
# --------------------------------------------------

# ID 순서대로 10개씩 묶고 각 묶음의 첫 3개를 테스트 데이터로 사용

position_in_group = np.arange(len(df)) % 10

test_mask = position_in_group < 3

X_train = X.loc[~test_mask].copy()
X_test = X.loc[test_mask].copy()

y_train = y.loc[~test_mask].copy()
y_test = y.loc[test_mask].copy()


# --------------------------------------------------
# BMI 결측치 처리
# --------------------------------------------------

if "bmi" in selected_features:

    bmi_median = X_train["bmi"].median()

    if pd.isna(bmi_median):
        st.error("학습 데이터에서 BMI 중앙값을 계산할 수 없습니다.")
        st.stop()

    X_train["bmi"] = X_train["bmi"].fillna(bmi_median)
    X_test["bmi"] = X_test["bmi"].fillna(bmi_median)


if X_train.isna().any().any() or X_test.isna().any().any():
    st.error("선택한 특성에 결측치가 남아 있습니다.")
    st.stop()


if not np.isfinite(X_train.to_numpy(dtype=float)).all():
    st.error("학습 데이터에 무한대 또는 유효하지 않은 값이 있습니다.")
    st.stop()

if not np.isfinite(X_test.to_numpy(dtype=float)).all():
    st.error("테스트 데이터에 무한대 또는 유효하지 않은 값이 있습니다.")
    st.stop()


# --------------------------------------------------
# 데이터 분할 결과
# --------------------------------------------------

st.subheader("2. 데이터 분할 결과")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("전체 데이터", f"{len(df):,}개")

with col2:
    st.metric("학습 데이터", f"{len(X_train):,}개")

with col3:
    st.metric("테스트 데이터", f"{len(X_test):,}개")


st.caption(
    "10개씩 묶은 데이터에서 앞의 3개는 테스트, "
    "나머지 7개는 학습에 사용합니다."
)

st.write("선택한 특성:", ", ".join(
    FEATURE_NAMES[x] for x in selected_features
))

st.divider()


# --------------------------------------------------
# 3. 기본 모델 학습
# --------------------------------------------------

st.subheader("3. 기본 분류 모델")

# 로지스틱 회귀

logistic_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE
    ))
])

logistic_model.fit(X_train, y_train)


# 의사결정트리

tree_model = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=5,
    random_state=RANDOM_STATE
)

tree_model.fit(X_train, y_train)


# 기준 모델: 학습 데이터의 다수 클래스를 항상 예측

majority_class = y_train.mode()[0]


# --------------------------------------------------
# 4. 성능 개선 모델
# --------------------------------------------------

st.subheader("4. 성능 개선 모델")

st.write(
    "뇌졸중 데이터는 발생 사례와 비발생 사례의 수가 크게 다릅니다. "
    "이를 보완하기 위해 클래스 가중치를 적용하고 모델의 설정을 최적화합니다."
)


# 5겹 교차검증
# 학습 데이터만 이용하여 최적의 C 선택

class_counts = y_train.value_counts()

if class_counts.min() < 5:
    st.error("교차검증을 수행하기에 학습 데이터의 특정 클래스가 너무 적습니다.")
    st.stop()

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)


# 개선된 로지스틱 회귀

improved_logistic = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        random_state=RANDOM_STATE
    ))
])

logistic_search = GridSearchCV(
    estimator=improved_logistic,
    param_grid={
        "model__C": [0.01, 0.1, 1, 10, 100]
    },
    scoring="f1",
    cv=cv,
    n_jobs=-1
)

logistic_search.fit(X_train, y_train)

best_logistic = logistic_search.best_estimator_


# 개선된 의사결정트리
# 기존 깊이 3, 최소 리프 5개 조건 유지

improved_tree = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=RANDOM_STATE
)

improved_tree.fit(X_train, y_train)


st.success("성능 개선 모델의 학습이 완료되었습니다.")

st.write(
    f"로지스틱 회귀 최적 C 값: **{logistic_search.best_params_['model__C']}**"
)

st.write(
    f"5겹 교차검증 최고 F1-score: "
    f"**{logistic_search.best_score_:.4f}**"
)

st.divider()


# --------------------------------------------------
# 5. 모델 평가 함수
# --------------------------------------------------

def evaluate_model(model, X_data, y_data):

    prediction = model.predict(X_data)

    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(X_data)[:, 1]
    else:
        probability = prediction.astype(float)

    accuracy = accuracy_score(y_data, prediction)

    precision = precision_score(
        y_data, prediction, zero_division=0
    )

    recall = recall_score(
        y_data, prediction, zero_division=0
    )

    f1 = f1_score(
        y_data, prediction, zero_division=0
    )

    try:
        auc = roc_auc_score(y_data, probability)
    except ValueError:
        auc = np.nan

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "prediction": prediction,
        "probability": probability
    }


# 기준 모델 평가용

class BaselineModel:

    def predict(self, X_data):
        return np.full(len(X_data), majority_class)

    def predict_proba(self, X_data):
        return np.tile(
            [1 - majority_class, majority_class],
            (len(X_data), 1)
        )


baseline_model = BaselineModel()


models = {
    "기준 모델": baseline_model,
    "로지스틱 회귀(확률로 답하는 모델)": logistic_model,
    "의사결정트리(질문으로 답하는 모델)": tree_model,
    "개선된 로지스틱 회귀": best_logistic,
    "개선된 의사결정트리": improved_tree
}


results = {}

for name, model in models.items():

    train_result = evaluate_model(
        model, X_train, y_train
    )

    test_result = evaluate_model(
        model, X_test, y_test
    )

    results[name] = {
        "train": train_result,
        "test": test_result
    }


# --------------------------------------------------
# 6. 모델별 정확도 비교
# --------------------------------------------------

st.subheader("5. 모델별 정확도 비교")

for name, result in results.items():

    train_acc = result["train"]["accuracy"]
    test_acc = result["test"]["accuracy"]

    st.markdown(f"**{name}**")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "테스트 정확도",
            f"{test_acc:.2%}"
        )

    with col2:
        st.metric(
            "학습 정확도",
            f"{train_acc:.2%}"
        )

    with col3:
        st.metric(
            "학습-테스트 차이",
            f"{abs(train_acc - test_acc):.2%}"
        )


st.divider()


# --------------------------------------------------
# 7. 상세 채점 결과
# --------------------------------------------------

st.subheader("6. 상세 성능 평가")

st.write(
    "테스트 데이터를 기준으로 모델의 예측 능력을 평가합니다."
)

metric_rows = []

for name, result in results.items():

    test = result["test"]

    metric_rows.append({
        "모델": name,
        "정확도": test["accuracy"],
        "정밀도": test["precision"],
        "재현율": test["recall"],
        "F1-score": test["f1"],
        "ROC-AUC": test["auc"]
    })


metric_df = pd.DataFrame(metric_rows)

display_df = metric_df.copy()

for col in [
    "정확도",
    "정밀도",
    "재현율",
    "F1-score",
    "ROC-AUC"
]:
    display_df[col] = display_df[col].apply(
        lambda x: "-" if pd.isna(x) else f"{x:.4f}"
    )

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)


st.caption(
    "정밀도는 뇌졸중이라고 예측한 결과의 신뢰도를, "
    "재현율은 실제 뇌졸중 사례를 찾아낸 정도를 나타냅니다. "
    "ROC-AUC는 두 집단을 구분하는 능력을 평가합니다."
)


# --------------------------------------------------
# 8. 성능 비교 그래프
# --------------------------------------------------

st.subheader("7. 성능 개선 전후 비교")

comparison_df = metric_df[
    metric_df["모델"].isin([
        "로지스틱 회귀(확률로 답하는 모델)",
        "의사결정트리(질문으로 답하는 모델)",
        "개선된 로지스틱 회귀",
        "개선된 의사결정트리"
    ])
]

comparison_long = comparison_df.melt(
    id_vars="모델",
    value_vars=[
        "정확도",
        "정밀도",
        "재현율",
        "F1-score",
        "ROC-AUC"
    ],
    var_name="평가 지표",
    value_name="점수"
)

fig_comparison = px.bar(
    comparison_long,
    x="평가 지표",
    y="점수",
    color="모델",
    barmode="group",
    title="기본 모델과 개선 모델의 성능 비교",
    range_y=[0, 1]
)

st.plotly_chart(
    fig_comparison,
    use_container_width=True
)


st.divider()


# --------------------------------------------------
# 9. 혼동 행렬
# --------------------------------------------------

st.subheader("8. 혼동 행렬")

confusion_model_name = st.selectbox(
    "혼동 행렬을 확인할 모델을 선택하세요.",
    list(models.keys())
)

selected_model = models[confusion_model_name]

selected_prediction = results[
    confusion_model_name
]["test"]["prediction"]

cm = confusion_matrix(
    y_test,
    selected_prediction,
    labels=[0, 1]
)

fig_cm = px.imshow(
    cm,
    text_auto=True,
    x=["뇌졸중 없음 예측", "뇌졸중 있음 예측"],
    y=["실제 뇌졸중 없음", "실제 뇌졸중 있음"],
    color_continuous_scale="Blues",
    title=f"{confusion_model_name} 혼동 행렬"
)

fig_cm.update_layout(
    xaxis_title="예측 결과",
    yaxis_title="실제 결과"
)

st.plotly_chart(
    fig_cm,
    use_container_width=True
)

st.caption(
    "왼쪽 위와 오른쪽 아래는 올바르게 예측한 수이며, "
    "나머지 두 칸은 잘못 예측한 수입니다."
)


st.divider()


# --------------------------------------------------
# 10. 두 특성을 이용한 예측 영역
# --------------------------------------------------

st.subheader("9. 분류 결과 시각화")

st.write(
    "선택한 특성 중 처음 두 가지를 가로축과 세로축으로 사용합니다. "
    "나머지 특성은 테스트 데이터의 중앙값으로 고정합니다."
)

x_feature = selected_features[0]
y_feature = selected_features[1]

other_features = [
    f for f in selected_features
    if f not in [x_feature, y_feature]
]

fixed_values = {}

for feature in other_features:
    fixed_values[feature] = float(
        X_test[feature].median()
    )

if other_features:
    st.write(
        "고정한 특성: " +
        ", ".join(
            f"{FEATURE_NAMES[f]} = {fixed_values[f]:.2f}"
            for f in other_features
        )
    )


# 시각화 범위

x_min = float(X_test[x_feature].min())
x_max = float(X_test[x_feature].max())

y_min = float(X_test[y_feature].min())
y_max = float(X_test[y_feature].max())

x_padding = (x_max - x_min) * 0.05 or 1
y_padding = (y_max - y_min) * 0.05 or 1

x_values = np.linspace(
    x_min - x_padding,
    x_max + x_padding,
    100
)

y_values = np.linspace(
    y_min - y_padding,
    y_max + y_padding,
    100
)

xx, yy = np.meshgrid(x_values, y_values)


grid_data = pd.DataFrame({
    x_feature: xx.ravel(),
    y_feature: yy.ravel()
})

for feature in other_features:
    grid_data[feature] = fixed_values[feature]

grid_data = grid_data[selected_features]


# 기본 로지스틱 회귀의 확률

logistic_probability = logistic_model.predict_proba(
    grid_data
)[:, 1].reshape(xx.shape)


# 개선된 의사결정트리의 예측 영역

tree_prediction = improved_tree.predict(
    grid_data
).reshape(xx.shape)


fig_region = go.Figure()


# 트리 예측 영역

fig_region.add_trace(
    go.Heatmap(
        x=x_values,
        y=y_values,
        z=tree_prediction,
        colorscale=[
            [0, "rgba(70,130,180,0.15)"],
            [1, "rgba(255,100,100,0.15)"]
        ],
        showscale=False,
        name="의사결정트리 영역"
    )
)


# 로지스틱 회귀 0.5 경계선

fig_region.add_trace(
    go.Contour(
        x=x_values,
        y=y_values,
        z=logistic_probability,
        contours=dict(
            start=0.5,
            end=0.5,
            size=0.01,
            coloring="none",
            showlabels=False
        ),
        line=dict(
            color="black",
            width=3
        ),
        showscale=False,
        name="로지스틱 회귀 경계선"
    )
)


# 테스트 데이터

test_plot = X_test[[x_feature, y_feature]].copy()

test_plot["실제 결과"] = y_test.map({
    0: "뇌졸중 없음",
    1: "뇌졸중 있음"
})

fig_region.add_trace(
    go.Scatter(
        x=test_plot[x_feature],
        y=test_plot[y_feature],
        mode="markers",
        marker=dict(
            size=7,
            color=test_plot["실제 결과"].map({
                "뇌졸중 없음": "royalblue",
                "뇌졸중 있음": "red"
            }),
            line=dict(width=0.5, color="white")
        ),
        text=test_plot["실제 결과"],
        name="테스트 데이터"
    )
)


fig_region.update_layout(
    title="로지스틱 회귀 경계선과 의사결정트리 예측 영역",
    xaxis_title=FEATURE_NAMES[x_feature],
    yaxis_title=FEATURE_NAMES[y_feature],
    height=600
)

st.plotly_chart(
    fig_region,
    use_container_width=True
)


# 경계선이 시각화 범위 안에 있는지 확인

if logistic_probability.min() > 0.5 or logistic_probability.max() < 0.5:
    st.info(
        "현재 시각화 범위에서는 로지스틱 회귀의 "
        "0.5 확률 경계선이 나타나지 않습니다."
    )


st.caption(
    "검은 선은 로지스틱 회귀의 예측 확률이 0.5인 경계이며, "
    "배경 영역은 개선된 의사결정트리의 예측 결과입니다."
)


st.divider()


# --------------------------------------------------
# 11. 의사결정트리 구조
# --------------------------------------------------

st.subheader("10. 의사결정트리 구조")

st.write(
    "개선된 의사결정트리가 어떤 질문을 통해 "
    "뇌졸중 여부를 판단하는지 확인합니다."
)

tree_structure = improved_tree.tree_

dot_lines = [
    "digraph Tree {",
    'node [shape=box, style="filled,rounded", '
    'fontname="Malgun Gothic", fontsize=10];',
    'edge [fontname="Malgun Gothic", fontsize=9];'
]


node_sample_counts = []
leaf_no_stroke = 0
leaf_stroke = 0
used_features = set()


# 각 노드의 실제 학습 데이터 통계

decision_path = improved_tree.decision_path(X_train)

for node_id in range(tree_structure.node_count):

    mask = decision_path[:, node_id].toarray().ravel().astype(bool)

    node_y = y_train.iloc[np.where(mask)[0]]

    sample_count = len(node_y)
    stroke_count = int(node_y.sum())

    stroke_ratio = (
        stroke_count / sample_count
        if sample_count > 0 else 0
    )

    node_sample_counts.append(sample_count)

    if tree_structure.children_left[node_id] == _tree.TREE_LEAF:

        predicted_class = improved_tree.classes_[
            np.argmax(tree_structure.value[node_id][0])
        ]

        if predicted_class == 0:
            leaf_no_stroke += 1
            fill_color = "#DCEBFA"
            prediction_label = "뇌졸중 없음"
        else:
            leaf_stroke += 1
            fill_color = "#FADDDD"
            prediction_label = "뇌졸중 있음"

        label = (
            f"예측: {prediction_label}\\n"
            f"학습 데이터: {sample_count}개\\n"
            f"실제 뇌졸중: {stroke_count}개\\n"
            f"실제 비율: {stroke_ratio:.1%}"
        )

        dot_lines.append(
            f'{node_id} [label="{label}", fillcolor="{fill_color}"];'
        )

    else:

        feature_index = tree_structure.feature[node_id]

        feature_name = selected_features[feature_index]

        used_features.add(feature_name)

        threshold = tree_structure.threshold[node_id]

        label = (
            f"{FEATURE_NAMES[feature_name]} <= {threshold:.2f}?\\n"
            f"학습 데이터: {sample_count}개\\n"
            f"실제 뇌졸중: {stroke_count}개\\n"
            f"실제 비율: {stroke_ratio:.1%}"
        )

        dot_lines.append(
            f'{node_id} [label="{label}", fillcolor="#FFF2CC"];'
        )

        left = tree_structure.children_left[node_id]
        right = tree_structure.children_right[node_id]

        dot_lines.append(
            f'{node_id} -> {left} [label="예"];'
        )

        dot_lines.append(
            f'{node_id} -> {right} [label="아니오"];'
        )


dot_lines.append("}")

dot_string = "\n".join(dot_lines)

st.graphviz_chart(
    dot_string,
    use_container_width=True
)


# --------------------------------------------------
# 12. 트리 분석
# --------------------------------------------------

leaf_count = leaf_no_stroke + leaf_stroke

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("전체 리프 노드", f"{leaf_count}개")

with col2:
    st.metric("뇌졸중 없음 예측 리프", f"{leaf_no_stroke}개")

with col3:
    st.metric("뇌졸중 있음 예측 리프", f"{leaf_stroke}개")


if used_features:
    st.write(
        "**의사결정트리가 실제 사용한 특성:** "
        + ", ".join(
            FEATURE_NAMES[f] for f in sorted(used_features)
        )
    )
else:
    st.write("의사결정트리에서 분기 특성이 사용되지 않았습니다.")


st.divider()


# --------------------------------------------------
# 13. 최종 분석
# --------------------------------------------------

st.subheader("11. 최종 분석")

best_model_name = metric_df.loc[
    metric_df["F1-score"].idxmax(),
    "모델"
]

best_f1 = metric_df["F1-score"].max()

best_recall_model = metric_df.loc[
    metric_df["재현율"].idxmax(),
    "모델"
]

best_recall = metric_df["재현율"].max()


st.write(
    f"- 테스트 데이터 기준 F1-score가 가장 높은 모델은 "
    f"**{best_model_name}**입니다. ({best_f1:.4f})"
)

st.write(
    f"- 재현율이 가장 높은 모델은 "
    f"**{best_recall_model}**입니다. ({best_recall:.4f})"
)

st.write(
    "- 성능 개선 모델은 클래스 가중치와 교차검증을 활용하여 "
    "데이터 불균형과 모델 설정 문제를 보완했습니다."
)

st.info(
    "이 결과는 공개된 데이터에 대한 학습용 분석입니다. "
    "실제 의료 진단이나 뇌졸중 위험 판단에 직접 사용할 수 없습니다."
)
