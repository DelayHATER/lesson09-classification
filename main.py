
import json
from pathlib import Path

import pandas as pd
import streamlit as st


# ==========================================
# 1. 기본 설정
# ==========================================

st.set_page_config(
    page_title="뇌졸중 예측 실습실",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("🧠 뇌졸중 예측 실습실")
st.markdown(
    "뇌졸중 데이터를 살펴보고, 데이터의 의미를 이해하며 "
    "머신러닝 예측 실습을 준비하는 공간입니다."
)

st.divider()


# ==========================================
# 2. 데이터 및 기록 파일 설정
# ==========================================

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "greatsong/modudata/main/data/stroke.csv"
)

RECORD_DIR = Path("내_기록")
RECORD_FILE = RECORD_DIR / "stroke_기록.json"

RECORD_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 3. 기록 불러오기
# ==========================================

def load_record():
    """기존에 저장한 우리말 뜻과 출처를 불러옵니다."""

    if RECORD_FILE.exists():
        try:
            with open(RECORD_FILE, "r", encoding="utf-8") as file:
                return json.load(file)
        except (json.JSONDecodeError, OSError):
            st.warning("저장된 기록을 읽지 못했습니다. 새 기록을 사용합니다.")

    return {
        "meanings": {},
        "source": ""
    }


def save_record(meanings, source):
    """우리말 뜻과 데이터 출처를 JSON 파일에 저장합니다."""

    record = {
        "meanings": meanings,
        "source": source
    }

    with open(RECORD_FILE, "w", encoding="utf-8") as file:
        json.dump(
            record,
            file,
            ensure_ascii=False,
            indent=4
        )


saved_record = load_record()


# ==========================================
# 4. CSV 데이터 불러오기
# ==========================================

@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


try:
    df = load_data()

except Exception as e:
    st.error("데이터를 불러오지 못했습니다.")
    st.error(f"오류 내용: {e}")
    st.info("인터넷 연결과 데이터 주소를 확인해 주세요.")
    st.stop()


# ==========================================
# 5. 데이터 기본 정보 계산
# ==========================================

total_people = len(df)
total_columns = len(df.columns)

stroke_count = int(
    (df["stroke"] == 1).sum()
)

stroke_ratio = (
    stroke_count / total_people * 100
    if total_people > 0
    else 0
)


# ==========================================
# 6. 숫자 카드 4개
# ==========================================

st.subheader("📊 데이터 한눈에 보기")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="👥 전체 사람 수",
        value=f"{total_people:,}명"
    )

with col2:
    st.metric(
        label="📋 전체 열 개수",
        value=f"{total_columns}개"
    )

with col3:
    st.metric(
        label="🧠 뇌졸중 발생 인원",
        value=f"{stroke_count:,}명"
    )

with col4:
    st.metric(
        label="📈 뇌졸중 발생 비율",
        value=f"{stroke_ratio:.2f}%"
    )


st.divider()


# ==========================================
# 7. 데이터 열 정보 표
# ==========================================

st.subheader("📖 데이터 열 살펴보기")

st.write(
    "각 열의 이름과 값의 종류, 빈 값 개수를 확인해 보세요. "
    "**우리말 뜻** 칸은 교재를 참고하여 직접 작성할 수 있습니다."
)

meanings = saved_record.get("meanings", {})

column_info = []

for column in df.columns:

    if pd.api.types.is_numeric_dtype(df[column]):
        value_type = "숫자형"
    else:
        value_type = "문자형"

    missing_count = int(df[column].isna().sum())

    column_info.append({
        "열 이름": column,
        "우리말 뜻": meanings.get(column, ""),
        "값의 종류": value_type,
        "빈 값 개수": missing_count
    })

info_df = pd.DataFrame(column_info)

# 우리말 뜻만 수정할 수 있도록 설정
edited_info = st.data_editor(
    info_df,
    use_container_width=True,
    hide_index=True,
    num_rows="fixed",
    disabled=[
        "열 이름",
        "값의 종류",
        "빈 값 개수"
    ],
    column_config={
        "열 이름": st.column_config.TextColumn(
            "열 이름",
            width="medium"
        ),
        "우리말 뜻": st.column_config.TextColumn(
            "우리말 뜻 (직접 작성)",
            width="large",
            help="교재를 참고하여 각 열의 우리말 뜻을 입력하세요."
        ),
        "값의 종류": st.column_config.TextColumn(
            "값의 종류",
            width="small"
        ),
        "빈 값 개수": st.column_config.NumberColumn(
            "빈 값 개수",
            width="small"
        )
    },
    key="column_editor"
)


st.divider()


# ==========================================
# 8. 데이터 처음 5행 표시
# ==========================================

st.subheader("🔍 데이터 미리보기")

st.write(
    "실제 데이터의 처음 다섯 줄입니다. "
    "원본 데이터의 값을 그대로 확인할 수 있습니다."
)

st.dataframe(
    df.head(5),
    use_container_width=True,
    hide_index=True
)


st.divider()


# ==========================================
# 9. 데이터 출처 입력
# ==========================================

st.subheader("📚 데이터 출처")

st.write(
    "교재에 안내된 데이터 출처를 아래에 직접 작성하세요."
)

source_text = st.text_area(
    "데이터 출처",
    value=saved_record.get("source", ""),
    placeholder="교재를 참고하여 데이터 출처를 입력하세요.",
    height=120,
    key="source_editor"
)


# ==========================================
# 10. 기록 저장
# ==========================================

st.divider()

save_col1, save_col2 = st.columns([1, 3])

with save_col1:
    save_clicked = st.button(
        "💾 기록 저장하기",
        type="primary",
        use_container_width=True
    )

with save_col2:
    st.caption(
        "작성한 우리말 뜻과 데이터 출처를 "
        "'내_기록/stroke_기록.json' 파일에 저장합니다."
    )


if save_clicked:

    meaning_dict = {}

    for _, row in edited_info.iterrows():
        meaning_dict[row["열 이름"]] = row["우리말 뜻"]

    try:
        save_record(
            meanings=meaning_dict,
            source=source_text
        )

        st.success(
            "저장 완료! 다음에 앱을 다시 실행해도 "
            "작성한 기록이 유지됩니다."
        )

    except OSError as e:
        st.error(f"파일 저장 중 오류가 발생했습니다: {e}")


# ==========================================
# 11. 하단 안내
# ==========================================

st.divider()

st.caption("🧠 뇌졸중 예측 실습실 | 데이터 탐색 단계")

st.caption(
    "※ 이 데이터는 머신러닝 학습을 위한 교육용 자료이며, "
    "실제 의료 진단이나 뇌졸중 위험 판단에 사용해서는 안 됩니다."
)
