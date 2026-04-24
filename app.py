from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from app_data import (
    COMPETENCY_AXES,
    COUNSELOR_MARKERS,
    RELATIONAL_EDGES,
    RELATIONAL_SCHEMA,
    ROLE_CATALOG,
    ROLE_FAMILY_ORDER,
    SUMMARY_COLUMN_DICT,
    WHISPER_FILE_TYPES,
    WHISPER_LANGUAGE_MAP,
    WHISPER_LANGUAGE_OPTIONS,
    WHISPER_MODEL_OPTIONS,
)


DATA_DIR = Path(__file__).parent / "stt_samples"


@st.cache_data
def load_interviews(data_dir: Path) -> list[dict]:
    interviews = []
    for path in sorted(data_dir.glob("INT-*.json")):
        with path.open("r", encoding="utf-8") as f:
            interviews.append(json.load(f))
    return interviews


def build_summary_rows(interviews: list[dict]) -> list[dict]:
    rows = []
    for interview in interviews:
        meta = interview["metadata"]
        client = meta["client"]
        utterances = interview["utterances"]
        rows.append(
            {
                "interview_id": interview["interview_id"],
                "session_date": meta["session_date"],
                "client_name": client["name"],
                "major": client["major"],
                "grade": client["grade"],
                "tags": ", ".join(client["persona_tags"]),
                "duration": meta["duration"],
                "utterance_count": len(utterances),
            }
        )
    return rows


SPEAKER_CODE_TO_LABEL = {"C": "상담사", "A": "학생"}


def build_utterance_rows(interviews: list[dict]) -> list[dict]:
    rows = []
    for interview in interviews:
        client = interview["metadata"]["client"]
        for utterance in interview["utterances"]:
            rows.append(
                {
                    "interview_id": interview["interview_id"],
                    "client_name": client["name"],
                    "major": client["major"],
                    "speaker": SPEAKER_CODE_TO_LABEL.get(utterance["speaker"], "미분류"),
                    "speaker_code": utterance["speaker"],
                    "text": utterance["text"],
                    "start": utterance["start"],
                    "end": utterance["end"],
                    "confidence": utterance["confidence"],
                }
            )
    return rows


def build_framework_rows() -> list[dict]:
    rows = []
    for role, info in ROLE_CATALOG.items():
        rows.append(
            {
                "role": role,
                "family": info["family"],
                "problem_focus": info["problem_focus"],
                "work_mode": info["work_mode"],
                "fit_majors": ", ".join(info["fit_majors"]),
                "evidence_keywords": ", ".join(info["keywords"]),
            }
        )
    return rows


def analyze_interview(interview: dict) -> list[dict]:
    client = interview["metadata"]["client"]
    major = client["major"]
    tags = client["persona_tags"]
    utterances = interview["utterances"]
    transcript = " ".join(utterance["text"] for utterance in utterances)
    analysis_rows = []

    for role, info in ROLE_CATALOG.items():
        score = 0
        evidence = []

        if role in tags:
            score += 60
            evidence.append("persona_tags에 직접 포함")

        if major in info["fit_majors"]:
            score += 15
            evidence.append(f"전공 적합: {major}")

        matched_keywords = [keyword for keyword in info["keywords"] if keyword in transcript]
        if matched_keywords:
            score += min(25, len(matched_keywords) * 5)
            evidence.append(f"본문 키워드: {', '.join(matched_keywords[:5])}")

        if score > 0:
            analysis_rows.append(
                {
                    "role": role,
                    "family": info["family"],
                    "score": score,
                    "problem_focus": info["problem_focus"],
                    "work_mode": info["work_mode"],
                    "evidence": " / ".join(evidence),
                }
            )

    analysis_rows.sort(key=lambda row: (-row["score"], row["role"]))
    return analysis_rows


def extract_competency_profile(interview: dict) -> list[dict]:
    student_text = " ".join(
        utterance["text"] for utterance in interview["utterances"] if utterance["speaker"] == "A"
    )
    rows = []

    for axis, info in COMPETENCY_AXES.items():
        matched_keywords = [keyword for keyword in info["keywords"] if keyword in student_text]
        score = min(100, len(matched_keywords) * 12)
        if score > 0:
            rows.append(
                {
                    "axis": axis,
                    "score": score,
                    "description": info["description"],
                    "evidence_keywords": ", ".join(matched_keywords[:6]),
                }
            )

    rows.sort(key=lambda row: (-row["score"], row["axis"]))
    return rows


def generate_recommendation_summary(interview: dict, top_roles: list[dict], competency_rows: list[dict]) -> str:
    client = interview["metadata"]["client"]
    name = client["name"]
    major = client["major"]
    role_names = [row["role"] for row in top_roles[:3]]
    axis_names = [row["axis"] for row in competency_rows[:2]]

    if not role_names:
        return f"{name} 학생은 현재 대화만으로는 특정 직무를 강하게 추천할 근거가 아직 부족합니다."

    role_text = ", ".join(role_names)
    axis_text = ", ".join(axis_names) if axis_names else "기본 역량"

    return (
        f"{name} 학생은 {major} 전공 배경 위에서 {axis_text} 성향이 두드러지며, "
        f"대화 속 경험과 키워드를 보면 {role_text} 방향과의 접점이 큽니다. "
        "즉, 단순 관심 수준이 아니라 실제로 문제를 푸는 방식이 해당 직무와 맞물린다고 볼 수 있습니다."
    )


def analyze_counselor_quality(interview: dict) -> dict:
    counselor_utterances = [
        utterance["text"] for utterance in interview["utterances"] if utterance["speaker"] == "C"
    ]

    open_question_markers = COUNSELOR_MARKERS["탐색 질문"]
    validation_markers = COUNSELOR_MARKERS["공감/지지"]
    summary_markers = COUNSELOR_MARKERS["구조화/정리"]
    action_markers = COUNSELOR_MARKERS["실행 제안"]

    open_question_count = 0
    validation_count = 0
    summary_count = 0
    action_count = 0

    for text in counselor_utterances:
        if "?" in text and any(marker in text for marker in open_question_markers):
            open_question_count += 1
        if any(marker in text for marker in validation_markers):
            validation_count += 1
        if any(marker in text for marker in summary_markers):
            summary_count += 1
        if any(marker in text for marker in action_markers):
            action_count += 1

    total = max(1, len(counselor_utterances))
    exploration_score = min(100, int((open_question_count / total) * 220))
    empathy_score = min(100, validation_count * 20 + summary_count * 8)
    structuring_score = min(100, summary_count * 25 + open_question_count * 5)
    actionability_score = min(100, action_count * 25)
    balance_score = max(
        0,
        100 - abs((open_question_count + summary_count) - action_count) * 12,
    )

    return {
        "exploration_score": exploration_score,
        "empathy_score": empathy_score,
        "structuring_score": structuring_score,
        "actionability_score": actionability_score,
        "balance_score": balance_score,
        "open_question_count": open_question_count,
        "validation_count": validation_count,
        "summary_count": summary_count,
        "action_count": action_count,
        "total_counselor_utterances": total,
    }


def find_counselor_examples(interview: dict, category: str, limit: int = 3) -> list[str]:
    markers = COUNSELOR_MARKERS[category]
    examples = []
    for utterance in interview["utterances"]:
        if utterance["speaker"] != "C":
            continue
        if any(marker in utterance["text"] for marker in markers):
            examples.append(utterance["text"])
    return examples[:limit]


def find_supporting_utterances(interview: dict, role: str, limit: int = 3) -> list[dict]:
    keywords = ROLE_CATALOG[role]["keywords"]
    matches = []
    for utterance in interview["utterances"]:
        if utterance["speaker"] != "A":
            continue
        match_count = sum(1 for keyword in keywords if keyword in utterance["text"])
        if match_count > 0:
            matches.append(
                {
                    "speaker": "학생",
                    "text": utterance["text"],
                    "weight": match_count,
                }
            )
    matches.sort(key=lambda row: (-row["weight"], row["text"]))
    return matches[:limit]


SPEAKER_LABELS = {"C": ("상담사", "speaker-c"), "A": ("학생", "speaker-a")}


def infer_dtype_label(series: pd.Series) -> str:
    dtype = str(series.dtype)
    mapping = {
        "object": "문자열 (object)",
        "int64": "정수 (int64)",
        "float64": "실수 (float64)",
        "bool": "불리언 (bool)",
        "datetime64[ns]": "날짜/시간 (datetime64)",
    }
    return mapping.get(dtype, dtype)


def render_summary_column_dictionary(summary_df: pd.DataFrame) -> None:
    st.markdown("#### 1단계 — 지금 표의 컬럼이 무엇인지 이해하기")
    st.caption(
        "위 표의 각 열(column)이 어떤 필드이고, 어떤 타입이며, 원본 JSON의 어디에서 왔는지 정리한 '컬럼 사전'입니다."
    )

    actual_dtypes = {col: infer_dtype_label(summary_df[col]) for col in summary_df.columns}
    rows = []
    for entry in SUMMARY_COLUMN_DICT:
        rows.append(
            {
                "컬럼 (영문)": entry["컬럼"],
                "한글명": entry["한글명"],
                "의미": entry["의미"],
                "정의된 타입": entry["데이터 타입"],
                "실제 pandas 타입": actual_dtypes.get(entry["컬럼"], "-"),
                "예시 값": entry["예시"],
                "원본 JSON 경로": entry["원본 경로"],
            }
        )
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    st.caption(
        "※ '정의된 타입'은 우리가 의도한 타입이고, '실제 pandas 타입'은 현재 DataFrame이 추론한 타입입니다. "
        "두 값이 다르면 타입 변환(캐스팅)이 필요할 수 있다는 신호입니다."
    )


def render_relational_schema() -> None:
    st.markdown("#### 2단계 — 원본 JSON을 관계형 테이블로 나누면?")
    st.markdown(
        """
        지금 보는 표는 인터뷰 한 건을 한 줄로 요약한 모습입니다. 하지만 원본 데이터에는
        학생 프로필, 상담사, 발화 내용 같은 서로 다른 종류의 정보가 한 JSON 안에 섞여 있습니다.
        데이터베이스에서는 보통 이런 정보를 **종류별로 다른 테이블**로 나눠 저장하고, 각 테이블을
        **키(key)**로 연결합니다.
        """
    )

    for table in RELATIONAL_SCHEMA:
        with st.expander(f"📋 {table['name']} 테이블", expanded=False):
            st.write(table["description"])
            table_rows = [
                {
                    "필드": col[0],
                    "데이터 타입": col[1],
                    "키": col[2] or "-",
                    "설명": col[3],
                }
                for col in table["columns"]
            ]
            st.dataframe(pd.DataFrame(table_rows), width="stretch", hide_index=True)

    st.markdown("**용어 정리**")
    st.markdown(
        """
        - **PK (Primary Key, 기본키)**: 해당 테이블의 각 행을 유일하게 식별하는 필드
        - **FK (Foreign Key, 외래키)**: 다른 테이블의 기본키를 가리키는 필드. 테이블 간 연결 고리
        - **1:N 관계**: 한 행이 다른 테이블의 여러 행과 연결되는 구조 (예: 인터뷰 1건 ↔ 발화 여러 개)
        - **N:M 관계**: 양쪽 모두 여러 개로 연결될 수 있는 구조. 보통 중간 테이블로 풀어낸다
        """
    )


def render_erd_diagram() -> None:
    st.markdown("#### 3단계 — 테이블 사이의 관계를 그림으로")
    st.caption("화살표는 '참조하는 방향'입니다. 예: Interview → Client 는 Interview가 Client를 참조.")

    lines = [
        "digraph ERD {",
        '  rankdir=LR;',
        '  bgcolor="transparent";',
        '  node [shape=box, style="rounded,filled", fillcolor="#ffffff", '
        'color="#1f4b6e", fontname="Helvetica", fontcolor="#17354d"];',
        '  edge [color="#8b5e34", fontname="Helvetica", fontcolor="#8b5e34", fontsize=10];',
    ]
    for table in RELATIONAL_SCHEMA:
        pk_fields = ", ".join(col[0] for col in table["columns"] if "PK" in col[2])
        label = f"{table['name']}\\nPK: {pk_fields}" if pk_fields else table["name"]
        lines.append(f'  {table["name"]} [label="{label}"];')

    for src, dst, cardinality, note in RELATIONAL_EDGES:
        lines.append(f'  {src} -> {dst} [label="{cardinality}"];')
    lines.append("}")

    st.graphviz_chart("\n".join(lines))

    st.markdown("**관계 요약**")
    edge_df = pd.DataFrame(
        [
            {"From": src, "To": dst, "관계": cardinality, "의미": note}
            for src, dst, cardinality, note in RELATIONAL_EDGES
        ]
    )
    st.dataframe(edge_df, width="stretch", hide_index=True)


def render_transcript(interview: dict) -> None:
    for utterance in interview["utterances"]:
        speaker, speaker_class = SPEAKER_LABELS.get(
            utterance["speaker"], ("미분류", "speaker-other")
        )
        with st.container(border=True):
            st.markdown(
                f"""
                <div class="transcript-head">
                    <span class="speaker-chip {speaker_class}">{speaker}</span>
                    <span class="time-chip">{utterance['start']} - {utterance['end']}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.write(utterance["text"])
            st.caption(f"confidence: {utterance['confidence']}")


def render_page_style() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #f4f1ea;
            --surface: rgba(255, 255, 255, 0.82);
            --surface-strong: #ffffff;
            --border: #ddd6c8;
            --text: #1f2937;
            --muted: #667085;
            --accent: #1f4b6e;
            --accent-soft: #d9e7f2;
            --accent-strong: #17354d;
            --warm: #8b5e34;
            --warm-soft: #f1e3d2;
            --student: #2f6f59;
            --student-soft: #dff1ea;
        }
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(217, 231, 242, 0.65), transparent 34%),
                radial-gradient(circle at top right, rgba(241, 227, 210, 0.7), transparent 28%),
                linear-gradient(180deg, #fbfaf7 0%, var(--bg) 100%);
            color: var(--text);
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }
        .hero-card {
            background: linear-gradient(135deg, rgba(255,255,255,0.92) 0%, rgba(249,245,238,0.88) 100%);
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 1.3rem 1.4rem 1.1rem 1.4rem;
            box-shadow: 0 10px 30px rgba(31, 41, 55, 0.06);
            margin-bottom: 1.25rem;
        }
        div[data-testid="stExpander"] {
            margin-bottom: 0.9rem;
        }
        div[data-testid="stHorizontalBlock"]:has(> div[data-testid="stMetric"]) {
            margin-bottom: 1rem;
        }
        .hero-eyebrow {
            display: inline-block;
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: var(--accent);
            background: var(--accent-soft);
            border-radius: 999px;
            padding: 0.35rem 0.6rem;
            margin-bottom: 0.7rem;
        }
        .hero-title {
            font-size: 2rem;
            line-height: 1.1;
            font-weight: 700;
            color: var(--accent-strong);
            margin-bottom: 0.45rem;
        }
        .hero-desc {
            color: var(--muted);
            font-size: 1rem;
            line-height: 1.6;
            margin: 0;
        }
        .stt-summary {
            background: linear-gradient(135deg, rgba(217, 231, 242, 0.45) 0%, rgba(255,255,255,0.92) 100%);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 1rem 1.1rem;
            margin-bottom: 0.9rem;
        }
        div[data-testid="stMetric"] {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 0.9rem 1rem;
            box-shadow: 0 8px 20px rgba(31, 41, 55, 0.04);
        }
        div[data-testid="stExpander"] {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 16px;
            overflow: hidden;
            box-shadow: 0 8px 20px rgba(31, 41, 55, 0.03);
        }
        div[data-testid="stDataFrame"] {
            border: 1px solid var(--border);
            border-radius: 16px;
            overflow: hidden;
            background: var(--surface-strong);
        }
        div[data-testid="stTabs"] {
            margin-top: 0.4rem;
        }
        div[data-testid="stTabs"] button {
            border-radius: 999px;
            color: var(--muted);
            border: 1px solid transparent;
            background: transparent;
            padding: 0.4rem 0.85rem;
            height: auto;
            min-height: 0;
            white-space: nowrap;
        }
        div[data-testid="stTabs"] button[aria-selected="true"] {
            color: var(--accent-strong);
            background: rgba(255, 255, 255, 0.88);
            border-color: var(--border);
        }
        div[data-testid="stTabs"] [data-baseweb="tab-list"] {
            gap: 0.35rem;
            background: rgba(255,255,255,0.4);
            padding: 0.35rem;
            border-radius: 18px;
            border: 1px solid rgba(221, 214, 200, 0.9);
            flex-wrap: wrap;
            row-gap: 0.35rem;
        }
        div[data-testid="stTabs"] [data-baseweb="tab-list"] [data-baseweb="tab-highlight"],
        div[data-testid="stTabs"] [data-baseweb="tab-list"] [data-baseweb="tab-border"] {
            display: none;
        }
        div[data-testid="stTabs"] [data-baseweb="tab-panel"] {
            padding-top: 1rem;
        }
        div[data-testid="stSelectbox"] > div,
        div[data-testid="stMultiSelect"] > div,
        div[data-testid="stTextInput"] > div {
            border-radius: 14px;
        }
        .transcript-head {
            display: flex;
            align-items: center;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-bottom: 0.55rem;
        }
        .speaker-chip,
        .time-chip {
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            padding: 0.25rem 0.6rem;
            font-size: 0.78rem;
            font-weight: 600;
        }
        .speaker-c {
            background: var(--accent-soft);
            color: var(--accent-strong);
        }
        .speaker-a {
            background: var(--student-soft);
            color: var(--student);
        }
        .speaker-other {
            background: #ece7dc;
            color: var(--muted);
        }
        .time-chip {
            background: #f4efe7;
            color: var(--warm);
        }
        h1, h2, h3 {
            letter-spacing: -0.02em;
        }
        h3 {
            color: var(--accent-strong);
        }
        .section-note {
            color: var(--muted);
            font-size: 0.95rem;
            margin-top: -0.2rem;
            margin-bottom: 0.8rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def seconds_to_timestamp(seconds: float) -> str:
    total = max(0, int(seconds))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


@st.cache_resource
def load_whisper_model(model_name: str):
    try:
        import whisper  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("`openai-whisper` 패키지가 설치되어 있지 않습니다.") from exc

    return whisper.load_model(model_name)


def transcribe_uploaded_audio(uploaded_file, model_name: str, language: str | None) -> dict:
    suffix = Path(uploaded_file.name).suffix or ".wav"
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            temp_path = Path(temp_file.name)

        model = load_whisper_model(model_name)
        transcribe_kwargs = {"task": "transcribe"}
        if language:
            transcribe_kwargs["language"] = language

        result = model.transcribe(str(temp_path), **transcribe_kwargs)
        segments = result.get("segments", [])
        rows = []
        for index, segment in enumerate(segments, start=1):
            rows.append(
                {
                    "segment_id": index,
                    "start": seconds_to_timestamp(float(segment["start"])),
                    "end": seconds_to_timestamp(float(segment["end"])),
                    "text": segment["text"].strip(),
                }
            )

        draft = {
            "interview_id": "UPLOAD-DRAFT",
            "metadata": {
                "source_file": uploaded_file.name,
                "model": model_name,
                "language": language or "auto",
                "total_utterances": len(rows),
            },
            "utterances": [
                {
                    "id": f"UTT-{idx:03d}",
                    "speaker": "U",
                    "start": row["start"],
                    "end": row["end"],
                    "text": row["text"],
                }
                for idx, row in enumerate(rows, start=1)
            ],
        }

        return {
            "text": result.get("text", "").strip(),
            "segments_df": pd.DataFrame(rows),
            "draft_json": draft,
        }
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass


def main() -> None:
    st.set_page_config(page_title="WE Meet STT Viewer", layout="wide")
    render_page_style()
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-eyebrow">Career Analysis Workspace</div>
            <div class="hero-title">WE Meet STT Viewer</div>
            <p class="hero-desc">
                상담 STT 데이터를 탐색하고, 직무 추천이 어떤 근거로 나오는지 구조적으로 분석하는 인터뷰 분석 대시보드입니다.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not DATA_DIR.exists():
        st.error(f"데이터 폴더를 찾을 수 없습니다: {DATA_DIR}")
        st.stop()

    interviews = load_interviews(DATA_DIR)
    if not interviews:
        st.warning("불러온 인터뷰 데이터가 없습니다.")
        st.stop()

    summary_df = pd.DataFrame(build_summary_rows(interviews))
    utterance_df = pd.DataFrame(build_utterance_rows(interviews))
    framework_df = pd.DataFrame(build_framework_rows())
    family_categories = ROLE_FAMILY_ORDER + sorted(
        set(framework_df["family"]) - set(ROLE_FAMILY_ORDER)
    )
    framework_df["family"] = pd.Categorical(
        framework_df["family"], categories=family_categories, ordered=True
    )
    framework_df = framework_df.sort_values(["family", "role"]).reset_index(drop=True)

    analysis_rows = []
    for interview in interviews:
        interview_analysis = analyze_interview(interview)
        for row in interview_analysis:
            analysis_rows.append({"interview_id": interview["interview_id"], **row})

    analysis_df = pd.DataFrame(analysis_rows)

    all_majors = sorted(summary_df["major"].unique().tolist())
    all_tags = sorted(
        {
            tag.strip()
            for tags in summary_df["tags"].tolist()
            for tag in tags.split(",")
            if tag.strip()
        }
    )

    with st.expander("필터", expanded=False):
        st.markdown(
            "<div class='section-note'>전체 탭에 공통으로 적용되는 기본 필터입니다. 전공을 모두 해제하면 결과가 비어 표시됩니다.</div>",
            unsafe_allow_html=True,
        )
        filter_col1, filter_col2 = st.columns(2)
        with filter_col1:
            selected_majors = st.multiselect("전공", options=all_majors, default=all_majors)
        with filter_col2:
            selected_tags = st.multiselect("태그", options=all_tags)

        if not selected_majors:
            st.caption("전공이 선택되어 있지 않아 모든 탭의 결과가 비어 있게 됩니다.")

    filtered_ids = set(
        summary_df.loc[summary_df["major"].isin(selected_majors), "interview_id"].tolist()
    )

    if selected_tags:
        selected_tag_set = set(selected_tags)
        tag_ids = {
            row["interview_id"]
            for _, row in summary_df.iterrows()
            if selected_tag_set & {tag.strip() for tag in row["tags"].split(",") if tag.strip()}
        }
        filtered_ids &= tag_ids

    filtered_summary_df = summary_df[summary_df["interview_id"].isin(filtered_ids)].copy()
    filtered_summary_df = filtered_summary_df.sort_values("interview_id")
    filtered_interviews = [
        interview for interview in interviews if interview["interview_id"] in filtered_ids
    ]
    filtered_analysis_df = analysis_df[analysis_df["interview_id"].isin(filtered_ids)].copy()

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_col1.metric("인터뷰 수", len(filtered_interviews))
    metric_col2.metric("전공 수", filtered_summary_df["major"].nunique() if not filtered_summary_df.empty else 0)
    metric_col3.metric(
        "총 발화 수",
        int(filtered_summary_df["utterance_count"].sum()) if not filtered_summary_df.empty else 0,
    )
    metric_col4.metric("직무 후보 수", filtered_analysis_df["role"].nunique() if not filtered_analysis_df.empty else 0)

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
        ["목록", "대화 보기", "추천 구조", "직무 추천 분석", "상담사 코멘트 분석", "본문 검색 결과", "Whisper STT"]
    )

    with tab1:
        st.subheader("인터뷰 목록")
        st.caption("현재 필터 기준으로 인터뷰 메타데이터를 빠르게 훑어볼 수 있습니다.")
        st.dataframe(filtered_summary_df, width="stretch", hide_index=True)

        st.divider()
        st.markdown("### 📚 이 표는 어떻게 만들어졌을까?")
        st.markdown(
            "이 화면은 여러 건의 상담 인터뷰 원본 데이터에서 **필요한 필드만 뽑아 한 줄씩 요약**한 결과입니다. "
            "아래 순서대로 따라가면, 데이터가 어떻게 구조화되어 있는지 한 번에 이해할 수 있어요."
        )

        render_summary_column_dictionary(filtered_summary_df)

        st.divider()
        render_relational_schema()

        st.divider()
        render_erd_diagram()

        st.divider()
        st.markdown("#### 🎯 정리하면")
        st.markdown(
            """
            1. 원본 데이터는 JSON 한 파일 안에 **인터뷰, 학생, 상담사, 발화, 태그**가 섞여 있습니다.
            2. 이를 데이터베이스 관점에서 보면 **5개 테이블**로 나눌 수 있고, 각각 PK/FK로 연결됩니다.
            3. 지금 보시는 목록은 그중 **Interview + Client** 정보를 조인(JOIN)한 뒤 필요한 컬럼만 뽑은 결과예요.
            4. 다른 탭(대화 보기, 직무 추천 분석)은 **Utterance 테이블**을 활용하는 예시입니다.
            """
        )

    with tab2:
        st.subheader("대화 상세")
        st.caption("선택한 인터뷰의 메타데이터와 전체 상담 발화를 순서대로 확인합니다.")
        options = [interview["interview_id"] for interview in filtered_interviews]
        if not options:
            st.info("현재 필터 조건에 맞는 인터뷰가 없습니다.")
        else:
            selected_id = st.selectbox("인터뷰 선택", options=options, key="transcript_select")
            selected_interview = next(
                interview for interview in filtered_interviews if interview["interview_id"] == selected_id
            )
            meta = selected_interview["metadata"]
            client = meta["client"]

            info_col1, info_col2 = st.columns(2)
            with info_col1:
                st.markdown(
                    f"""
                    **학생명**: {client['name']}  
                    **전공**: {client['major']}  
                    **학년**: {client['grade']}
                    """
                )
            with info_col2:
                st.markdown(
                    f"""
                    **상담일**: {meta['session_date']}  
                    **상담 시간**: {meta['duration']}  
                    **태그**: {", ".join(client['persona_tags'])}
                    """
                )

            render_transcript(selected_interview)

    with tab3:
        st.subheader("직무 추천 구조")
        st.caption("직무 추천이 어떤 기준으로 계산되는지 미리 보는 프레임워크입니다.")
        st.markdown(
            """
            이 앱은 직무 추천을 아래 4개 축으로 구조화합니다.

            1. **전공 적합성**: 해당 직무에 자주 연결되는 전공인지  
            2. **명시적 관심사**: `persona_tags`에 직무가 직접 들어 있는지  
            3. **본문 근거**: 대화 중 반복되는 키워드와 경험이 무엇인지  
            4. **업무 방식 적합성**: 문제를 어떻게 푸는 사람인지
            """
        )

        st.dataframe(framework_df, width="stretch", hide_index=True)

    with tab4:
        st.subheader("인터뷰별 직무 추천 분석")
        options = [interview["interview_id"] for interview in filtered_interviews]
        if not options:
            st.info("현재 필터 조건에 맞는 인터뷰가 없습니다.")
        else:
            selected_id = st.selectbox("분석할 인터뷰 선택", options=options, key="analysis_select")
            selected_interview = next(
                interview for interview in filtered_interviews if interview["interview_id"] == selected_id
            )
            selected_rows = analyze_interview(selected_interview)
            client = selected_interview["metadata"]["client"]

            summary_col1, summary_col2, summary_col3 = st.columns(3)
            summary_col1.metric("학생", client["name"])
            summary_col2.metric("전공", client["major"])
            summary_col3.metric("관심 태그 수", len(client["persona_tags"]))
            st.caption(f"명시된 관심 직무 태그: {', '.join(client['persona_tags'])}")

            top_rows = selected_rows[:5]
            competency_rows = extract_competency_profile(selected_interview)
            if not top_rows:
                st.info("추천 가능한 직무 신호를 찾지 못했습니다.")
            else:
                with st.container(border=True):
                    st.subheader("추천 사유 요약문")
                    st.write(generate_recommendation_summary(selected_interview, top_rows, competency_rows))

                analysis_col1, analysis_col2 = st.columns([1.1, 1.4])
                with analysis_col1:
                    st.subheader("역량 축 자동 추출")
                    competency_df = pd.DataFrame(competency_rows)
                    if competency_df.empty:
                        st.info("역량 축 신호를 충분히 찾지 못했습니다.")
                    else:
                        st.dataframe(competency_df, width="stretch", hide_index=True)

                with analysis_col2:
                    st.subheader("상위 추천 직무")
                    top_df = pd.DataFrame(top_rows)
                    st.dataframe(
                        top_df[["role", "family", "score", "problem_focus", "work_mode", "evidence"]],
                        width="stretch",
                        hide_index=True,
                    )

                st.subheader("추천 근거 카드")
                for row in top_rows[:3]:
                    with st.container(border=True):
                        st.markdown(f"### {row['role']}")
                        st.markdown(
                            f"""
                            **직무 패밀리**: {row['family']}  
                            **핵심 문제**: {row['problem_focus']}  
                            **업무 방식**: {row['work_mode']}  
                            **추천 점수**: {row['score']}  
                            **근거 요약**: {row['evidence']}
                            """
                        )

                        support_rows = find_supporting_utterances(selected_interview, row["role"])
                        if support_rows:
                            st.markdown("**근거가 된 발화**")
                            for support in support_rows:
                                st.write(f"- {support['speaker']}: {support['text']}")

                st.subheader("추천 해석 방식")
                structure_df = pd.DataFrame(
                    [
                        {"분석 축": "전공 적합성", "설명": "해당 직무에 연결되는 전공인지 확인"},
                        {"분석 축": "명시적 관심사", "설명": "persona_tags에 직무가 직접 포함되는지 확인"},
                        {"분석 축": "본문 키워드", "설명": "대화 속 반복 표현과 경험 키워드를 근거로 반영"},
                        {"분석 축": "업무 방식", "설명": "병목 찾기, 실험, 설명, 인터뷰, 협상 같은 일하는 방식을 직무와 연결"},
                        {"분석 축": "역량 축", "설명": "분석형, 설득형, 조율형, 실험형 신호를 학생 발화에서 추출"},
                    ]
                )
                st.dataframe(structure_df, width="stretch", hide_index=True)

    with tab5:
        st.subheader("상담사 코멘트 분석")
        st.caption("탐색, 공감, 구조화, 실행 제안 관점에서 상담사 발화를 나눠서 봅니다.")
        options = [interview["interview_id"] for interview in filtered_interviews]
        if not options:
            st.info("현재 필터 조건에 맞는 인터뷰가 없습니다.")
        else:
            selected_id = st.selectbox("분석할 인터뷰 선택", options=options, key="counselor_select")
            selected_interview = next(
                interview for interview in filtered_interviews if interview["interview_id"] == selected_id
            )
            quality = analyze_counselor_quality(selected_interview)

            q1, q2, q3, q4, q5 = st.columns(5)
            q1.metric("탐색 점수", quality["exploration_score"])
            q2.metric("공감 점수", quality["empathy_score"])
            q3.metric("구조화 점수", quality["structuring_score"])
            q4.metric("실행 제안 점수", quality["actionability_score"])
            q5.metric("균형 점수", quality["balance_score"])

            detail_df = pd.DataFrame(
                [
                    {"항목": "탐색 질문 수", "값": quality["open_question_count"]},
                    {"항목": "공감/지지 표현 수", "값": quality["validation_count"]},
                    {"항목": "정리/요약 표현 수", "값": quality["summary_count"]},
                    {"항목": "실행 제안 수", "값": quality["action_count"]},
                    {"항목": "상담사 총 발화 수", "값": quality["total_counselor_utterances"]},
                ]
            )
            st.dataframe(detail_df, width="stretch", hide_index=True)

            st.subheader("유형별 예시 발화")
            categories = ["탐색 질문", "공감/지지", "구조화/정리", "실행 제안"]
            example_col1, example_col2 = st.columns(2)
            for idx, category in enumerate(categories):
                target_col = example_col1 if idx % 2 == 0 else example_col2
                with target_col:
                    with st.container(border=True):
                        st.markdown(f"### {category}")
                        examples = find_counselor_examples(selected_interview, category)
                        if examples:
                            for example in examples:
                                st.write(f"- {example}")
                        else:
                            st.write("- 해당 유형으로 분류된 발화가 없습니다.")

            st.subheader("해석 가이드")
            st.markdown(
                """
                - **탐색 점수**: 학생 이야기를 충분히 끌어내는 질문이 있는지  
                - **공감 점수**: 학생 감정과 상황을 인정하는 표현이 있는지  
                - **구조화 점수**: 대화 내용을 정리하고 의미를 묶어주는지  
                - **실행 제안 점수**: 다음 행동으로 이어지는 제안이 있는지  
                - **균형 점수**: 탐색 없이 바로 조언하거나, 조언 없이 끝나는 편향이 적은지
                """
            )

    with tab6:
        st.subheader("본문 검색 결과")
        st.caption("현재 필터 범위 안에서 특정 표현이 등장하는 발화를 바로 찾습니다.")
        search_text = st.text_input(
            "검색어 입력",
            placeholder="예: 반도체, 마케터, 국제기구",
            key="tab_search_text",
        )
        if search_text:
            result_df = utterance_df[
                utterance_df["interview_id"].isin(filtered_ids)
                & utterance_df["text"].str.contains(search_text, case=False, na=False)
            ].copy()
            st.dataframe(result_df, width="stretch", hide_index=True)
        else:
            st.info("검색어를 입력하면 현재 필터 범위 안에서 발화 단위 결과를 볼 수 있습니다.")

    with tab7:
        st.subheader("Whisper STT")
        st.caption("직무 인터뷰 음성 파일을 업로드하면 Whisper로 자동 전사하고, 텍스트와 JSON 초안을 바로 확인합니다.")

        config_col1, config_col2 = st.columns(2)
        with config_col1:
            whisper_model = st.selectbox("Whisper 모델", options=WHISPER_MODEL_OPTIONS, index=1, help="작을수록 빠르고, 클수록 정확도가 높지만 더 느립니다.")
        with config_col2:
            language_label = st.selectbox("언어", options=WHISPER_LANGUAGE_OPTIONS, index=1)
        selected_language = WHISPER_LANGUAGE_MAP[language_label]

        uploaded_audio = st.file_uploader(
            "음성 파일 업로드",
            type=WHISPER_FILE_TYPES,
            help="파일을 올리면 자동으로 전사가 시작됩니다.",
        )

        if uploaded_audio is not None:
            st.audio(uploaded_audio)
            st.markdown(
                f"""
                <div class="stt-summary">
                    <strong>업로드 파일</strong>: {uploaded_audio.name}<br/>
                    <strong>선택 모델</strong>: {whisper_model}<br/>
                    <strong>전사 언어</strong>: {language_label}
                </div>
                """,
                unsafe_allow_html=True,
            )

            try:
                with st.spinner("Whisper로 음성을 전사하는 중입니다..."):
                    transcription = transcribe_uploaded_audio(
                        uploaded_audio,
                        whisper_model,
                        selected_language,
                    )
            except Exception as exc:
                st.error(
                    "Whisper 전사에 실패했습니다. `openai-whisper` 설치와 `ffmpeg` 사용 가능 여부를 확인해주세요."
                )
                st.exception(exc)
            else:
                st.success("전사가 완료되었습니다.")

                transcript_col1, transcript_col2 = st.columns([1.3, 1])
                with transcript_col1:
                    st.subheader("전사 텍스트")
                    st.text_area(
                        "full_transcript",
                        value=transcription["text"],
                        height=280,
                        label_visibility="collapsed",
                    )
                    st.download_button(
                        "텍스트 다운로드",
                        data=transcription["text"],
                        file_name=f"{Path(uploaded_audio.name).stem}_transcript.txt",
                        mime="text/plain",
                    )

                with transcript_col2:
                    st.subheader("세그먼트")
                    st.dataframe(transcription["segments_df"], width="stretch", hide_index=True)

                st.subheader("JSON 초안")
                json_text = json.dumps(transcription["draft_json"], ensure_ascii=False, indent=2)
                st.code(json_text, language="json")
                st.download_button(
                    "JSON 초안 다운로드",
                    data=json_text,
                    file_name=f"{Path(uploaded_audio.name).stem}_draft.json",
                    mime="application/json",
                )
        else:
            st.info("음성 파일을 올리면 이 탭 안에서 자동으로 Whisper STT가 실행됩니다.")


if __name__ == "__main__":
    main()
