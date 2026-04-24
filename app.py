from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st


DATA_DIR = Path(__file__).parent / "stt_samples"

COMPETENCY_AXES = {
    "분석형": {
        "description": "패턴을 찾고 원인을 구조적으로 해석하는 성향",
        "keywords": ["분석", "지표", "패턴", "원인", "해석", "비교", "데이터", "병목", "구조", "왜"],
    },
    "설득형": {
        "description": "메시지를 다듬고 전달 효과를 높이는 성향",
        "keywords": ["설득", "카피", "표현", "말투", "브랜드", "문장", "전달", "설명", "콘텐츠"],
    },
    "조율형": {
        "description": "여러 사람과 관점을 연결하고 공통점을 찾는 성향",
        "keywords": ["조율", "팀", "갈등", "공통", "협력", "중간", "의견", "합의", "정리"],
    },
    "실험형": {
        "description": "가설을 세우고 직접 검증하며 개선하는 성향",
        "keywords": ["실험", "검증", "가설", "조건", "바꿔", "측정", "테스트", "시도", "최적화"],
    },
}

ROLE_CATALOG = {
    "브랜드 콘텐츠": {
        "family": "콘텐츠/마케팅",
        "problem_focus": "브랜드 메시지와 반응 설계",
        "work_mode": "콘텐츠 기획, 반응 분석, 카피 실험",
        "fit_majors": ["영어영문학과", "국어국문학과", "언론정보학과"],
        "keywords": ["브랜드", "카피", "콘텐츠", "인스타", "반응", "저장", "팔로워"],
    },
    "카피라이팅": {
        "family": "콘텐츠/마케팅",
        "problem_focus": "짧은 문장으로 설득과 전환 만들기",
        "work_mode": "문장 실험, 브랜드 보이스 설계",
        "fit_majors": ["영어영문학과", "국어국문학과"],
        "keywords": ["카피", "문장", "단어", "브랜드 말투", "수사학", "표현"],
    },
    "콘텐츠 에디터": {
        "family": "콘텐츠/마케팅",
        "problem_focus": "콘텐츠 품질과 메시지 정합성 관리",
        "work_mode": "편집, 피드백, 구조 정리",
        "fit_majors": ["영어영문학과", "국어국문학과", "언론정보학과"],
        "keywords": ["피드백", "수정", "글", "논리", "에디터"],
    },
    "콘텐츠 스트래티지": {
        "family": "콘텐츠/마케팅",
        "problem_focus": "브랜드 관점에서 콘텐츠 방향 설계",
        "work_mode": "콘텐츠 기준 정의, 브랜드 톤 전략화",
        "fit_majors": ["국어국문학과", "영어영문학과", "언론정보학과"],
        "keywords": ["전략", "브랜드", "기준", "분석", "저장해두는"],
    },
    "데이터 엔지니어링": {
        "family": "데이터/플랫폼",
        "problem_focus": "데이터 흐름과 처리 인프라 설계",
        "work_mode": "파이프라인 구축, 병목 최적화",
        "fit_majors": ["컴퓨터공학과"],
        "keywords": ["파이프라인", "Spark", "로그 데이터", "병목", "Kafka", "흐름"],
    },
    "MLOps": {
        "family": "데이터/플랫폼",
        "problem_focus": "모델 운영 자동화와 배포 안정화",
        "work_mode": "데이터-모델-인프라 연결",
        "fit_majors": ["컴퓨터공학과"],
        "keywords": ["모델", "배포", "모니터링", "자동화", "인프라", "MLOps"],
    },
    "파이프라인": {
        "family": "데이터/플랫폼",
        "problem_focus": "대규모 데이터 처리 구조 설계",
        "work_mode": "ETL, 배치, 스트리밍 설계",
        "fit_majors": ["컴퓨터공학과"],
        "keywords": ["파이프라인", "데이터 흐름", "처리", "최적화", "구조"],
    },
    "운영 최적화": {
        "family": "산업/운영",
        "problem_focus": "병목 제거와 효율 향상",
        "work_mode": "시뮬레이션, 대기열 분석, 운영 개선",
        "fit_majors": ["산업공학과"],
        "keywords": ["병목", "대기열", "운영", "효율", "시뮬레이션", "흐름"],
    },
    "프로덕트 데이터 분석": {
        "family": "데이터/제품",
        "problem_focus": "사용자 행동 데이터로 제품 의사결정 지원",
        "work_mode": "지표 설계, 퍼널 분석, 실험 해석",
        "fit_majors": ["산업공학과", "경제학부", "컴퓨터공학과"],
        "keywords": ["데이터", "지표", "사용자", "흐름", "실험", "분석"],
    },
    "실험 설계": {
        "family": "데이터/제품",
        "problem_focus": "가설 검증을 위한 테스트 구조 설계",
        "work_mode": "가설 분해, 검증 조건 설계",
        "fit_majors": ["산업공학과", "심리학과", "경제학부"],
        "keywords": ["가설", "검증", "실험", "조건", "데이터부터 보자"],
    },
    "반도체 공정": {
        "family": "R&D/제조",
        "problem_focus": "공정 조건 최적화와 수율 개선",
        "work_mode": "공정 개발, 이상 원인 분석, 조건 조정",
        "fit_majors": ["물리천문학부", "재료공학부", "전기정보공학부"],
        "keywords": ["반도체", "공정", "측정", "노이즈", "장비", "조건"],
    },
    "광학 센서": {
        "family": "R&D/하드웨어",
        "problem_focus": "광학·이미지 센서 성능 향상",
        "work_mode": "계측, 센서 평가, 신호 분석",
        "fit_majors": ["물리천문학부", "전기정보공학부"],
        "keywords": ["센서", "광학", "이미지", "노이즈", "신호", "측정값"],
    },
    "R&D 엔지니어": {
        "family": "R&D/하드웨어",
        "problem_focus": "실험 기반 문제 해결과 제품 개선",
        "work_mode": "원인 분석, 실험 반복, 기술 검증",
        "fit_majors": ["물리천문학부", "환경공학부", "컴퓨터공학과"],
        "keywords": ["실험", "원인", "데이터", "장비", "연구", "검증"],
    },
    "데이터 저널리즘": {
        "family": "미디어/데이터",
        "problem_focus": "데이터로 사회 이슈를 해석해 전달",
        "work_mode": "데이터 분석, 시각화, 기사 구성",
        "fit_majors": ["언론정보학과", "사회학과", "경제학부"],
        "keywords": ["기사", "데이터", "시각화", "투표율", "패턴", "인포그래픽"],
    },
    "마케팅 인사이트": {
        "family": "마케팅/분석",
        "problem_focus": "소비자 행동과 반응 해석",
        "work_mode": "캠페인 해석, 행동 인사이트 도출",
        "fit_majors": ["언론정보학과", "경영학과", "경제학부"],
        "keywords": ["반응", "읽혔", "시각화", "소비자", "인사이트", "마케팅"],
    },
    "B2B 콘텐츠": {
        "family": "콘텐츠/마케팅",
        "problem_focus": "전문 정보를 설득력 있는 콘텐츠로 번역",
        "work_mode": "리포트, 사례, 정보 구조화",
        "fit_majors": ["언론정보학과", "영어영문학과", "국어국문학과"],
        "keywords": ["리포트", "스토리", "기사", "콘텐츠", "설명"],
    },
    "UX 리서치": {
        "family": "리서치/제품",
        "problem_focus": "사용자 문제 발견과 행동 이유 탐색",
        "work_mode": "인터뷰, 관찰, 인사이트 정리",
        "fit_majors": ["심리학과"],
        "keywords": ["인터뷰", "사용자", "질문", "이유", "행동", "설문"],
    },
    "HR테크": {
        "family": "HR/테크",
        "problem_focus": "채용과 평가 시스템의 공정성·효율성 개선",
        "work_mode": "채용 데이터, 편향 검토, 시스템 설계",
        "fit_majors": ["심리학과", "컴퓨터공학과"],
        "keywords": ["채용", "편향", "공정", "AI 면접", "HR테크"],
    },
    "조직개발": {
        "family": "HR/조직",
        "problem_focus": "팀 문화와 협업 구조 개선",
        "work_mode": "조직 진단, 갈등 해석, 문화 개선",
        "fit_majors": ["심리학과", "경영학과"],
        "keywords": ["조직", "팀", "갈등", "문화", "심리적 안전감"],
    },
    "핀테크": {
        "family": "금융/데이터",
        "problem_focus": "금융 문제를 데이터와 제품으로 해결",
        "work_mode": "리스크 분석, 지표 해석, 서비스 개선",
        "fit_majors": ["경제학부", "컴퓨터공학과"],
        "keywords": ["금융", "핀테크", "리스크", "대출", "고객", "토스"],
    },
    "데이터 애널리스트": {
        "family": "데이터/분석",
        "problem_focus": "데이터를 의사결정 언어로 해석",
        "work_mode": "모델 해석, 비즈니스 지표 분석",
        "fit_majors": ["경제학부", "산업공학과", "경영학과"],
        "keywords": ["계수", "해석", "분석", "비즈니스", "지표", "SQL"],
    },
    "퀀트 리서치": {
        "family": "금융/정량",
        "problem_focus": "금융 모델과 변수 해석",
        "work_mode": "정량 모델링, 수익률 분석",
        "fit_majors": ["경제학부", "수리과학부", "통계학과"],
        "keywords": ["수익률", "모델", "회귀분석", "변수", "계수", "예측"],
    },
    "기후테크": {
        "family": "환경/기술",
        "problem_focus": "기술 기반 탄소 감축과 운영 개선",
        "work_mode": "배출 저감, 에너지 최적화, 공정 개선",
        "fit_majors": ["환경공학부"],
        "keywords": ["기후", "감축", "배출", "에너지", "운영 조건", "최적화"],
    },
    "탄소관리": {
        "family": "환경/데이터",
        "problem_focus": "배출량 측정과 감축 포인트 분석",
        "work_mode": "배출 데이터 분석, 보고, 감축 전략 수립",
        "fit_majors": ["환경공학부"],
        "keywords": ["탄소", "배출량", "계측", "에너지 사용량", "감축", "데이터"],
    },
    "자원순환": {
        "family": "환경/공정",
        "problem_focus": "폐기물·배터리 재활용 공정 개선",
        "work_mode": "공정 이해, 처리 효율 개선, 안정화",
        "fit_majors": ["환경공학부", "화학생물공학부"],
        "keywords": ["재활용", "폐배터리", "자원순환", "공정", "처리"],
    },
    "공공정책": {
        "family": "정책/대외협력",
        "problem_focus": "정책 변화 해석과 대응 전략 수립",
        "work_mode": "규제 분석, 입법 모니터링, 대외 커뮤니케이션",
        "fit_majors": ["정치외교학부", "행정학과"],
        "keywords": ["정책", "규제", "기업 공공정책팀", "정부", "입법"],
    },
    "디지털 거버넌스": {
        "family": "정책/기술",
        "problem_focus": "기술 규제와 국제 조율 이해",
        "work_mode": "거버넌스 분석, 국제 규제 비교",
        "fit_majors": ["정치외교학부", "컴퓨터공학과"],
        "keywords": ["AI 거버넌스", "EU AI Act", "규제", "국제", "디지털"],
    },
    "글로벌 BD": {
        "family": "사업개발/대외협력",
        "problem_focus": "이해관계자 조율과 파트너십 개발",
        "work_mode": "협상, 공통 언어 찾기, 파트너십 기획",
        "fit_majors": ["정치외교학부", "경영학과"],
        "keywords": ["합의안", "이해관계자", "공통 언어", "파트너십", "글로벌"],
    },
}

ROLE_FAMILY_ORDER = [
    "콘텐츠/마케팅",
    "데이터/플랫폼",
    "데이터/제품",
    "산업/운영",
    "R&D/제조",
    "R&D/하드웨어",
    "미디어/데이터",
    "마케팅/분석",
    "리서치/제품",
    "HR/테크",
    "HR/조직",
    "금융/데이터",
    "금융/정량",
    "환경/기술",
    "환경/데이터",
    "환경/공정",
    "정책/대외협력",
    "정책/기술",
    "사업개발/대외협력",
]


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
                    "speaker": "상담사" if utterance["speaker"] == "C" else "학생",
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

    open_question_markers = ["어떤", "어떻게", "왜", "언제", "무엇", "어디", "누구", "먼저", "계기"]
    validation_markers = ["이해해요", "그럴 수", "괜찮", "위안", "맞아요", "충분히", "그럴 것 같", "고생"]
    summary_markers = ["정리", "들어보니", "보면", "말한 것", "말씀하신", "즉", "그러니까"]
    action_markers = ["해보세요", "정리해보세요", "찾아보세요", "익혀두", "포트폴리오", "다음", "써보세요"]

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
    markers_by_category = {
        "탐색 질문": ["어떤", "어떻게", "왜", "언제", "무엇", "계기"],
        "공감/지지": ["이해해요", "그럴 수", "괜찮", "위안", "맞아요", "고생"],
        "구조화/정리": ["정리", "들어보니", "보면", "말한 것", "즉", "그러니까"],
        "실행 제안": ["해보세요", "정리해보세요", "찾아보세요", "익혀두", "포트폴리오", "써보세요"],
    }
    markers = markers_by_category[category]
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
        match_count = sum(1 for keyword in keywords if keyword in utterance["text"])
        if match_count > 0:
            matches.append(
                {
                    "speaker": "상담사" if utterance["speaker"] == "C" else "학생",
                    "text": utterance["text"],
                    "weight": match_count,
                }
            )
    matches.sort(key=lambda row: (-row["weight"], row["speaker"]))
    return matches[:limit]


def render_transcript(interview: dict) -> None:
    for utterance in interview["utterances"]:
        speaker = "상담사" if utterance["speaker"] == "C" else "학생"
        speaker_class = "speaker-c" if utterance["speaker"] == "C" else "speaker-a"
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
        div[data-testid="stTabs"] button {
            border-radius: 999px;
            color: var(--muted);
            border: 1px solid transparent;
            background: transparent;
            padding: 0.45rem 0.9rem;
        }
        div[data-testid="stTabs"] button[aria-selected="true"] {
            color: var(--accent-strong);
            background: rgba(255, 255, 255, 0.88);
            border-color: var(--border);
        }
        div[data-testid="stTabs"] [data-baseweb="tab-list"] {
            gap: 0.4rem;
            background: rgba(255,255,255,0.4);
            padding: 0.35rem;
            border-radius: 999px;
            border: 1px solid rgba(221, 214, 200, 0.9);
        }
        div[data-testid="stSelectbox"] > div,
        div[data-testid="stMultiSelect"] > div,
        div[data-testid="stTextInput"] > div {
            border-radius: 14px;
        }
        .transcript-head {
            display: flex;
            align-items: center;
            gap: 0.55rem;
            margin-bottom: 0.35rem;
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
    framework_df = pd.DataFrame(build_framework_rows()).sort_values(["family", "role"])

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
            "<div class='section-note'>전체 탭에 공통으로 적용되는 기본 필터입니다.</div>",
            unsafe_allow_html=True,
        )
        filter_col1, filter_col2 = st.columns(2)
        with filter_col1:
            selected_majors = st.multiselect("전공", options=all_majors, default=all_majors)
        with filter_col2:
            selected_tags = st.multiselect("태그", options=all_tags)

    filtered_ids = set(
        summary_df.loc[summary_df["major"].isin(selected_majors), "interview_id"].tolist()
    )

    if selected_tags:
        tag_ids = {
            row["interview_id"]
            for _, row in summary_df.iterrows()
            if any(tag in row["tags"] for tag in selected_tags)
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

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["목록", "대화 보기", "추천 구조", "직무 추천 분석", "상담사 코멘트 분석", "본문 검색 결과"]
    )

    with tab1:
        st.subheader("인터뷰 목록")
        st.caption("현재 필터 기준으로 인터뷰 메타데이터를 빠르게 훑어볼 수 있습니다.")
        st.dataframe(filtered_summary_df, width="stretch", hide_index=True)

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


if __name__ == "__main__":
    main()
