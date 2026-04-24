# WE Meet STT Viewer

`stt_samples` 폴더의 상담 JSON 데이터를 탐색하고, 직무 추천 근거를 구조적으로 분석하는 Streamlit 앱입니다.

## 실행

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 기능

- 인터뷰 목록 조회
- 전공/태그 기준 필터링
- 발화 본문 검색
- 인터뷰별 메타데이터 및 전체 대화 확인
- 직무 추천 구조 프레임워크 조회
- 인터뷰별 상위 추천 직무와 근거 발화 확인
- 역량 축(분석형/설득형/조율형/실험형) 자동 추출
- 추천 사유 요약문 자동 생성
- 상담사 코멘트 품질 분석
