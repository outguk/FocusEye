# 👁️ FocusEye+: 비언어 집중도 분석 웹 서비스

AI 기반의 실시간 웹캠 영상 분석을 통해 사용자의 **눈 깜빡임 빈도**에 따른 경고 알림을 제공하고 **집중 시간**을 측정하여 **비언어적 집중도 평가**를 제공하는 웹 서비스입니다.  
YOLOv3와 Mediapipe를 활용한 영상 인식과 FastAPI 기반 백엔드 서버를 통해 실시간 분석 결과를 시각적으로 제공합니다.

---

## 주요 기능

- **실시간 영상 업로드 및 분석**
  - 웹캠 기반 실시간 영상 스트리밍 처리
  - YOLOv3를 이용한 사람 탐지
  - Mediapipe Face Mesh 기반 눈 깜빡임 인식

- **집중도 분석**
  - 사용자의 눈 깜빡임 횟수(Blink Count) 측정
  - 사람 탐지 시간 기반 집중 시간 측정

- **사용자별 세션 관리 및 기록**
  - 로그인/회원가입 기능
  - 세션별 측정 결과 저장 및 조회
  - 깜빡임 빈도(분당 횟수) 및 총 집중 시간 시각화 제공

- **웹 기반 UI**
  - 사용자별 메인 화면 및 분석 결과 페이지 제공
  - WebSocket을 이용한 실시간 데이터 전송

---

## 사용 기술 스택

### Backend
- **Python 3.10**
- **FastAPI** – 비동기 REST API 서버
- **SQLAlchemy** – ORM 기반 DB 모델 정의
- **MySQL** – 사용자 및 세션 데이터 저장
- **WebSocket** – 실시간 분석 데이터 스트리밍

### AI/영상 처리
- **YOLOv3** – 객체 탐지를 통한 사람 인식
- **Mediapipe** – Face Mesh 기반 눈 감김(EAR) 추적
- **OpenCV** – 프레임 처리 및 시각화

---

## 작동 방식

1. **회원가입/로그인 후 접속**
2. 웹 페이지에서 `측정 시작` 버튼 클릭 시 웹캠 분석 시작
3. YOLOv3를 통해 사람 유무 탐지 → 집중 시간 측정
4. Mediapipe로 눈 깜빡임 여부 분석 → Blink Count 측정
5. 분석 종료 시 세션 저장 및 결과 시각화 제공

---

## 프로젝트 구조

```
📁 templates/             # 화면 디자인
├── main.py                # FastAPI 서버, WebSocket 및 API 처리
├── ai_processing.py       # YOLO 및 Mediapipe 분석 모듈
├── models.py              # SQLAlchemy ORM 모델 정의
├── database.py            # DB 연결 및 초기화 설정
├── coco.names             # YOLO 클래스 이름 목록
```

## 실제 서비스 이미지
1. 분석 시 웹캠 화면
<img width="238" alt="1" src="https://github.com/user-attachments/assets/558fe4a2-c13d-4529-aec7-dd772593e9e7" />

2. 분석 페이지 화면
<img width="455" alt="2" src="https://github.com/user-attachments/assets/2bc380ae-3353-4559-966b-fbb3031dd867" />

