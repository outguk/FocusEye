# main.py
from fastapi import FastAPI, Depends, HTTPException, Request, Form, WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState
from sqlalchemy.orm import Session
from database import SessionLocal, init_db
from contextlib import asynccontextmanager
from pydantic import BaseModel
from passlib.context import CryptContext
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse
from typing import List
from datetime import datetime

import asyncio
import json
import logging
import models

""" 이 코드에서서 print 문은 모두 상태 작동을 위한 로그 남기기"""

# react 연결을 위함
from fastapi.middleware.cors import CORSMiddleware

# 영상 처리 로직 라이브러리 ai_processing.py의 함수 import
from ai_processing import start_video_capture, stop_video_capture, process_camera_data, user_states_lock

# 로깅 설정
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# SQLAlchemy 관련 로거 설정
logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
logging.getLogger("sqlalchemy.engine.Engine").setLevel(logging.WARNING)
logging.getLogger("sqlalchemy.orm").setLevel(logging.WARNING)
logging.getLogger("sqlalchemy.pool").setLevel(logging.WARNING)
logging.getLogger("sqlalchemy").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# 연결된 클라이언트 WebSocket들을 관리하기 위한 클래스
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        # 새로운 WebSocket 연결 수락 및 활성 연결 목록에 추가
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        # WebSocket 연결 종료 시 활성 연결 목록에서 제거
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send_message(self, message: str):
        # 모든 활성 연결에 메시지 전송
        for connection in self.active_connections:
            await connection.send_text(message)

# 소켓 관리 클래스 연결 객체 생성
manager = ConnectionManager()

# 비밀번호 해시를 위해 Passlib 사용

# FastAPI 애플리케이션 설정
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 애플리케이션 시작 시 데이터베이스 초기화
    logger.info("Starting lifespan... Initializing database if not exists.")
    init_db()  # 데이터베이스 초기화
    yield
    logger.info("Ending lifespan...")

# FastAPI 초기화, 템플릿 설정, 비밀번호 해시를 위해 Passlib 사용
app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory="templates")
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# 데이터베이스 세션 생성 함수
# 각 요청에 대해 독립적인 데이터베이스 세션을 제공
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# 비밀번호 해시 함수
# 입력된 비밀번호를 해시하여 반환
def get_password_hash(password):
    return pwd_context.hash(password)

# 비밀번호 검증 함수
# 평문 비밀번호와 해시된 비밀번호를 비교하여 일치 여부 반환
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

# 메인 페이지 렌더링 엔드포인트
@app.get("/", response_class=HTMLResponse)
async def get_main_page(request: Request):
    # main.html 템플릿을 렌더링하여 반환
    return templates.TemplateResponse("main.html", {"request": request})

# 회원가입 폼 렌더링 엔드포인트
@app.get("/signup", response_class=HTMLResponse)
async def get_signup_form(request: Request):
    # signup.html 템플릿을 렌더링하여 반환
    return templates.TemplateResponse("signup.html", {"request": request})

# 회원가입 엔드포인트 정의 (Form 데이터로 수집)
@app.post("/signup")
def signup(
    name: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    # 이메일 중복 확인
    existing_user = db.query(models.User).filter(models.User.email == email).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="이미 존재하는 이메일입니다.")
    # 비밀번호 해싱
    hashed_password = get_password_hash(password)
    # 새로운 사용자 생성 및 데이터베이스에 추가
    new_user = models.User(name=name, email=email, password=hashed_password)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    # 회원가입 완료 후 메인 화면으로 리다이렉트
    return RedirectResponse(url="/", status_code=303)

# 로그인 폼 렌더링 엔드포인트
@app.get("/login", response_class=HTMLResponse)
async def get_login_form(request: Request):
    # login.html 템플릿을 렌더링하여 반환
    return templates.TemplateResponse("login.html", {"request": request})

# 로그인 엔드포인트 정의 (Form 데이터로 수집)
@app.post("/login")
def login(
    name: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    # 사용자 이름으로 사용자 조회
    db_user = db.query(models.User).filter(models.User.name == name).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다.")

    # 비밀번호 검증
    if not verify_password(password, db_user.password):
        raise HTTPException(status_code=401, detail="비밀번호가 일치하지 않습니다.")

    # 로그인 성공 시 사용자별 메인 화면으로 리다이렉트
    return RedirectResponse(url=f"/main/{db_user.id}", status_code=303)

# 로그인 후 사용자별 메인 화면 렌더링 엔드포인트
@app.get("/main/{user_id}", response_class=HTMLResponse)
async def get_main_page(user_id: int, request: Request, db: Session = Depends(get_db)):
    # 사용자 ID로 사용자 조회
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다.")
    # user_main.html 템플릿을 렌더링하여 반환
    return templates.TemplateResponse("user_main.html", {"request": request, "user": user})

# 회원 목록 엔드포인트 정의
@app.get("/users", response_class=HTMLResponse)
async def get_users_list(request: Request, db: Session = Depends(get_db)):
    # 모든 사용자 조회
    users = db.query(models.User).all()
    # users.html 템플릿을 렌더링하여 반환
    return templates.TemplateResponse("users.html", {"request": request, "users": users})


# 분석 화면 가져오기기
@app.get("/analysis/{user_id}", response_class=HTMLResponse)
async def get_analysis_page(request: Request, user_id: int, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다.")
    return templates.TemplateResponse("analysis.html", {"request": request, "user_id": user_id})

# WebSocket 연결 처리 엔드포인트
@app.websocket("/ws/blink/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id:int):
    await manager.connect(websocket)

    # WebSocket 세션에서 독립적인 데이터베이스 세션 사용
    db = SessionLocal()
    session = None

    try:
        # 사용자별 세션 ID 계산
        previous_sessions = db.query(models.MeasurementSession).filter(
            models.MeasurementSession.user_id == user_id
        ).count()
        user_session_id = previous_sessions + 1

        # 새로운 측정 세션 생성
        session = models.MeasurementSession(
            user_id=user_id,
            user_session_id=user_session_id,
            start_time=datetime.now(),
            total_blinks=0,
            total_time = 0.0,
            average_blink_rate=0.0,
            focus_time =0.0,
            session_status="in-progress"
        )
        db.add(session)
        db.commit()
        db.refresh(session)
        print(f"New session started with ID: {session.user_id} - {session.user_session_id}")

        operating = False  # 동작 상태 플래그

        # WebSocket 메시지 처리 루프
        while True:
              data = await websocket.receive_text()
              if data == "Operating" and not operating:
                # Operating 메시지 수신 시 동작 상태 활성화
                operating = True
                # 실시간 데이터 전송 태스크 생성
                realtime_task = asyncio.create_task(send_realtime_data(websocket, user_id))
                print(f"User {user_id}: Operating signal received.")
                await start_video_capture(user_id)  # 비동기 함수 호출  # 깜박임 데이터 업데이트 작업 시작
                print(f"User {user_id}: start_video_capture 호출됨.")
                
              elif data == "STOP":
                operating = False
                print(f"User {user_id}: STOP signal received.")
          
                # 측정 종료 시 깜박임 평균 횟수, 종료 시간, 측정 상태 업데이트
                blink_count, elapsed_time = process_camera_data(user_id)
                # DB 업데이트
                session.total_blinks += blink_count
                session.focus_time = elapsed_time
                session.end_time = datetime.now()

                # 총 깜박임 횟수를 경과 시간(분)으로 나누어 평균 깜박임 비율(분당 깜박임 횟수)
                elapsed_minutes = max((session.end_time - session.start_time).total_seconds() / 60, 1) # 1은 1분 보다 측정 시간이 적을 때 1분으로 설정
                
                session.total_time = round((session.end_time - session.start_time).total_seconds(),1)
                session.average_blink_rate = round(session.total_blinks / elapsed_minutes,1)

                # 세션 상태 완료로 변경
                session.session_status = "completed"
                print(f"User {user_id}: DB updated.")
                db.commit()
                db.refresh(session)

                stop_video_capture(user_id) # 영상 캡처 종료
                print(f"User {user_id}: stop_video_capture 작동 완료.")

                # 종료 시 소켓 연결 종료
                await websocket.close()
                break

    except WebSocketDisconnect:
        print("WebSocket disconnected during the loop.")
        stop_video_capture(user_id)  # 연결 종료 시 강제 종료 처리 or operating = False
        print(f"User {user_id}: stop_video_capture(disconnected에서) 호출됨.")
        if session and session.session_status == "in-progress":
          session.end_time = datetime.now()
          session.session_status = "completed"

          # 강제 종료 시 평균 깜박임 횟수 업데이트
          # 총 깜박임 횟수를 경과 시간(분)으로 나누어 평균 깜박임 비율(분당 깜박임 횟수)
          elapsed_minutes = max((session.end_time - session.start_time).total_seconds() / 60, 1) # 1은 1분 보다 측정 시간이 적을 때 1분으로 설정
          session.total_time = round((session.end_time - session.start_time).total_seconds(),1)
          session.average_blink_rate = round(session.total_blinks / elapsed_minutes, 1)

          db.commit()

    except Exception as e:
        print(f"Error receiving WebSocket data: {e}")

    finally:
        # 실시간 데이터 전송 종료
        realtime_task.cancel()
        manager.disconnect(websocket)
        db.close() # 세션을 마지막에 반드시 닫아야 함
        print("WebSocket and database session closed.")

# 사용자 데이터 조회 엔드포인트
@app.get("/api/get_all_sessions/{user_id}")
def get_all_sessions(user_id: int, db: Session = Depends(get_db)):
    # 사용자의 모든 세션 데이터를 조회
    sessions = db.query(models.MeasurementSession).filter(
        models.MeasurementSession.user_id == user_id
    ).order_by(models.MeasurementSession.start_time.desc()).limit(10).all()

    if not sessions:
        raise HTTPException(status_code=404, detail="세션 데이터가 없습니다.")

    # 모든 세션 정보와 가장 최근 세션 정보 반환
    return {
        "sessions": [
            {
                "session_id": session.user_session_id,
                "total_blinks": session.total_blinks,
                "focus_time": session.focus_time,
                "start_time": session.start_time,
                "end_time": session.end_time,
                "total_time":session.total_time,
                "average_blink_rate": session.average_blink_rate
            }
            for session in sessions
        ],
        "latest_session": {
            "session_id": sessions[0].user_session_id,
            "total_blinks": sessions[0].total_blinks,
            "focus_time": sessions[0].focus_time,
            "start_time": sessions[0].start_time,
            "end_time": sessions[0].end_time,
            "total_time":sessions[0].total_time,
            "average_blink_rate": sessions[0].average_blink_rate
        }
    }


# 실시간 데이터 전송 작업
async def send_realtime_data(websocket, user_id):
    while True:
        # 상태 확인 및 데이터 전송
          with user_states_lock:
              blink_count, elapsed_time = process_camera_data(user_id)

          await websocket.send_text(json.dumps({
              "blink_count": blink_count,
              "focus_time": elapsed_time  
          }))
          await asyncio.sleep(1)