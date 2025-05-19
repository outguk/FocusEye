from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Float
from sqlalchemy.orm import relationship
from database import Base
from datetime import datetime, timezone

# User 모델 정의 (데이터베이스에 users 테이블을 정의)
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)  # 사용자 ID (기본 키)
    name = Column(String(50), index=True)  # 사용자 이름
    email = Column(String(50), unique=True, index=True)  # 사용자 이메일 (고유값)
    password = Column(String(200))  # 해싱된 비밀번호

    # 관계 설정: User는 여러 MeasurementSession을 가질 수 있음
    measurement_sessions = relationship("MeasurementSession", back_populates="user")

class MeasurementSession(Base):
    __tablename__ = "measurement_sessions"

    session_id = Column(Integer, primary_key=True, index=True)  # 세션 ID (기본 키)
    user_id = Column(Integer, ForeignKey('users.id'))  # 사용자 ID (외래 키)
    user_session_id = Column(Integer, nullable=False)  # user_id 별로 독립적인 세션 ID
    start_time = Column(DateTime, default=datetime.now(timezone.utc))  # 세션 시작 시간
    end_time = Column(DateTime, nullable=True)  # 세션 종료 시간 (측정 완료 시 설정)
    total_time = Column(Float, default=0.0)  # 세션 측정 시간
    total_blinks = Column(Integer, default=0)  # 총 깜박임 횟수 
    average_blink_rate = Column(Float, default=0.0)  # 평균 눈 깜박임 횟수 (분당(60초당) 횟수 25로 기준)
    focus_time = Column(Float, default=0.0) # 집중 시간
    session_status = Column(String(20), default="in-progress")  # 세션 상태

    # User와의 관계 설정
    user = relationship("User", back_populates="measurement_sessions")