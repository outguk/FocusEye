from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MySQL 데이터베이스 URL 설정
DATABASE_URL = "mysql+pymysql://eyeApi_user1:ehowl464@localhost:3306/your_database"

# SQLAlchemy 엔진 생성
logger.info("Creating database engine...")
# engine = create_engine(DATABASE_URL, echo=True)
engine = create_engine(DATABASE_URL, echo=False, future=True) # 로그 줄이려면 echo=False로

# MetaData와 Base 선언
metadata = MetaData()
Base = declarative_base()

# 세션 설정
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, future=True)
# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 데이터베이스 초기화 함수
def init_db():
    logger.info("Initializing database...")
    Base.metadata.create_all(bind=engine)