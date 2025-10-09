import os
from datetime import datetime

from sqlalchemy import Column, DateTime, Integer, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Используем переменную окружения DATABASE_URL или дефолтное значение (SQLite)
# Абсолютный путь к файлу БД в каталоге data/app.db
_BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_DEFAULT_SQLITE_PATH = os.path.join(_BASE_DIR, "data", "app.db")
_DEFAULT_DATABASE_URL = f"sqlite:///{_DEFAULT_SQLITE_PATH}"

DATABASE_URL = os.environ.get("DATABASE_URL", _DEFAULT_DATABASE_URL)

# Параметры для SQLite (игнорируются другими драйверами)
_ENGINE_KWARGS = {}
if DATABASE_URL.startswith("sqlite"):
    os.makedirs(os.path.dirname(_DEFAULT_SQLITE_PATH), exist_ok=True)
    _ENGINE_KWARGS["connect_args"] = {"check_same_thread": False}

engine = create_engine(DATABASE_URL, **_ENGINE_KWARGS)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, index=True)
    comment = Column(String, nullable=False)
    predicted_class = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


# Создать таблицы (вызывать один раз при старте)
def init_db():
    Base.metadata.create_all(bind=engine)
