import logging
import os
import pickle

import uvicorn
from database import Prediction, SessionLocal, init_db
from fastapi import FastAPI
from pydantic import BaseModel
from sqlalchemy.exc import SQLAlchemyError

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logging.info("Starting FastAPI application")

# Константы для путей к моделям и меткам классов
_BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LR_MODEL_PATH = os.path.join(
    _BASE_DIR, "models", "lr_adv.pkl"
)  # получаем значение (от 0 до 3)
TFIDF_PATH = os.path.join(
    _BASE_DIR, "models", "tf_idf_adv.pkl"
)  # получаем фичи (вектор)
ID2TEXT = {0: "негативная", 1: "позитивная", 2: "мусор", 3: "нейтральная"}


# Создание FastAPI приложения
app = FastAPI()


# Инициализация базы данных при старте
@app.on_event("startup")
def on_startup():
    try:
        init_db()
        logging.info("Database initialized")
    except Exception as e:
        # Логируем, но даем приложению стартовать для диагностики
        logging.error(f"Ошибка инициализации БД: {e}")


# Загрузка моделей
def load_models():
    logging.info("Loading models...")
    with open(LR_MODEL_PATH, "rb") as f:
        lr_model = pickle.load(f)
    with open(TFIDF_PATH, "rb") as f:
        tfidf = pickle.load(f)
    logging.info("Models loaded successfully")
    return lr_model, tfidf


lr_model, tfidf = load_models()


class TextRequest(BaseModel):
    text: str


@app.post("/predict/")
async def predict(request: TextRequest):
    logging.info(f"Received text for prediction: {request.text}")
    text = request.text
    features = tfidf.transform([text])
    logging.info("Making prediction...")
    predicted_class = lr_model.predict(features)[0]
    predicted_class_text = ID2TEXT[int(predicted_class)]
    logging.info(f"Predicted class: {predicted_class}")

    # Сохраняем в базу данных
    db = SessionLocal()
    try:
        db_obj = Prediction(comment=text, predicted_class=predicted_class_text)
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
    except SQLAlchemyError as e:
        db.rollback()
        logging.error(f"Ошибка при сохранении в БД: {e}")
    finally:
        db.close()

    return {"predicted_class": predicted_class_text}


@app.get("/")
async def root():
    return {
        "message": "Welcome to the Emotion Classification API. Use /predict/ to classify text."
    }


@app.get("/hello")
async def root():
    return {
        "message": "Hello, world! This is a simple FastAPI application for text classification."
    }


# еще один тест
@app.get("/v1/hello")
async def root():
    return {"message": "Hello from v1, world!."}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
