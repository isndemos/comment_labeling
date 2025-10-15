import logging
import pickle

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from sqlalchemy.exc import SQLAlchemyError

from database import Prediction, SessionLocal, init_db
from utils import load_models, load_models_NN

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logging.info("Starting FastAPI application")


# Создание FastAPI приложения
app = FastAPI()
ID2TEXT = {0: "негативная", 1: "позитивная", 2: "мусор", 3: "нейтральная"}
# Инициализация базы данных при старте
@app.on_event("startup")
def on_startup():
    try:
        init_db()
        logging.info("Database initialized")
        # Load and cache model/tokenizer in app state
        model, tokenizer = load_models_NN()
        app.state.model = model
        app.state.tokenizer = tokenizer
        logging.info("Model and tokenizer attached to app state")
    except Exception as e:
        # Логируем, но даем приложению стартовать для диагностики
        logging.error(f"Ошибка инициализации БД: {e}")



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




@app.post("/predict_NN/")
async def predict(request: TextRequest):
    logging.info(f"Received text for prediction: {request.text}")
    text = request.text
    model = app.state.model
    tokenizer = app.state.tokenizer
    features = tokenizer(
        text, truncation=True, padding=True, max_length=512, return_tensors="pt"
    )
    logging.info("Making prediction...")
    with torch.no_grad():
        outputs = model(**features)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

    predicted_class = int(torch.argmax(predictions, dim=-1).item())
    predicted_class_text = ID2TEXT[predicted_class]
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