import logging
import os

import requests
import streamlit as st

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logging.info("Запуск Streamlit приложения")

# Получаем URL бэкенда из переменной окружения или используем значение по умолчанию
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")
logging.info(f"Используем URL бэкенда: {BACKEND_URL}")

st.title("Классификация текста")

text = st.text_area("Введите текст для классификации:")

if st.button("Классифицировать (tf-idf+logreg)"):
    if text.strip():
        logging.info(f"Получен текст для классификации: {text}")
        # Отправка POST запроса к FastAPI серверу
        logging.info("Отправка запроса на сервер...")
        response = requests.post(f"{BACKEND_URL}/predict/", json={"text": text})
        if response.ok:
            logging.info("Получен ответ от сервера")
            result = response.json()
            st.success(f"Класс: {result['predicted_class']}")
        else:
            logging.error(f"Ошибка при обращении к серверу: {response.status_code}")
            st.error("Ошибка при обращении к серверу.")
    else:
        st.warning("Пожалуйста, введите текст.")

if st.button("Классифицировать (fine-tuned BERT)"):
    if text.strip():
        logging.info(f"Получен текст для классификации: {text}")
        # Отправка POST запроса к FastAPI серверу
        logging.info("Отправка запроса на сервер...")
        response = requests.post(f"{BACKEND_URL}/predict_NN/", json={"text": text})
        if response.ok:
            logging.info("Получен ответ от сервера")
            result = response.json()
            st.success(f"Класс: {result['predicted_class']}")
        else:
            logging.error(f"Ошибка при обращении к серверу: {response.status_code}")
            st.error("Ошибка при обращении к серверу.")
    else:
        st.warning("Пожалуйста, введите текст.")