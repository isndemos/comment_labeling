import os

# Константы для путей к моделям и меткам классов
LR_MODEL_PATH = "models/lr_adv.pkl"  # получаем значение (от 0 до 3)
TFIDF_PATH = "models/tf_idf_adv.pkl"  # получаем фичи (вектор)



# Константы для путей к моделям и меткам классов NN
_BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(_BASE_DIR, "models", "LERC")  # чекпоинт обученной bert модели
