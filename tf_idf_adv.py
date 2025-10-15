import pickle

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

# Read data
X_train_df = pd.read_csv("./data/output/X_train.csv", index_col=0)
X_test_df = pd.read_csv("./data/output/X_test.csv", index_col=0)
y_train_sr = pd.read_csv("./data/output/y_train.csv", index_col=0)["label"]
y_test_sr = pd.read_csv("./data/output/y_test.csv", index_col=0)["label"]

# Extract text series (ensure string dtype)
X_train_text = X_train_df["text"].astype(str)
X_test_text = X_test_df["text"].astype(str)


pipeline = Pipeline(
    [("tfidf", TfidfVectorizer()), ("clf", LogisticRegression(max_iter=1000))]
)
param_grid = {
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "clf__C": [0.01, 0.1, 1, 10],
    "clf__penalty": ["l2"],
    "clf__solver": ["lbfgs"],
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring="f1_weighted", n_jobs=-1)
grid_search.fit(X_train_text, y_train_sr)


y_pred = grid_search.predict(X_test_text)

print("Accuracy score:", accuracy_score(y_test_sr, y_pred))
print("Precision score:", precision_score(y_test_sr, y_pred, average="weighted"))
print("Recall score:", recall_score(y_test_sr, y_pred, average="weighted"))
print("F1 score:", f1_score(y_test_sr, y_pred, average="weighted"))

# Сохраним модели
with open("./models/tf_idf_adv.pkl", "wb") as file:
    pickle.dump(grid_search.best_estimator_.named_steps["tfidf"], file)

with open("./models/lr_adv.pkl", "wb") as file:
    pickle.dump(grid_search.best_estimator_.named_steps["clf"], file)
