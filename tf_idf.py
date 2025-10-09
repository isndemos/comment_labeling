import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Read data
X_train_df = pd.read_csv("./data/output/X_train.csv", index_col=0)
X_test_df = pd.read_csv("./data/output/X_test.csv", index_col=0)
y_train_sr = pd.read_csv("./data/output/y_train.csv", index_col=0)["label"]
y_test_sr = pd.read_csv("./data/output/y_test.csv", index_col=0)["label"]

# Extract text series (ensure string dtype)
X_train_text = X_train_df["text"].astype(str)
X_test_text = X_test_df["text"].astype(str)

# Vectorize: fit on train, transform both
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train_text)
X_test_tfidf = vectorizer.transform(X_test_text)

# Train classifier
lr = LogisticRegression()
lr.fit(X_train_tfidf, y_train_sr)
# make predicts
y_pred = lr.predict(X_test_tfidf)

from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)

acc = accuracy_score(y_test_sr, y_pred)
prec = precision_score(y_test_sr, y_pred, average="weighted")
rec = recall_score(y_test_sr, y_pred, average="weighted")
f1 = f1_score(y_test_sr, y_pred, average="weighted")


print("accuracy score:", acc)
print("precision score:", prec)
print("recall score:", rec)
print("f1 score:", f1)

import pickle

with open("./models/tf_idf_start.pkl", "wb") as file:
    pickle.dump(vectorizer, file=file)

with open("./models/lr_start.pkl", "wb") as file:
    pickle.dump(lr, file=file)
