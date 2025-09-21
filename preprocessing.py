import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("./data/interim/data_read.csv", names=["text", "label"], header=0)
df["label"] = df["label"].astype(str).str.strip().str.lower()
df = df.dropna()
TEXT2ID = {"негативная": 0, "позитивная": 1, "мусор": 2, "нейтральная": 3}
df["label"] = df["label"].map(TEXT2ID)

ax = (
    df["label"]
    .value_counts()
    .sort_values(ascending=False)
    .plot(kind="bar", figsize=(6, 4), rot=45)
)
ax.set_xlabel("Класс")
ax.set_ylabel("Количество")
ax.set_title("Распределение классов")
plt.savefig("analyze_labels.svg")


X = df.drop(columns=["label"])
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=12, stratify=y
)

X_train.to_csv("./data/output/X_train.csv")
X_test.to_csv("./data/output/X_test.csv")
y_train.to_csv("./data/output/y_train.csv")
y_test.to_csv("./data/output/y_test.csv")
