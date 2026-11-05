"""Train a classifier on data.csv -> model.pkl. Prints accuracy for your results slide."""
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("data.csv", header=None)
X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values
print("Samples per sign:\n", pd.Series(y).value_counts())

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(Xtr, ytr)

pred = model.predict(Xte)
print(f"\nAccuracy: {accuracy_score(yte, pred) * 100:.2f}%")
print(classification_report(yte, pred))

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
print("Saved model.pkl")
