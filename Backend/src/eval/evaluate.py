"""Measure detection quality on a held-out split.

Runs the pipeline against a test set the model never saw and reports
precision / recall / F1 / false-positive-rate / accuracy for:

  - RegexScanner       (rule layer; no training, so it sees the whole set fresh)
  - CustomMLScanner    (a fresh TF-IDF + Logistic Regression, trained only on
                        the train split, mirroring training/text_classifier_train.py)
  - Aggregate          (flag if EITHER layer flags — how /detect actually decides)

We train a fresh model here rather than loading our_scanner.pkl so the numbers
are a true held-out measurement (and reproducible on any scikit-learn version).

Run:
    cd Backend/src/eval && python evaluate.py
"""
import os
import sys

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Import the real regex scanner from src/.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from regex_scanner import RegexScanner  # noqa: E402

DATASET = os.path.join(os.path.dirname(__file__), "..", "training", "prompt_injection_dataset.csv")
SEED = 42


def metrics(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t and p)
    fp = sum(1 for t, p in zip(y_true, y_pred) if not t and p)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t and not p)
    tn = sum(1 for t, p in zip(y_true, y_pred) if not t and not p)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    acc = (tp + tn) / len(y_true) if y_true else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "fpr": fpr, "accuracy": acc,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def main():
    df = pd.read_csv(DATASET).dropna(subset=["text", "detected"])
    df["detected"] = df["detected"].astype(bool)
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"].tolist(), df["detected"].tolist(),
        test_size=0.2, random_state=SEED, stratify=df["detected"].tolist(),
    )

    # Train the ML layer on the train split only.
    vectorizer = TfidfVectorizer()
    Xtr = vectorizer.fit_transform(X_train)
    model = LogisticRegression(max_iter=2000)
    model.fit(Xtr, y_train)
    classes = list(model.classes_)
    mal_idx = classes.index(True) if True in classes else len(classes) - 1

    # Predict on the held-out test split.
    regex = RegexScanner()
    Xte = vectorizer.transform(X_test)
    ml_proba = model.predict_proba(Xte)[:, mal_idx]

    regex_pred = [regex.scan(t)["flagged"] for t in X_test]
    ml_pred = [p >= 0.5 for p in ml_proba]
    agg_pred = [r or m for r, m in zip(regex_pred, ml_pred)]

    n_pos = sum(y_test)
    print(f"Held-out test set: {len(y_test)} prompts ({n_pos} malicious / {len(y_test) - n_pos} benign)\n")
    header = f"{'scanner':<16}{'precision':>10}{'recall':>9}{'F1':>7}{'FPR':>8}{'accuracy':>10}"
    print(header)
    print("-" * len(header))
    for name, pred in [("RegexScanner", regex_pred), ("CustomMLScanner", ml_pred), ("Aggregate", agg_pred)]:
        m = metrics(y_test, pred)
        print(f"{name:<16}{m['precision']:>10.3f}{m['recall']:>9.3f}{m['f1']:>7.3f}{m['fpr']:>8.3f}{m['accuracy']:>10.3f}")


if __name__ == "__main__":
    main()
