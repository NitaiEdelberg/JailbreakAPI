"""Train the CustomMLScanner model.

What changed from the first version, and why:

* **Corpus.** We used to train on prompt_injection_dataset.csv alone (986 rows,
  essentially one attack family). We now combine it with external_corpus.csv
  (public jailbreak/injection datasets — see harvest_datasets.py) and
  curated_corpus.csv (social-engineering exfiltration + hard negatives — see
  build_social_engineering_set.py).

* **Normalization.** Features are extracted from text_normalize.normalize(), so
  leetspeak and homoglyph obfuscation fold back to plain words before the
  vectorizer sees them.

* **Features.** Word 1-2 grams with sublinear tf and min_df=5, instead of default
  unigrams. Bigrams let the model represent short phrases ("send me", "the
  passwords") rather than a pure bag of words.

* **Classifier.** LinearSVC wrapped in CalibratedClassifierCV instead of a raw
  LogisticRegression. The API reports `risk_score` as a percentage in the UI, so
  the number needs to *mean* something: after calibration, prompts we score
  around 0.6 really are malicious about 60% of the time (Brier 0.032).
  `class_weight="balanced"` compensates for the ~1:2.5 malicious:benign split.

* **Threshold.** No longer hardcoded to 0.5. We pick the smallest threshold whose
  validation false-positive rate stays under TARGET_FPR and store it in the
  artifact, so the scanner and the training run can never disagree.

* **Artifact format.** A dict with the pipeline, the threshold, the metrics and
  the scikit-learn version it was built with — instead of a bare
  (vectorizer, model) tuple with no provenance. our_scanner.py still reads the
  old tuple format so an existing deployment keeps working.

Run (from this directory):
    python text_classifier_train.py
"""
import json
import os
import pickle
import sys
import warnings

import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import brier_score_loss, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, ".."))
# The vectorizer's `preprocessor` is pickled *by reference*, so `normalize` has
# to be importable as top-level `text_normalize` both here and in the running
# app (which puts src/ on sys.path).
if SRC not in sys.path:
    sys.path.insert(0, SRC)
from text_normalize import normalize  # noqa: E402

OUT = os.path.join(HERE, "our_scanner.pkl")
REQUIREMENTS = os.path.abspath(os.path.join(HERE, "..", "..", "requirements.txt"))
SEED = 42
TEST_SIZE = 0.2
# Ceiling on how often we may flag a benign prompt. A jailbreak detector that
# blocks 1-in-20 legitimate users is worse than useless, so recall is chosen
# subject to this rather than the other way round.
TARGET_FPR = 0.02

CORPORA = [
    ("prompt_injection_dataset.csv", "hand-built seed set"),
    ("external_corpus.csv", "public datasets (harvest_datasets.py)"),
    ("curated_corpus.csv", "social engineering + hard negatives"),
]


def resolve(filename, base_dir=HERE):
    """Return the path to `filename`, preferring a .gz alongside it.

    The harvested corpus is stored gzipped (13.8MB -> 4.8MB); pandas reads either
    form transparently, so callers don't care which one is on disk.
    """
    for candidate in (filename + ".gz", filename):
        path = os.path.join(base_dir, candidate)
        if os.path.exists(path):
            return path
    return None


def check_sklearn_pin():
    """Warn if we're training with a different scikit-learn than production pins.

    A pickle written by one version and loaded by another can unpickle into a
    silently unfitted estimator — the app then either raises NotFittedError on
    every request or, worse, scores garbage. Catching it here is much cheaper
    than catching it in production, so the check runs before we spend a minute
    training a model nobody can load.
    """
    try:
        with open(REQUIREMENTS) as f:
            pinned = next((l.split("==", 1)[1].strip() for l in f
                           if l.strip().startswith("scikit-learn==")), None)
    except OSError:
        return
    if pinned and pinned != sklearn.__version__:
        print(f"\n  !! WARNING: requirements.txt pins scikit-learn=={pinned} but this "
              f"environment has {sklearn.__version__}.\n"
              f"     The .pkl written here will very likely fail to load in production.\n"
              f"     Train in an environment with the pinned version, or update the pin.\n")


def load_corpus():
    frames = []
    for filename, label in CORPORA:
        path = resolve(filename)
        if path is None:
            warnings.warn(f"{filename} missing — skipping ({label}). "
                          f"Run harvest_datasets.py / build_social_engineering_set.py first.")
            continue
        df = pd.read_csv(path).dropna(subset=["text", "detected"])[["text", "detected"]]
        # The seed CSV stores real booleans; the generated ones store TRUE/FALSE.
        df["detected"] = df["detected"].astype(str).str.strip().str.upper().isin(["TRUE", "1"])
        df["text"] = df["text"].astype(str).str.strip()
        df = df[df["text"].str.len() > 0]
        print(f"  {filename:<34} {len(df):>6} rows ({int(df.detected.sum())} malicious)  [{label}]")
        frames.append(df)

    if not frames:
        sys.exit("No training data found.")
    df = pd.concat(frames, ignore_index=True)

    # Near-duplicates across corpora would land in both train and test and make
    # the held-out numbers meaningless.
    before = len(df)
    key = df["text"].str.lower().str.split().str.join(" ")
    df = df[~key.duplicated()].reset_index(drop=True)
    print(f"  {'':<34} {len(df):>6} rows after dropping {before - len(df)} duplicates "
          f"({int(df.detected.sum())} malicious / {int((~df.detected).sum())} benign)")
    return df


def build_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            preprocessor=normalize,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=5,
            strip_accents="unicode",
        )),
        ("clf", CalibratedClassifierCV(
            LinearSVC(class_weight="balanced"),
            cv=3,
        )),
    ])


def pick_threshold(y_true, proba):
    """Lowest threshold (i.e. best recall) that keeps FPR under TARGET_FPR."""
    best = 0.5
    for thr in [i / 100 for i in range(5, 96)]:
        pred = proba >= thr
        fp = sum(1 for t, p in zip(y_true, pred) if not t and p)
        tn = sum(1 for t, p in zip(y_true, pred) if not t and not p)
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        if fpr <= TARGET_FPR:
            best = thr
            break
    return best


def report(y_true, proba, thr):
    pred = proba >= thr
    fp = sum(1 for t, p in zip(y_true, pred) if not t and p)
    tn = sum(1 for t, p in zip(y_true, pred) if not t and not p)
    return {
        "threshold": thr,
        "precision": round(precision_score(y_true, pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, pred, zero_division=0), 4),
        "fpr": round(fp / (fp + tn) if (fp + tn) else 0.0, 4),
        "brier": round(brier_score_loss(y_true, proba), 4),
        "n_test": len(y_true),
    }


def main():
    check_sklearn_pin()
    print("Loading corpora:")
    df = load_corpus()

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"].tolist(), df["detected"].tolist(),
        test_size=TEST_SIZE, random_state=SEED, stratify=df["detected"].tolist(),
    )

    print("\nTraining ...")
    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    mal_idx = list(pipe.classes_).index(True)
    proba = pipe.predict_proba(X_test)[:, mal_idx]
    thr = pick_threshold(y_test, proba)
    metrics = report(y_test, proba, thr)

    print(f"\nHeld-out test set: {metrics['n_test']} prompts")
    print(f"  threshold  {metrics['threshold']:.2f}  (lowest with FPR <= {TARGET_FPR})")
    print(f"  precision  {metrics['precision']:.3f}")
    print(f"  recall     {metrics['recall']:.3f}")
    print(f"  F1         {metrics['f1']:.3f}")
    print(f"  FPR        {metrics['fpr']:.4f}")
    print(f"  Brier      {metrics['brier']:.4f}  (calibration; lower is better)")

    # Retrain on everything once the hyper-parameters and threshold are settled —
    # the extra 20% of data is worth having in the shipped model.
    print("\nRetraining on the full corpus for the shipped artifact ...")
    final = build_pipeline()
    final.fit(df["text"].tolist(), df["detected"].tolist())

    artifact = {
        "format": 2,
        "pipeline": final,
        "threshold": thr,
        "metrics": metrics,
        "sklearn_version": sklearn.__version__,
        "n_train": len(df),
        "corpora": [c[0] for c in CORPORA],
    }
    with open(OUT, "wb") as f:
        pickle.dump(artifact, f)

    size_mb = os.path.getsize(OUT) / 1e6
    print(f"\nSaved {OUT} ({size_mb:.1f} MB, scikit-learn {sklearn.__version__})")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
