import pickle  # convert python object hierarchy into a byte stream
import os


class CustomMLScanner:
    """Our own TF-IDF + classifier, trained on a prompt-injection dataset.

    Catches novel phrasings the regex can't enumerate. `risk` is always the
    model's P(malicious) so the score is meaningful for safe inputs too (a low
    number), not just for hits.
    """

    name = "CustomMLScanner"
    THRESHOLD = 0.5

    def __init__(self):
        model_path = os.path.join(os.path.dirname(__file__), "training/our_scanner.pkl")
        with open(model_path, "rb") as f:
            self.vectorizer, self.model = pickle.load(f)
        # Locate the "malicious" (label 1) column in predict_proba, defensively —
        # don't assume it's index 1 in case the model was fit on a different order.
        classes = list(getattr(self.model, "classes_", [0, 1]))
        self.mal_idx = classes.index(1) if 1 in classes else len(classes) - 1

    def scan(self, text: str):
        """Return a uniform verdict dict: {flagged, risk, matched}."""
        vec = self.vectorizer.transform([text or ""])
        # P(malicious): a single, consistent risk score regardless of the verdict.
        risk = round(float(self.model.predict_proba(vec)[0][self.mal_idx]), 2)
        return {"flagged": risk >= self.THRESHOLD, "risk": risk, "matched": None}
