import logging
import os
import pickle  # convert python object hierarchy into a byte stream

import sklearn


class CustomMLScanner:
    """Our own TF-IDF + calibrated linear classifier, trained on a mix of a
    hand-built seed set, public jailbreak datasets and a curated
    social-engineering set (see training/text_classifier_train.py).

    Catches novel phrasings the regex can't enumerate. `risk` is always the
    model's calibrated P(malicious), so the score is meaningful for safe inputs
    too (a low number), not just for hits — which matters because the UI shows it
    to the user as a percentage.
    """

    name = "CustomMLScanner"
    # Fallback only. The real threshold is chosen at training time (to hold the
    # false-positive rate under a target) and stored in the artifact, so the
    # model and its decision boundary can never drift apart.
    THRESHOLD = 0.5

    def __init__(self, model_path=None):
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), "training/our_scanner.pkl")
        with open(model_path, "rb") as f:
            artifact = pickle.load(f)

        self.threshold = self.THRESHOLD
        self.metrics = {}
        self.trained_with = None

        if isinstance(artifact, dict) and "pipeline" in artifact:
            # Current format: a full pipeline plus its provenance.
            self._pipeline = artifact["pipeline"]
            self.threshold = float(artifact.get("threshold", self.THRESHOLD))
            self.metrics = artifact.get("metrics", {})
            self.trained_with = artifact.get("sklearn_version")
            self._predict_proba = self._pipeline.predict_proba
            classes = list(getattr(self._pipeline, "classes_", [False, True]))
        else:
            # Legacy format: a bare (vectorizer, model) tuple. Kept so an older
            # our_scanner.pkl still loads without a retrain.
            vectorizer, model = artifact
            self._pipeline = None
            self._predict_proba = lambda texts: model.predict_proba(vectorizer.transform(texts))
            classes = list(getattr(model, "classes_", [0, 1]))

        # Locate the "malicious" column in predict_proba defensively — the label
        # is True in the current corpus and was 1 in the legacy pickle, and we
        # shouldn't assume either is at index 1.
        self.mal_idx = next(
            (i for i, c in enumerate(classes) if c in (1, True, "1", "TRUE")),
            len(classes) - 1,
        )

        # A pickle written by a different scikit-learn can unpickle into a
        # silently broken estimator (an unfitted TfidfVectorizer that raises
        # NotFittedError on the first request, or worse, one that returns
        # garbage). Say so loudly at boot rather than at 3am.
        if self.trained_with and self.trained_with != sklearn.__version__:
            logging.warning(
                "our_scanner.pkl was trained with scikit-learn %s but this "
                "environment has %s. Re-run training/text_classifier_train.py "
                "with the pinned version to avoid silently wrong scores.",
                self.trained_with, sklearn.__version__,
            )

        # Fail fast if the artifact is unusable, so scanners.py can degrade to
        # regex-only instead of returning nonsense on every request.
        probe = self._predict_proba(["health check"])
        if probe is None or len(probe) != 1:
            raise RuntimeError("model produced no prediction for a probe input")

    def scan(self, text: str):
        """Return a uniform verdict dict: {flagged, risk, matched}."""
        # P(malicious): a single, consistent risk score regardless of the verdict.
        risk = round(float(self._predict_proba([text or ""])[0][self.mal_idx]), 2)
        return {"flagged": risk >= self.threshold, "risk": risk, "matched": None}
