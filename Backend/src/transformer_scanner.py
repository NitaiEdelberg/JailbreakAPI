"""Optional semantic layer: a small fine-tuned encoder, run through ONNX Runtime.

Why this is opt-in and not on by default
----------------------------------------
The regex layer knows phrasings we enumerated; the TF-IDF layer knows words and
short phrases it saw in training. Neither generalizes to a genuinely novel way of
saying the same thing. A transformer does — that is the gap this fills.

The catch is memory, and it is the reason this file is gated behind an env var:

    protectai/deberta-v3-base-prompt-injection-v2   738 MB  (ONNX, fp32)
    llm-guard PromptInjection (same model)         ~500 MB+ resident
    MiniLM-L6 fine-tuned by us, int8 ONNX           ~25 MB   <- what fits

Render's free instance is 512 MB total, and FastAPI + scikit-learn + numpy
already use a chunk of it. Anything base-sized OOMs on boot — that is exactly
why requirements.txt has llm-guard commented out. So:

  * default (Render free): this scanner stays off, the API runs regex + TF-IDF.
  * locally or on a paid instance: set ENABLE_TRANSFORMER_SCANNER=1 and point
    TRANSFORMER_MODEL_DIR at a model directory.

Producing that model is training/train_transformer.py.

NOTE: this code path has not been benchmarked — the sandbox this was written in
could not reach the model host, so there are no accuracy numbers for it yet.
Treat the defaults as a starting point, not a tuned configuration.
"""
import logging
import os

# Truncate rather than reject long prompts: attacks often pad with filler to push
# the payload past a model's window, and the first N tokens are where the
# instruction usually sits. The rule + TF-IDF layers still see the whole text.
MAX_TOKENS = 512


class TransformerScanner:
    """Uniform scanner contract over an ONNX sequence-classification model."""

    name = "TransformerScanner"
    THRESHOLD = 0.5

    def __init__(self, model_dir=None, threshold=None):
        model_dir = model_dir or os.getenv("TRANSFORMER_MODEL_DIR")
        if not model_dir:
            raise RuntimeError("TRANSFORMER_MODEL_DIR is not set")
        if not os.path.isdir(model_dir):
            raise RuntimeError(f"model dir not found: {model_dir}")

        # Imported here, not at module scope, so merely importing this file costs
        # nothing on the free tier where the deps aren't installed.
        import numpy as np
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self._np = np
        self.threshold = float(threshold if threshold is not None
                               else os.getenv("TRANSFORMER_THRESHOLD", self.THRESHOLD))

        tok_path = os.path.join(model_dir, "tokenizer.json")
        if not os.path.exists(tok_path):
            raise RuntimeError(f"tokenizer.json missing in {model_dir}")
        self._tok = Tokenizer.from_file(tok_path)
        self._tok.enable_truncation(max_length=MAX_TOKENS)

        onnx_path = os.path.join(model_dir, "model.onnx")
        if not os.path.exists(onnx_path):
            raise RuntimeError(f"model.onnx missing in {model_dir}")
        opts = ort.SessionOptions()
        # One thread: this runs inside a web worker on a small box, and letting
        # ORT spawn a thread per core just fights uvicorn for the CPU.
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self._session = ort.InferenceSession(onnx_path, opts, providers=["CPUExecutionProvider"])
        self._input_names = {i.name for i in self._session.get_inputs()}

        # Which logit column means "malicious". Saved by our training script; a
        # third-party model may instead label it in config.json.
        self.mal_idx = self._resolve_malicious_index(model_dir)

        # Fail fast so scanners.py can degrade instead of erroring per request.
        self.scan("health check")
        logging.info("TransformerScanner ready (%s, threshold %.2f)", model_dir, self.threshold)

    @staticmethod
    def _resolve_malicious_index(model_dir):
        import json
        cfg_path = os.path.join(model_dir, "config.json")
        if os.path.exists(cfg_path):
            try:
                with open(cfg_path) as f:
                    id2label = json.load(f).get("id2label") or {}
                for idx, label in id2label.items():
                    if str(label).strip().upper() in {"INJECTION", "JAILBREAK", "MALICIOUS", "LABEL_1", "UNSAFE"}:
                        return int(idx)
            except Exception as e:
                logging.warning("Could not read id2label from config.json: %s", e)
        return 1  # conventional ordering: [benign, malicious]

    def _encode(self, text):
        enc = self._tok.encode(text)
        feed = {"input_ids": self._np.array([enc.ids], dtype=self._np.int64)}
        if "attention_mask" in self._input_names:
            feed["attention_mask"] = self._np.array([enc.attention_mask], dtype=self._np.int64)
        if "token_type_ids" in self._input_names:
            feed["token_type_ids"] = self._np.array(
                [getattr(enc, "type_ids", [0] * len(enc.ids))], dtype=self._np.int64)
        return feed

    def scan(self, text: str):
        """Return a uniform verdict dict: {flagged, risk, matched}."""
        logits = self._session.run(None, self._encode(text or ""))[0][0]
        # Softmax in a numerically stable way.
        shifted = logits - self._np.max(logits)
        probs = self._np.exp(shifted) / self._np.sum(self._np.exp(shifted))
        risk = round(float(probs[self.mal_idx]), 2)
        return {"flagged": risk >= self.threshold, "risk": risk, "matched": None}
