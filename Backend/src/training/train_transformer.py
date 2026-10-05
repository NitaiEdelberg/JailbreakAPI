"""Fine-tune a small encoder on our corpus and export it as int8 ONNX.

This produces the model that transformer_scanner.py loads. It is deliberately
NOT part of the deploy: training needs torch (~800MB of wheels) and a few GB of
RAM, while the *output* is a ~25MB ONNX file the API can load on a modest box.

Why MiniLM and not something stronger: the whole constraint is that the artifact
has to fit next to FastAPI and scikit-learn. For reference, the obvious
off-the-shelf choice, protectai/deberta-v3-base-prompt-injection-v2, is a 738MB
ONNX file — roughly 1.5x the entire Render free instance.

    base model      params   fp32 ONNX   int8 ONNX
    MiniLM-L6         22M       ~90MB       ~25MB     <- default here
    MiniLM-L12        33M      ~130MB       ~35MB
    DeBERTa-v3-base  184M      ~738MB      ~190MB     (does not fit)

Usage:
    pip install torch transformers onnx onnxruntime optimum[onnxruntime] datasets
    python train_transformer.py --out ../models/jailbreak-minilm

Then to use it:
    ENABLE_TRANSFORMER_SCANNER=1 TRANSFORMER_MODEL_DIR=src/models/jailbreak-minilm ./start.sh

NOTE: this script has not been executed end to end — the environment it was
written in had no way to download the base model. Expect to debug it on first
run, and treat the hyper-parameters as defaults rather than tuned values.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, ".."))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SEED = 42


def load_corpus():
    """Same three CSVs, same dedup as text_classifier_train.py."""
    import pandas as pd
    from text_classifier_train import CORPORA, resolve  # single source of truth

    frames = []
    for filename, _ in CORPORA:
        path = resolve(filename)
        if path is None:
            continue
        df = pd.read_csv(path).dropna(subset=["text", "detected"])[["text", "detected"]]
        df["detected"] = df["detected"].astype(str).str.strip().str.upper().isin(["TRUE", "1"])
        df["text"] = df["text"].astype(str).str.strip()
        frames.append(df[df["text"].str.len() > 0])
    if not frames:
        sys.exit("No training data found — run harvest_datasets.py first.")

    df = pd.concat(frames, ignore_index=True)
    key = df["text"].str.lower().str.split().str.join(" ")
    return df[~key.duplicated()].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default=DEFAULT_MODEL)
    ap.add_argument("--out", default=os.path.join(SRC, "models", "jailbreak-minilm"))
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--no-quantize", action="store_true", help="skip int8 export (3-4x larger)")
    args = ap.parse_args()

    import numpy as np
    import torch
    from sklearn.metrics import f1_score, precision_score, recall_score
    from sklearn.model_selection import train_test_split
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              Trainer, TrainingArguments)

    df = load_corpus()
    print(f"corpus: {len(df)} rows ({int(df.detected.sum())} malicious)")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"].tolist(), df["detected"].astype(int).tolist(),
        test_size=0.2, random_state=SEED, stratify=df["detected"].tolist(),
    )

    tok = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=2,
        # transformer_scanner.py reads these back to find the malicious column.
        id2label={0: "BENIGN", 1: "INJECTION"},
        label2id={"BENIGN": 0, "INJECTION": 1},
    )

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, texts, labels):
            self.enc = tok(texts, truncation=True, max_length=args.max_length, padding="max_length")
            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, i):
            item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
            item["labels"] = torch.tensor(self.labels[i])
            return item

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "precision": precision_score(labels, preds, zero_division=0),
            "recall": recall_score(labels, preds, zero_division=0),
            "f1": f1_score(labels, preds, zero_division=0),
        }

    os.makedirs(args.out, exist_ok=True)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=os.path.join(args.out, "_checkpoints"),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            learning_rate=args.lr,
            eval_strategy="epoch",
            save_strategy="no",
            logging_steps=50,
            seed=SEED,
        ),
        train_dataset=Dataset(X_train, y_train),
        eval_dataset=Dataset(X_test, y_test),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    print("held-out:", trainer.evaluate())

    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)

    # --- ONNX export -----------------------------------------------------
    # transformer_scanner.py wants model.onnx + tokenizer.json + config.json in
    # one directory, which is exactly what optimum writes.
    from optimum.onnxruntime import ORTModelForSequenceClassification
    ort_model = ORTModelForSequenceClassification.from_pretrained(args.out, export=True)
    ort_model.save_pretrained(args.out)

    if not args.no_quantize:
        # Dynamic int8 quantization: ~4x smaller, a few tenths of a point of F1.
        # On a memory-bound free instance that trade is worth taking.
        from optimum.onnxruntime import ORTQuantizer
        from optimum.onnxruntime.configuration import AutoQuantizationConfig
        quantizer = ORTQuantizer.from_pretrained(args.out)
        quantizer.quantize(
            save_dir=args.out,
            quantization_config=AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False),
        )
        # optimum names it model_quantized.onnx; the scanner looks for model.onnx.
        q = os.path.join(args.out, "model_quantized.onnx")
        if os.path.exists(q):
            os.replace(q, os.path.join(args.out, "model.onnx"))

    onnx_path = os.path.join(args.out, "model.onnx")
    if os.path.exists(onnx_path):
        print(f"\nwrote {onnx_path} ({os.path.getsize(onnx_path)/1e6:.1f} MB)")
    with open(os.path.join(args.out, "training_meta.json"), "w") as f:
        json.dump({"base_model": args.base_model, "n_train": len(X_train),
                   "epochs": args.epochs, "max_length": args.max_length}, f, indent=2)
    print(f"\nEnable it with:\n"
          f"  ENABLE_TRANSFORMER_SCANNER=1 TRANSFORMER_MODEL_DIR={args.out} ./start.sh")


if __name__ == "__main__":
    main()
