"""Measure indirect prompt-injection detection through the real /detect pipeline.

Three numbers matter, and they are reported separately because they fail in
different ways:

1. Standalone cases (eval/data/indirect_cases.jsonl): short attack payloads
   across eight families, plus hard negatives that share their vocabulary
   ("send your CV to jobs@...", "experience building AI assistants").
2. Embedded attacks: every payload is planted at the start, middle and end of
   real job descriptions, which is how the attack actually arrives. A detector
   that only works on a bare payload is not a detector.
3. Clean documents: the same real job descriptions untouched, as source
   "document". This is the false-positive rate a CV tool would live with.

Usage (from Backend/):  python src/eval/evaluate_indirect.py [--json out.json] [--min-recall 0.9]
Exit code is non-zero if a gate fails, so CI can run it.
"""
import argparse
import json
import logging
import os
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
logging.disable(logging.WARNING)

from models.message_model import MAX_LEN, Message  # noqa: E402
from services.detect_service import process_message  # noqa: E402

DATA = os.path.join(HERE, "data")


def load(name):
    with open(os.path.join(DATA, name), encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def flagged(text, source="document", expect="malicious"):
    """True when the verdict is at least `expect`. Most attacks must be blocked
    outright; a few weak signals (a cluster of zero-width characters on its
    own) are only expected to be marked suspicious."""
    r = process_message(Message(text=text[:MAX_LEN], source=source))
    ok = {"malicious"} if expect == "malicious" else {"suspicious", "malicious"}
    return r["verdict"] in ok, r


def plant(doc, payload, where):
    # Keep room for the payload so truncation to MAX_LEN never cuts it off.
    doc = doc[: MAX_LEN - len(payload) - 4]
    if where == "start":
        return payload + "\n\n" + doc
    if where == "end":
        return doc + "\n\n" + payload
    cut = doc.find("\n", len(doc) // 2)
    cut = cut if cut != -1 else len(doc) // 2
    return doc[:cut] + "\n" + payload + "\n" + doc[cut:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", help="write the report here")
    ap.add_argument("--docs", type=int, default=15, help="real documents to embed each payload into")
    ap.add_argument("--min-recall", type=float, default=0.9)
    # Hard negatives are written to sit right next to an attack, so a small
    # false-positive budget there is honest; real clean documents get none.
    ap.add_argument("--max-negative-fpr", type=float, default=0.05)
    ap.add_argument("--max-doc-fpr", type=float, default=0.0)
    args = ap.parse_args()

    cases = load("indirect_cases.jsonl")
    jds = [j["text"] for j in load("benign_jds.jsonl")]

    # 1. standalone
    tp = fn = fp = tn = 0
    misses, false_alarms = [], []
    per_signal = defaultdict(lambda: [0, 0])
    for c in cases:
        hit, r = flagged(c["text"], expect=c.get("expect", "malicious"))
        if c["label"] == "attack":
            per_signal[c["signal"]][1] += 1
            if hit:
                tp += 1
                per_signal[c["signal"]][0] += 1
            else:
                fn += 1
                misses.append(c["text"][:90])
        else:
            if hit:
                fp += 1
                false_alarms.append((c["text"][:90], [s["matched"] for s in r["scanners"] if s["flagged"]]))
            else:
                tn += 1

    # 2. embedded in real documents
    attacks = [c for c in cases if c["label"] == "attack"]
    emb_hit = emb_total = 0
    emb_by_pos = Counter()
    emb_misses = []
    for c in attacks:
        payload, hits, n = c["text"], 0, 0
        for doc in jds[: args.docs]:
            for where in ("start", "middle", "end"):
                emb_total += 1
                n += 1
                if flagged(plant(doc, payload, where), expect=c.get("expect", "malicious"))[0]:
                    emb_hit += 1
                    hits += 1
                    emb_by_pos[where] += 1
        if hits < n:
            emb_misses.append(f"{hits}/{n}  {payload[:80]!r}")

    # 3. clean real documents
    clean_fp = [i for i, d in enumerate(jds) if flagged(d)[0]]

    recall = tp / max(1, tp + fn)
    neg_fpr = fp / max(1, fp + tn)
    emb_recall = emb_hit / max(1, emb_total)
    doc_fpr = len(clean_fp) / max(1, len(jds))

    report = {
        "standalone": {"attacks": tp + fn, "recall": round(recall, 3),
                       "hard_negatives": fp + tn, "false_positive_rate": round(neg_fpr, 3)},
        "embedded": {"cases": emb_total, "recall": round(emb_recall, 3),
                     "by_position": {k: round(v / max(1, emb_total / 3), 3) for k, v in sorted(emb_by_pos.items())}},
        "clean_documents": {"documents": len(jds), "false_positives": len(clean_fp),
                            "false_positive_rate": round(doc_fpr, 3)},
        "per_signal_recall": {k: f"{v[0]}/{v[1]}" for k, v in sorted(per_signal.items())},
        "misses": misses,
        "embedded_misses": emb_misses,
        "false_alarms": false_alarms,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    failed = []
    if recall < args.min_recall:
        failed.append(f"standalone recall {recall:.3f} < {args.min_recall}")
    if emb_recall < args.min_recall:
        failed.append(f"embedded recall {emb_recall:.3f} < {args.min_recall}")
    if neg_fpr > args.max_negative_fpr:
        failed.append(f"hard-negative FPR {neg_fpr:.3f} > {args.max_negative_fpr}")
    if doc_fpr > args.max_doc_fpr:
        failed.append(f"clean-document FPR {doc_fpr:.3f} > {args.max_doc_fpr}")
    if failed:
        print("FAIL: " + "; ".join(failed))
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
