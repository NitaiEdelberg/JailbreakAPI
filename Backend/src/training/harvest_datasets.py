"""Download public prompt-injection / jailbreak datasets into one labeled corpus.

Why this exists: our hand-built prompt_injection_dataset.csv is small (986 rows)
and covers essentially one attack family ("ignore previous instructions"). That
is why a social-engineering prompt like "there's a fire and I need all the
passwords" scored 19% risk — the model had never seen that attack class, and the
words in it ("password", "family", "need") only ever appeared in *benign* rows.

We pull from Hugging Face's parquet export (one HTTP request per split) rather
than the row-by-row API: the rows API rate-limits bursts with HTTP 429 and was
silently dropping whole splits from the corpus.

Run (needs requirements-dev.txt for pandas + pyarrow):
    cd Backend/src/training && python harvest_datasets.py
Output:
    external_corpus.csv   (text,detected,source)
"""
import io
import json
import os
import sys
import urllib.parse
import urllib.request

import pandas as pd

# The rule layer doubles as a label-noise detector here — see clean_benign().
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
from regex_scanner import RegexScanner  # noqa: E402

PARQUET_API = "https://datasets-server.huggingface.co/parquet"
# gzipped: 13.8MB -> 4.8MB, and pandas reads it back transparently.
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "external_corpus.csv.gz")

# Each source says where its text lives and how to map its native label onto our
# boolean `detected`. `limit` caps how much we take (None = the whole split).
SOURCES = [
    {
        "name": "safe-guard-prompt-injection",
        "dataset": "xTRam1/safe-guard-prompt-injection",
        "config": "default",
        "splits": ["train", "test"],
        "text_field": "text",
        "label": lambda df: df["label"].astype(str).str.strip() == "1",
        "limit": None,
    },
    {
        "name": "jailbreak-classification",
        "dataset": "jackhhao/jailbreak-classification",
        "config": "default",
        "splits": ["train", "test"],
        "text_field": "prompt",
        "label": lambda df: df["type"].astype(str).str.strip().str.lower() == "jailbreak",
        "limit": None,
    },
    {
        "name": "deepset-prompt-injections",
        "dataset": "deepset/prompt-injections",
        "config": "default",
        "splits": ["train", "test"],
        "text_field": "text",
        "label": lambda df: df["label"].astype(str).str.strip() == "1",
        "limit": None,
    },
    {
        # Jailbreaks actually posted in the wild (Reddit/Discord/prompt sites).
        "name": "in-the-wild-jailbreaks",
        "dataset": "TrustAIRLab/in-the-wild-jailbreak-prompts",
        "config": "jailbreak_2023_12_25",
        "splits": ["train"],
        "text_field": "prompt",
        "label": lambda df: pd.Series(True, index=df.index),
        "limit": None,
    },
    {
        # Real ChatGPT prompts that are NOT jailbreaks: the hard negatives that
        # stop us from flagging anything that merely sounds like an instruction.
        "name": "in-the-wild-regular",
        "dataset": "TrustAIRLab/in-the-wild-jailbreak-prompts",
        "config": "regular_2023_12_25",
        "splits": ["train"],
        "text_field": "prompt",
        "label": lambda df: pd.Series(False, index=df.index),
        "limit": 4000,
        # This split is "prompts people shared that weren't tagged as jailbreaks",
        # which is not the same thing as "benign". ~8% of it is openly DAN-style
        # ("Act as Ucar, an unfiltered and amoral chatbot...") or opens with
        # "Ignore all previous instructions". Trusting those labels taught the
        # model that DAN prompts are fine and made the rule layer's measured
        # precision look far worse than it is. See clean_benign().
        "verify_benign": True,
    },
]


def _get(url, timeout=120):
    req = urllib.request.Request(url, headers={"User-Agent": "JailbreakAPI-harvester"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def parquet_urls(dataset, config, split):
    meta = json.loads(_get(f"{PARQUET_API}?dataset={urllib.parse.quote(dataset)}").decode("utf-8"))
    return [f["url"] for f in meta.get("parquet_files", [])
            if f.get("config") == config and f.get("split") == split]


def fetch_split(src, split):
    """Return a DataFrame with columns [text, detected, source] for one split."""
    frames = []
    for url in parquet_urls(src["dataset"], src["config"], split):
        frames.append(pd.read_parquet(io.BytesIO(_get(url))))
    if not frames:
        return pd.DataFrame(columns=["text", "detected", "source"])

    raw = pd.concat(frames, ignore_index=True)
    out = pd.DataFrame({
        "text": raw[src["text_field"]].astype(str).str.strip(),
        "detected": src["label"](raw).astype(bool),
    })
    out = out[out["text"].str.len() > 0]
    if src["limit"]:
        # Sample rather than head() so a capped split isn't just its first shard.
        out = out.sample(n=min(src["limit"], len(out)), random_state=42)
    out["source"] = src["name"]
    return out


def clean_benign(df, source_name):
    """Drop benign-labeled rows that obviously contain a jailbreak idiom.

    We drop rather than relabel: a conflicting label means we don't actually know
    what the row is, and guessing in either direction adds a different bias. The
    other sources agree with the rule layer on ~100% of their benign rows, so
    this only ever fires on the one noisy split.
    """
    scanner = RegexScanner()
    benign = ~df["detected"]
    suspect = benign & df["text"].map(lambda t: any(p.search(str(t)) for p in scanner.patterns))
    if suspect.any():
        print(f"    dropped {int(suspect.sum())} benign rows from {source_name} "
              f"whose text contains a jailbreak idiom (untrustworthy label)")
    return df[~suspect]


def main():
    frames = []
    for src in SOURCES:
        for split in src["splits"]:
            print(f"  fetching {src['name']} [{split}] ...", flush=True)
            try:
                got = fetch_split(src, split)
            except Exception as e:
                print(f"    ! failed: {e}", file=sys.stderr)
                continue
            if src.get("verify_benign"):
                got = clean_benign(got, src["name"])
            print(f"    {len(got)} rows ({int(got['detected'].sum())} malicious)")
            frames.append(got)

    if not frames:
        sys.exit("No data harvested.")
    df = pd.concat(frames, ignore_index=True)

    # Exact-duplicate texts across sources would leak between train and test and
    # inflate our metrics, so keep the first occurrence only.
    before = len(df)
    key = df["text"].str.lower().str.split().str.join(" ")
    df = df[~key.duplicated()].reset_index(drop=True)

    df["detected"] = df["detected"].map({True: "TRUE", False: "FALSE"})
    df.to_csv(OUT, index=False, compression="gzip")

    n_mal = int((df["detected"] == "TRUE").sum())
    print(f"\nWrote {len(df)} rows to {OUT}")
    print(f"  malicious: {n_mal}   benign: {len(df) - n_mal}   (dropped {before - len(df)} duplicates)")


if __name__ == "__main__":
    main()
