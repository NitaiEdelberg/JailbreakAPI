"""A small, dependency-free client for JailbreakAPI.

    from jailbreak_guard import Guard
    guard = Guard()                                   # the public demo API
    if guard.check(user_message).is_malicious: ...
    safe_docs, quarantined = guard.filter_documents(retrieved_chunks)

The one decision every caller has to make is what happens when the guard itself
is down. That is `fail_open`: False (the default) raises GuardUnavailable so you
can refuse the request; True lets the text through and marks the result
`unavailable`, for apps that would rather degrade than stop.
"""
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

DEFAULT_URL = "https://jailbreak-api-backend.onrender.com"
SOURCES = ("user", "document", "web", "email", "tool", "rag")
BATCH_MAX_ITEMS = 25
BATCH_MAX_CHARS = 100_000
TEXT_MAX_CHARS = 10_000

# transport(method, url, json_body, timeout) -> (status, parsed_body, headers)
Transport = Callable[[str, str, Optional[dict], float], Tuple[int, Any, Dict[str, str]]]


class GuardError(Exception):
    """The API rejected the request (bad input). Not retried."""


class GuardUnavailable(GuardError):
    """The API could not be reached or kept failing. With fail_open=False this is
    raised; with fail_open=True the check returns an `unavailable` result."""


@dataclass
class Result:
    verdict: str                      # "safe" | "suspicious" | "malicious" | "unavailable"
    risk_score: float = 0.0
    source: str = "user"
    id: Optional[str] = None
    scanners: List[dict] = field(default_factory=list)
    judge: Optional[dict] = None
    raw: dict = field(default_factory=dict)

    @property
    def is_malicious(self) -> bool:
        return self.verdict == "malicious"

    @property
    def is_suspicious(self) -> bool:
        return self.verdict == "suspicious"

    @property
    def is_safe(self) -> bool:
        return self.verdict == "safe"

    @property
    def reasons(self) -> List[str]:
        """Human-readable 'why', one entry per scanner that flagged."""
        out = []
        for s in self.scanners:
            if s.get("flagged"):
                bits = [s.get("scanner", "?")]
                if s.get("signal"):
                    bits.append(s["signal"].replace("_", " "))
                if s.get("matched"):
                    bits.append(repr(s["matched"]))
                out.append(": ".join(bits))
        if self.judge and self.judge.get("injection"):
            out.append("LLMJudge: " + str(self.judge.get("reason", "")))
        return out

    @classmethod
    def from_api(cls, body: dict) -> "Result":
        return cls(verdict=body.get("verdict", "safe"), risk_score=float(body.get("risk_score", 0.0)),
                   source=body.get("source", "user"), id=body.get("id"), scanners=body.get("scanners", []),
                   judge=body.get("judge"), raw=body)


def _urllib_transport(method, url, body, timeout):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json", "User-Agent": "jailbreak-guard/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode("utf-8") or "null"), dict(r.headers)
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "ignore")
        try:
            parsed = json.loads(raw)
        except ValueError:
            parsed = {"detail": raw[:300]}
        return e.code, parsed, dict(e.headers or {})


class Guard:
    def __init__(self, base_url: str = DEFAULT_URL, *, timeout: float = 70.0, fail_open: bool = False,
                 retries: int = 1, max_retry_wait: float = 10.0, transport: Optional[Transport] = None,
                 sleep: Callable[[float], None] = time.sleep):
        # 70s default: the public demo runs on a free instance that takes ~50s to
        # wake. Self-hosted, set it to a couple of seconds.
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.fail_open = fail_open
        self.retries = retries
        self.max_retry_wait = max_retry_wait
        self._transport = transport or _urllib_transport
        self._sleep = sleep

    # -- plumbing ----------------------------------------------------------------

    def _request(self, method, path, body=None):
        last = "no attempt"
        for attempt in range(self.retries + 1):
            try:
                status, parsed, headers = self._transport(method, self.base_url + path, body, self.timeout)
            except Exception as e:  # network error, timeout, DNS
                last = f"{type(e).__name__}: {e}"
            else:
                if status == 200:
                    return parsed
                if status == 422 or 400 <= status < 500 and status != 429:
                    raise GuardError(f"HTTP {status}: {parsed}")
                last = f"HTTP {status}"
                if status == 429:
                    wait = headers.get("Retry-After") or headers.get("retry-after") or "1"
                    try:
                        delay = min(float(wait), self.max_retry_wait)
                    except ValueError:
                        delay = 1.0
                    if attempt < self.retries:
                        self._sleep(delay)
                    continue
            if attempt < self.retries:
                self._sleep(min(2.0 * (attempt + 1), self.max_retry_wait))
        raise GuardUnavailable(f"JailbreakAPI unavailable after {self.retries + 1} attempts ({last})")

    def _unavailable(self, source, id=None, err=None):
        if not self.fail_open:
            raise err
        return Result(verdict="unavailable", source=source, id=id, raw={"error": str(err)})

    # -- public API ----------------------------------------------------------------

    def check(self, text: str, source: str = "user") -> Result:
        """Screen one piece of text. `source` is where it came from: "user" for
        what the person typed, or document/web/email/tool/rag for content the
        model will read (turns on the indirect-injection rules)."""
        if source not in SOURCES:
            raise ValueError(f"source must be one of {SOURCES}")
        try:
            return Result.from_api(self._request("POST", "/detect", {"text": text[:TEXT_MAX_CHARS], "source": source}))
        except GuardUnavailable as e:
            return self._unavailable(source, err=e)

    def check_many(self, items: Iterable, source: str = "document") -> List[Result]:
        """Screen many texts. `items` are strings or (id, text) pairs. Split into
        API-sized batches automatically; results come back in input order."""
        norm = []
        for i, it in enumerate(items):
            iid, text = (it if isinstance(it, tuple) else (str(i), it))
            norm.append({"id": str(iid), "text": str(text)[:TEXT_MAX_CHARS], "source": source})
        results: List[Result] = []
        for batch in _chunks(norm):
            try:
                body = self._request("POST", "/detect/batch", {"items": batch})
                results.extend(Result.from_api(r) for r in body["results"])
            except GuardUnavailable as e:
                if not self.fail_open:
                    raise
                results.extend(Result(verdict="unavailable", source=source, id=b["id"], raw={"error": str(e)})
                               for b in batch)
        return results

    def filter_documents(self, docs: Sequence[Any], source: str = "rag", text_of: Callable[[Any], str] = None,
                         drop_suspicious: bool = False):
        """For RAG and agents: screen retrieved chunks / tool results before the
        model sees them. Returns (kept, quarantined) where quarantined is a list
        of (doc, Result). Works on strings, dicts with a 'text'/'page_content'
        key, or LangChain Documents, or pass `text_of`."""
        text_of = text_of or _default_text
        results = self.check_many([(str(i), text_of(d)) for i, d in enumerate(docs)], source=source)
        kept, quarantined = [], []
        for d, r in zip(docs, results):
            bad = r.is_malicious or (drop_suspicious and r.is_suspicious)
            (quarantined.append((d, r)) if bad else kept.append(d))
        return kept, quarantined

    def health(self) -> dict:
        return self._request("GET", "/health")


def _default_text(d):
    if isinstance(d, str):
        return d
    if isinstance(d, dict):
        return d.get("text") or d.get("page_content") or d.get("content") or ""
    return getattr(d, "page_content", None) or getattr(d, "text", None) or str(d)


def _chunks(items):
    batch, chars = [], 0
    for it in items:
        n = len(it["text"])
        if batch and (len(batch) >= BATCH_MAX_ITEMS or chars + n > BATCH_MAX_CHARS):
            yield batch
            batch, chars = [], 0
        batch.append(it)
        chars += n
    if batch:
        yield batch
