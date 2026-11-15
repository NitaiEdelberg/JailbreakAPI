import os
import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from routes import detect_route
from rate_limit import RateLimiter


app = FastAPI()

#cors middleware
app.add_middleware(
    CORSMiddleware,
    # Public, unauthenticated API: any origin may call it. No cookies or auth
    # headers are involved, so credentials stay off (a wildcard origin plus
    # credentials would let any site make credentialed calls).
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    # allow_headers governs the REQUEST; a browser hides every response header
    # outside the CORS-safelisted six unless it is named here. Without this the
    # frontend's `response.headers.get("Retry-After")` on a 429 was always null,
    # so the page could not tell anyone how long to wait.
    expose_headers=["Retry-After"],
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Basic per-IP rate limiting on the (relatively expensive) /detect endpoint so a
# single client can't hammer the scanners. Tune with RATE_LIMIT_PER_MIN.
_limiter = RateLimiter(int(os.getenv("RATE_LIMIT_PER_MIN", "60")))
# A batch holds up to 25 items, so it spends five units of the per-minute
# budget rather than one; otherwise batching would be a 25x bypass.
_COST = {"/detect": 1, "/detect/batch": 5}


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    cost = _COST.get(request.url.path) if request.method == "POST" else None
    if cost:
        ip = request.client.host if request.client else "unknown"
        if not _limiter.allow(ip, cost):
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Please slow down and try again shortly."},
                headers={"Retry-After": str(_limiter.retry_after(ip))},
            )
    return await call_next(request)


# Register routes
app.include_router(detect_route.router)


@app.get("/ping")
def ping():
    return {"message": "hey"}


@app.get("/health")
def health():
    """Which scanners are actually running. "degraded" means a core scanner
    (the ML model) failed to load and /detect is regex-only, so obfuscated
    attacks it would have caught can come back as safe."""
    from llm_judge import judge
    from scanners import scanners, unavailable
    return {
        "status": "degraded" if unavailable else "ok",
        "scanners": [s.name for s in scanners],
        "unavailable": unavailable,
        "llm_judge": judge.status(),
    }
