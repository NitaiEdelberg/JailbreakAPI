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
    allow_origins=["*"],  # For development only – allow all
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Basic per-IP rate limiting on the (relatively expensive) /detect endpoint so a
# single client can't hammer the scanners. Tune with RATE_LIMIT_PER_MIN.
_limiter = RateLimiter(int(os.getenv("RATE_LIMIT_PER_MIN", "60")))


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if request.url.path == "/detect" and request.method == "POST":
        ip = request.client.host if request.client else "unknown"
        if not _limiter.allow(ip):
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
