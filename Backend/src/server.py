from fastapi import FastAPI
from src.routes import detect_route
import logging
from fastapi.middleware.cors import CORSMiddleware


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

# Register routes
app.include_router(detect_route.router)

@app.get("/ping")
def ping():
    return {"message": "hey"}