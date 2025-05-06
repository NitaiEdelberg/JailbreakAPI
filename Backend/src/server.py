from fastapi import FastAPI
from routes import detect_route
import logging

app = FastAPI()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Register routes
app.include_router(detect_route.router)

@app.get("/ping")
def ping():
    return {"message": "hey"}