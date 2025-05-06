from fastapi import APIRouter
from models.message_model import Message
from services.detect_service import process_message

router = APIRouter()
@router.post("/detect")
def detect_jailbreak(message: Message):
    return process_message(message)


