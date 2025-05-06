import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

db_client = MongoClient(os.getenv("MONGO_URI"))
db = db_client["jailbreak_db"]

collection = db["jailbreak_attempts"]
