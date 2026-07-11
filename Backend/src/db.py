import os
import logging
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

# Attempt logging is optional. A missing/paused MongoDB must not crash boot:
# with a mongodb+srv:// URI, MongoClient() resolves the SRV record eagerly and
# raises if the cluster/DNS is gone. Guard it so the API still runs (detection
# works, logging is simply disabled). detect_service tolerates collection=None.
collection = None
mongo_uri = os.getenv("MONGO_URI")
if mongo_uri:
    try:
        db_client = MongoClient(mongo_uri, serverSelectionTimeoutMS=3000)
        collection = db_client["jailbreak_db"]["jailbreak_attempts"]
        logging.info("Connected MongoDB for attempt logging")
    except Exception as e:
        logging.warning(f"MongoDB unavailable, attempt logging disabled: {e}")
else:
    logging.warning("MONGO_URI not set, attempt logging disabled")
