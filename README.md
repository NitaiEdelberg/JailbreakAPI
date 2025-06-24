# JailbreakAPI

Modular API for detecting and blocking prompt injection attacks on chatbots.

* **Multi layered Detection:** Combines open-source LLM-Guard, and a trained ML model (scikit-learn).
* **REST API:** `/detect` endpoint (POST) – returns 200 if safe, 403 if prompt is flagged.
* **MongoDB Logging:** All requests and results are logged for review and model updates.
* **Easy Integration:** Works as a backend filter for any chatbot or LLM service.

## Quick Start

1. **Clone & Install:**

   ```bash
   git clone https://github.com/NitaiEdelberg/JailbreakAPI.git
   cd JailbreakAPI
   pip install -r requirements.txt
   ```

2. **Set up `.env`:**

   ```
   MONGO_URI=your_mongodb_uri
   ```

   (Use your own MongoDB Atlas or contact us for access)

3. **Run locally:**

cd Backend/src, then type: 
   ```bash
   uvicorn server:app --host 0.0.0.0 --port 8000
   ```

4. **Check the API:**
   Use Postman or Swagger UI at `http://localhost:8000/docs` to test `/detect`.
   
   You can also enter the deployed app here:  [jailbreakAPI](https://jailbreak-api-frontend.onrender.com).

## Project Structure

* `src/` – FastAPI server, routes, scanners, ML model, database logic
* `frontend/` – Simple HTML+JS prompt tester

## Frontend

Open `frontend/index.html` in your browser for a quick UI test.

---

*For more info contact us.*
