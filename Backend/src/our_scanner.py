import pickle # convert python object hierarchy into a byte stream
import os

class CustomMLScanner:
    def __init__(self):
        model_path = os.path.join(os.path.dirname(__file__), "training/our_scanner.pkl")
        with open(model_path, "rb") as f:
            self.vectorizer, self.model = pickle.load(f)

    def scan(self, text: str):
        vec = self.vectorizer.transform([text])
        pred = int(self.model.predict(vec)[0]) #predict by 0 'valid' or 1 'invalid'
        prob = float(self.model.predict_proba(vec)[0][pred]) # probability of the prediction being correct
        
        is_valid = not pred
        risk_score = round(prob, 2)
        
        return {
            "valid": is_valid,
            "confidence": risk_score,
            "message": "Detected by ML scanner" if not is_valid else "Safe input"
        }, is_valid, risk_score
