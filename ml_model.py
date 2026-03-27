import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from pathlib import Path

MODEL_DIR = Path(__file__).parent / "models"

FAKE = [
    "SHOCKING: You won't believe what happened next!",
    "Doctors hate this one simple trick!",
    "100% guaranteed cure for all diseases!",
    "Scientists confirm Earth is flat!",
    "Government hiding alien contact!",
    "Miracle pill - lose 50 pounds overnight!",
    "URGENT: Share before deleted!",
    "Big Pharma hates this cure!",
    "Secret society controls governments!",
    "Vaccine causes autism!",
    "5G causes coronavirus!",
    "You won $1,000,000! Claim now!",
    "Your computer is infected! Call now!",
    "Illuminati exposed!",
    "Mind control chips confirmed!",
]

REAL = [
    "Federal Reserve announced interest rate increase.",
    "MIT scientists developed new recycling method.",
    "Unemployment rate fell to 3.8 percent.",
    "Climate research published in peer-reviewed journal.",
    "Stock market closed higher on trade optimism.",
    "City council approved infrastructure budget.",
    "Company reported quarterly earnings.",
    "Health officials recommend flu vaccinations.",
    "University announced scholarship program.",
    "New healthcare legislation proposed.",
    "Team won championship in overtime.",
    "Economists predict GDP growth.",
    "Museum hosting art exhibition.",
    "Traffic restrictions during festival.",
    "Company plans to hire employees.",
]

class FakeNewsDetector:
    def __init__(self):
        self.vectorizer = None
        self.model = None
        self.is_loaded = False
    
    def train(self):
        texts = FAKE + REAL
        labels = [1]*len(FAKE) + [0]*len(REAL)
        aug_texts, aug_labels = [], []
        for t, l in zip(texts, labels):
            aug_texts.extend([t, t.lower()])
            aug_labels.extend([l, l])
        
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words='english')
        X = self.vectorizer.fit_transform(aug_texts)
        self.model = LogisticRegression(max_iter=1000, class_weight='balanced')
        self.model.fit(X, np.array(aug_labels))
        self.is_loaded = True
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, MODEL_DIR / "model.joblib")
        joblib.dump(self.vectorizer, MODEL_DIR / "vec.joblib")
    
    def load(self):
        try:
            self.model = joblib.load(MODEL_DIR / "model.joblib")
            self.vectorizer = joblib.load(MODEL_DIR / "vec.joblib")
            self.is_loaded = True
            return True
        except:
            return False
    
    def predict(self, text):
        if not self.is_loaded and not self.load():
            self.train()
        
        fake_words = ['shocking', "won't believe", 'click here', '100%', 'guaranteed', 'miracle', 'secret', 'exposed', 'urgent', 'cure']
        real_words = ['according to', 'study', 'researchers', 'reported', 'announced', 'percent', 'officials']
        
        text_lower = text.lower()
        fake_count = sum(1 for w in fake_words if w in text_lower)
        real_count = sum(1 for w in real_words if w in text_lower)
        
        X = self.vectorizer.transform([text])
        pred = self.model.predict(X)[0]
        prob = max(self.model.predict_proba(X)[0])
        
        if pred == 1:
            prob = min(prob + fake_count * 0.05, 0.99)
        
        return {
            "prediction": "FAKE" if pred == 1 else "REAL",
            "is_fake": bool(pred == 1),
            "confidence": round(prob * 100, 1)
        }

detector = None
def get_detector():
    global detector
    if detector is None:
        detector = FakeNewsDetector()
        if not detector.load():
            detector.train()
    return detector
File 3: Save as backend/server.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ml_model import get_detector

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class Request(BaseModel):
    text: str

@app.post("/api/predict")
def predict(req: Request):
    return get_detector().predict(req.text)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)