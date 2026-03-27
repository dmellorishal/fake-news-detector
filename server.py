from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from GoogleNews import GoogleNews
from ml_model import get_detector
import hashlib
from datetime import datetime

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Source credibility scores
SOURCE_SCORES = {
    "reuters": 95, "bbc": 90, "cnn": 78, "fox news": 65,
    "the new york times": 88, "washington post": 87,
    "the guardian": 86, "abc news": 82, "nbc news": 82,
    "associated press": 95, "npr": 88, "usa today": 75,
    "daily mail": 45, "buzzfeed": 60, "breitbart": 35,
}

def get_credibility(source):
    source_lower = source.lower()
    for name, score in SOURCE_SCORES.items():
        if name in source_lower:
            return score
    return 50

class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=10)

@app.get("/api/health")
def health():
    return {"status": "healthy"}

@app.get("/api/news/search")
def search_news(q: str = Query(..., min_length=2), max_results: int = 10):
    try:
        googlenews = GoogleNews(lang='en', period='7d')
        googlenews.get_news(q)
        results = googlenews.results()[:max_results]
        
        articles = []
        for r in results:
            article_id = hashlib.md5(f"{r.get('title','')}{r.get('link','')}".encode()).hexdigest()[:10]
            source = r.get('media', 'Unknown')
            articles.append({
                "id": article_id,
                "title": r.get('title', 'No title'),
                "source": source,
                "url": r.get('link', '#'),
                "description": r.get('desc', ''),
                "date": r.get('date', ''),
                "credibility_score": get_credibility(source)
            })
        
        return {"success": True, "query": q, "articles": articles}
    except Exception as e:
        return {"success": False, "query": q, "articles": [], "error": str(e)}

@app.post("/api/predict")
def predict(request: PredictionRequest):
    try:
        result = get_detector().predict(request.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)