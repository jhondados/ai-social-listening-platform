"""Social listening analyzer with PT-BR sentiment."""
from transformers import pipeline
from typing import List, Dict
from datetime import datetime
import re

class PTBRSentimentAnalyzer:
    def __init__(self):
        # BERTimbau fine-tuned for Portuguese sentiment
        self.sentiment = pipeline("text-classification",
            model="lxyuan/distilbert-base-multilingual-cased-sentiments-student",
            top_k=None)

    def analyze(self, text: str) -> Dict:
        text_clean = self.clean_text(text)
        if not text_clean: return {"sentiment": "neutral", "confidence": 0.5, "scores": {}}
        results = self.sentiment(text_clean[:512])[0]
        top = max(results, key=lambda x: x["score"])
        scores = {r["label"].lower(): round(r["score"], 3) for r in results}
        return {"sentiment": top["label"].lower(), "confidence": round(top["score"], 3), "scores": scores}

    def clean_text(self, text: str) -> str:
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"@\w+", "", text)
        text = re.sub(r"#(\w+)", r"\1", text)
        return text.strip()[:500]

class CrisisDetector:
    CRISIS_KEYWORDS = [
        "boicote", "fraude", "escandalo", "recall", "processo", "enganados",
        "vergonha", "absurdo", "golpe", "nao comprem", "horrivel"
    ]

    def __init__(self, sentiment_analyzer: PTBRSentimentAnalyzer):
        self.analyzer = sentiment_analyzer
        self.alert_threshold = 0.15  # 15% of posts negative

    def check_crisis(self, posts: List[Dict], brand: str) -> Dict:
        brand_posts = [p for p in posts if brand.lower() in p.get("text", "").lower()]
        if not brand_posts: return {"crisis": False}
        sentiments = [self.analyzer.analyze(p["text"])["sentiment"] for p in brand_posts[:100]]
        neg_rate = sentiments.count("negative") / max(len(sentiments), 1)
        keyword_hits = sum(1 for p in brand_posts if any(kw in p.get("text","").lower() for kw in self.CRISIS_KEYWORDS))
        crisis_score = neg_rate + (keyword_hits / max(len(brand_posts), 1)) * 0.5
        return {"crisis": crisis_score > self.alert_threshold, "crisis_score": round(crisis_score, 3),
                "negative_rate": round(neg_rate * 100, 1), "posts_analyzed": len(brand_posts),
                "alert_message": f"CRISE DETECTADA: {neg_rate:.1%} posts negativos para {brand}" if crisis_score > self.alert_threshold else None}
