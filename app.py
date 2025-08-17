import os
import json
import re
from dataclasses import dataclass
from typing import List, Dict, Any

from flask import Flask, request, render_template
from dotenv import load_dotenv
from markupsafe import escape
from PIL import Image
import google.generativeai as genai

# -----------------------------
# Config
# -----------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_ID = os.getenv("GEMINI_MODEL_ID", "gemini-1.5-pro-latest")

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Gemini
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel(GEMINI_MODEL_ID)

# -----------------------------
# Data structures
# -----------------------------
@dataclass
class GeminiSMSResult:
    verdict: str
    confidence: float
    reasons: List[str]
    risky_phrases: List[str]
    urls: List[Dict[str, Any]]
    user_message: str

# -----------------------------
# Helpers
# -----------------------------
OFFICIAL_DOMAINS = [
    "airtel.in", "icicibank.com", "hdfcbank.com", "sbi.co.in",
    "axisbank.com", "paytm.com", "phonepe.com", "google.com", "amazon.in"
]

def highlight_phrases(text: str, phrases: List[str]) -> str:
    escaped = escape(text or "")
    for p in sorted(set(phrases), key=len, reverse=True):
        pattern = re.compile(rf"(?i)\b({re.escape(p)})\b")
        escaped = pattern.sub(r"<mark>\\1</mark>", escaped)
    return escaped

# -----------------------------
# Gemini Analysis (Image-based)
# -----------------------------
def call_gemini_sms_image(image_path: str) -> GeminiSMSResult:
    prompt = """
You are a cybersecurity SMS phishing detector.
Analyze the text inside this SMS screenshot image.

Guidelines:
- safe → if legitimate, no suspicious links, no sensitive info requests
- suspicious → unusual wording/links, but not clearly phishing
- phishing → malicious intent, fake links, urgent CTA, sensitive info

Respond ONLY in valid JSON with this format:
{
  "verdict": "safe" | "suspicious" | "phishing",
  "confidence": 0.0-1.0,
  "reasons": ["short reason 1", "short reason 2"],
  "risky_phrases": ["word1","word2"],
  "urls": [{"url": "https://...", "status": "safe|suspicious|phishing"}],
  "user_message": "1-2 sentence explanation"
}
"""

    try:
        img = Image.open(image_path)
        resp = gemini_model.generate_content(
            [prompt, img],
            generation_config={"response_mime_type": "application/json"}
        )
        data = json.loads(resp.text)
    except Exception as e:
        data = {
            "verdict": "unknown",
            "confidence": 0.0,
            "reasons": [f"Gemini error: {str(e)}"],
            "risky_phrases": [],
            "urls": [],
            "user_message": "Automatic analysis failed."
        }

    verdict = str(data.get("verdict", "unknown")).lower()
    if verdict not in {"safe", "suspicious", "phishing"}:
        verdict = "unknown"

    try:
        confidence = float(data.get("confidence", 0.0))
    except Exception:
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    reasons = [str(r) for r in data.get("reasons", [])][:6]
    risky_phrases = list(set(data.get("risky_phrases", [])))

    urls_list = []
    for item in data.get("urls", []):
        u = str(item.get("url", "")).strip()
        s = str(item.get("status", "")).lower()
        if s not in {"safe","suspicious","phishing"}:
            s = "suspicious"
        urls_list.append({"url": u, "status": s})

    user_message = data.get("user_message", "").strip() or "No explanation provided."

    return GeminiSMSResult(
        verdict=verdict,
        confidence=confidence,
        reasons=reasons,
        risky_phrases=risky_phrases,
        urls=urls_list,
        user_message=user_message
    )

# -----------------------------
# Routes
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    analysis, sms_text, highlighted_text, uploaded_path = None, "", "", None

    if request.method == "POST":
        file = request.files.get("image")
        if file and file.filename:
            uploaded_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(uploaded_path)

            # Run Gemini analysis directly on image
            gem = call_gemini_sms_image(uploaded_path)

            badge = {"safe": "✅ Safe", "suspicious": "⚠ Suspicious", "phishing": "❌ Phishing"}.get(gem.verdict, "❔ Unknown")

            highlighted_text = highlight_phrases("(OCR skipped – Gemini vision used)", gem.risky_phrases)

            analysis = {
                "badge": badge,
                "verdict": gem.verdict,
                "confidence": f"{gem.confidence:.2f}",
                "reasons": gem.reasons,
                "risky_phrases": gem.risky_phrases,
                "urls": gem.urls,
                "user_message": gem.user_message,
                "recommendations": [
                    "Don’t click links in suspicious messages.",
                    "Never share passwords or OTP over SMS.",
                    "Contact the organization via official channels.",
                    "Report phishing SMS to your carrier or cybercrime authority.",
                    "Delete the SMS if phishing is confirmed."
                ]
            }

    return render_template("index.html",
                           image_path=uploaded_path,
                           sms_text=sms_text,
                           highlighted_text=highlighted_text,
                           analysis=analysis)

if __name__ == "__main__":
    app.run(debug=True)
