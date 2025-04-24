"""
Flask-service die zowel de RAG-endpoint (/chat) ALS de statische
frontend (/) serveert.  Start met:   python -m src.api
"""

from __future__ import annotations

from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from .rag import RAGChat

FRONT_DIR = Path(__file__).resolve().parent.parent / "frontend"

app = Flask(
    __name__,
    static_folder=str(FRONT_DIR),    
    static_url_path="",               
)
CORS(app)

chat = RAGChat()
@app.post("/chat")
def chat_endpoint():
    question = request.json.get("message")
    if not question:
        return jsonify(error="JSON moet 'message' bevatten"), 400
    return jsonify(reply=chat.answer(question))

@app.route("/")
def root():
    """Stuur index.html mee als iemand naar / gaat."""
    return send_from_directory(FRONT_DIR, "index.html")


@app.route("/<path:path>")
def static_proxy(path):
    return send_from_directory(FRONT_DIR, path)


if __name__ == "__main__":    
    app.run(host="0.0.0.0", port=5000, debug=True)