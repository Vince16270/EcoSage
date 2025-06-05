"""
Flask service that serves both the RAG endpoint (/chat) and the frontend (/).
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

# RAG endpoint 
@app.post("/chat")
def chat_endpoint():
    data = request.get_json(force=True)

    message = data.get("message")
    if not message:
        return jsonify(error="JSON moet een 'message' bevatten"), 400

    lang = data.get("lang", "en")

    try:
        reply = chat.answer(message, lang=lang)
        return jsonify(reply=reply)
    except Exception as e:
        # Print de volledige stack trace in de Flask-console
        import traceback
        traceback.print_exc()
        # Stuur een leesbare foutmelding terug in JSON
        err_msg = f"Server error tijdens chat.answer: {e!r}"
        return jsonify(error=err_msg), 500

@app.route("/")
def root():
    return send_from_directory(FRONT_DIR, "index.html")

@app.route("/<path:path>")
def static_proxy(path):
    return send_from_directory(FRONT_DIR, path)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)