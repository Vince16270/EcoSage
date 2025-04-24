"""
REST-API rondom RAGChat.

Start lokaal:
$ python -m src.api         # default: http://127.0.0.1:5000
"""

from __future__ import annotations

from flask import Flask, request, jsonify
from flask_cors import CORS

from .rag import RAGChat

chat = RAGChat()
app = Flask(__name__)
CORS(app)                          # zodat index.html op een andere poort mag praten


@app.post("/chat")
def chat_endpoint():
    """Verwacht JSON: { "message": "<user vraag>" }"""
    try:
        question: str = request.json["message"]
    except (TypeError, KeyError):
        return jsonify(error="JSON must have a 'message' field"), 400

    reply = chat.answer(question)
    return jsonify(reply=reply)


if __name__ == "__main__":  # pragma: no cover
    app.run(host="0.0.0.0", port=5000, debug=True)