from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq
import os
from dotenv import load_dotenv
load_dotenv()

# ── CONFIG ──────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # paste your key here
MODEL        = "llama-3.3-70b-versatile"
# ────────────────────────────────────────────────────────────

app    = Flask(__name__)
CORS(app)
client = Groq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = """
You are an expert Technology Adoption Analysis Assistant.

GREETING RULE: If the user sends a greeting (e.g. "hi", "hello", "hey", "good morning",
"how are you", etc.), respond warmly and professionally. Introduce yourself briefly as a
Technology Adoption Analysis Assistant and invite them to ask a question. Do NOT treat
greetings as off-topic.

You ONLY answer questions related to technology adoption, including:
- Frameworks: TAM, DOI, UTAUT, TOE
- Barriers and drivers of technology adoption
- Digital transformation strategies
- Tech adoption trends and statistics
- Case studies (cloud, AI, IoT, blockchain, etc.)
- Industry-specific adoption (healthcare, finance, education, etc.)
- Change management related to technology adoption

STRICT RULE: If the message is NOT a greeting AND NOT related to any of the above topics,
respond with exactly:
"I'm sorry, I can only assist with questions related to technology
adoption analysis. Please ask something relevant to that topic."

Never answer off-topic questions. Be professional and data-driven.
"""

conversation_history = []

@app.route("/chat", methods=["POST"])
def chat():
    data       = request.json
    user_input = data.get("message", "").strip()

    if not user_input:
        return jsonify({"error": "Empty message"}), 400

    conversation_history.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model    = MODEL,
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + conversation_history,
        temperature = 0.4,
        max_tokens  = 1024,
    )

    reply = response.choices[0].message.content
    conversation_history.append({"role": "assistant", "content": reply})

    return jsonify({"reply": reply})

@app.route("/reset", methods=["POST"])
def reset():
    conversation_history.clear()
    return jsonify({"status": "Conversation reset"})

if __name__ == "__main__":
    print("Server running at http://localhost:5000")
    app.run(debug=True, port=5000)