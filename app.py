from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

# ── CONFIG ──────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL = "llama-3.3-70b-versatile"

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set in environment variables")

# ────────────────────────────────────────────────────────────

app = Flask(__name__)
CORS(app)

client = Groq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = """
You are a Technology Adoption Analysis Assistant.

You ONLY answer questions related to:
- Technology adoption frameworks (TAM, DOI, UTAUT, TOE)
- Adoption barriers and drivers
- Digital transformation
- Technology trends (AI, IoT, Cloud, Blockchain)
- Industry adoption (healthcare, finance, education)
- Change management in technology

If the user message is NOT related to these topics AND NOT a greeting,
respond EXACTLY with:

"This question is outside my area of expertise. I can only help with technology adoption analysis."

If it is a greeting, respond warmly and introduce yourself.

Do not answer anything outside your domain.
"""

# ── MEMORY CONTROL ───────────────────────────────────────────
conversation_history = []
MAX_HISTORY = 10

# ── HELPER FUNCTIONS ─────────────────────────────────────────

def is_greeting(text):
    greetings = ["hi", "hello", "hey", "good morning", "good evening", "how are you"]
    return text.lower().strip() in greetings


def is_tech_adoption_related(text):
    keywords = [
        "technology adoption", "tam", "doi", "utaut", "toe",
        "digital transformation", "ai adoption", "cloud adoption",
        "iot adoption", "blockchain adoption",
        "barriers", "drivers", "innovation", "case study",
        "fintech", "healthtech", "edtech"
    ]
    text = text.lower()
    return any(keyword in text for keyword in keywords)

# ── ROUTES ───────────────────────────────────────────────────

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    user_input = data.get("message", "").strip()

    if not user_input:
        return jsonify({"error": "Empty message"}), 400

    # ✅ Greeting Handling (No API call needed)
    if is_greeting(user_input):
        return jsonify({
            "reply": "Hello! 👋 I'm your Technology Adoption Analysis Assistant. How can I help you today?"
        })

    # ✅ Topic Boundary Check (IMPORTANT 🔥)
    if not is_tech_adoption_related(user_input):
        return jsonify({
            "reply": "This question is outside my area of expertise. I can only help with technology adoption analysis."
        })

    # ✅ Store user message
    conversation_history.append({"role": "user", "content": user_input})

    # ✅ Limit history
    if len(conversation_history) > MAX_HISTORY:
        conversation_history.pop(0)

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + conversation_history,
            temperature=0.4,
            max_tokens=1024,
        )

        reply = response.choices[0].message.content

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # ✅ Store assistant reply
    conversation_history.append({"role": "assistant", "content": reply})

    return jsonify({"reply": reply})


@app.route("/reset", methods=["POST"])
def reset():
    conversation_history.clear()
    return jsonify({"status": "Conversation reset"})


# ── RUN SERVER ───────────────────────────────────────────────

if __name__ == "__main__":
    print("Server running at http://localhost:5000")
    app.run(debug=True, port=5000)