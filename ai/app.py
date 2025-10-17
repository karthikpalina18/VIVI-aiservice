from flask import Flask, request, jsonify
from flask_cors import CORS
from gpt4all import GPT4All
import os
from waitress import serve
import threading

app = Flask(__name__)
# allow specific origin or use CORS(app) for all origins (dev only)
CORS(app, resources={r"/chat": {"origins": "*"}})

MODEL_FILENAME = "mistral-7b-openorca.Q4_0.gguf"

def ensure_model():
    model_path = os.path.join(os.getcwd(), MODEL_FILENAME)
    if os.path.exists(model_path):
        return model_path
    # If you want auto download, instantiate GPT4All with allow_download=True
    model = GPT4All(model_name=MODEL_FILENAME, model_path=os.getcwd(), allow_download=True, verbose=True)
    return model_path

model_path = ensure_model()

# load model once at startup
model = GPT4All(model_name=MODEL_FILENAME, model_path=os.getcwd(), allow_download=False)

# protect access to the model from concurrent threads
model_lock = threading.Lock()

@app.route("/")
def home():
    return jsonify({"message": "Local GPT4All API is running!"})

@app.route("/chat", methods=["POST"])
def chat():
    try:
        user_input = request.json.get("message", "")
        if not user_input:
            return jsonify({"error": "No message provided"}), 400

        with model_lock:
            # single-threaded access to native model
            with model.chat_session():
                response = model.generate(user_input, max_tokens=200)

        return jsonify({"response": response})
    except Exception as e:
        app.logger.exception("Error during generation")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    serve(app, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
