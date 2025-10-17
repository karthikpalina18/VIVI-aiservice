from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
from gpt4all import GPT4All
import os
from waitress import serve

app = Flask(__name__)

# Enable CORS for all routes
CORS(app, resources={r"/*": {"origins": "*"}})  # You can specify origins instead of '*'

# Using a different model that's known to be available
MODEL_FILENAME = "mistral-7b-openorca.Q4_0.gguf"

def ensure_model():
    """Download model using GPT4All's built-in download capability."""
    model_path = os.path.join(os.getcwd(), MODEL_FILENAME)
    
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"✅ Found model file: {MODEL_FILENAME} ({size_mb:.0f}MB)")
        return model_path
    
    print(f"🔄 Downloading model {MODEL_FILENAME}...")
    try:
        # Let GPT4All handle the download
        model = GPT4All(
            model_name=MODEL_FILENAME,
            model_path=os.getcwd(),
            allow_download=True,  # Allow automatic download
            verbose=True
        )
        print("✅ Download complete.")
        return model_path
    except Exception as e:
        print(f"❌ Error: {e}")
        raise

# Ensure the model is downloaded before loading
model_path = ensure_model()

# Load GPT4All (already downloaded above)
model = GPT4All(
    model_name=MODEL_FILENAME,
    model_path=os.getcwd(),
    allow_download=False
)

@app.route("/")
def home():
    return jsonify({"message": "Local GPT4All API is running!"})

@app.route("/chat", methods=["POST"])
def chat():
    try:
        user_input = request.json.get("message", "")
        if not user_input:
            return jsonify({"error": "No message provided"}), 400
        
        with model.chat_session():
            response = model.generate(user_input, max_tokens=200)
        
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    serve(app, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
