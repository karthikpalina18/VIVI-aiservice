from flask import Flask, request, jsonify
from gpt4all import GPT4All
import os
from waitress import serve

app = Flask(__name__)

# Model info
MODEL_FILENAME = "ggml-nomic-ai-gpt4all-falcon-Q4_0.gguf"

# Download model from Hugging Face (no need for manual download)
def ensure_model():
    """Download model from Hugging Face if not already present."""
    model_path = os.path.join(os.getcwd(), MODEL_FILENAME)
    
    if os.path.exists(model_path):
        print(f"✅ Found model file: {MODEL_FILENAME}")
        return model_path
    
    print(f"🔄 Downloading model from Hugging Face...")
    try:
        from huggingface_hub import hf_hub_download
        
        path = hf_hub_download(
            repo_id="TheBloke/Nomic-Falcon-3B-GGUF",
            filename=MODEL_FILENAME,
            cache_dir=os.getcwd(),
            local_files_only=False
        )
        print("✅ Download complete.")
        return path
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        raise

# Ensure the model is downloaded before loading
model_path = ensure_model()

# Load GPT4All
model = GPT4All(model_name=MODEL_FILENAME, model_path=os.getcwd(), allow_download=False)

@app.route("/")
def home():
    return jsonify({"message": "Local GPT4All API is running!"})

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "")
    if not user_input:
        return jsonify({"error": "No message provided"}), 400
    
    with model.chat_session():
        response = model.generate(user_input, max_tokens=200)
    
    return jsonify({"response": response})

if __name__ == "__main__":
    serve(app, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
