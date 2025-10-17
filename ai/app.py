from flask import Flask, request, jsonify
from gpt4all import GPT4All
import os
from waitress import serve

app = Flask(__name__)

MODEL_FILENAME = "ggml-nomic-ai-gpt4all-falcon-Q4_0.gguf"
MODEL_URL = "https://gpt4all.io/models/ggml-nomic-ai-gpt4all-falcon-Q4_0.gguf"

def ensure_model():
    """Download model if not already present."""
    model_path = os.path.join(os.getcwd(), MODEL_FILENAME)
    
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"✅ Found model file: {MODEL_FILENAME} ({size_mb:.0f}MB)")
        return model_path
    
    print(f"🔄 Downloading model from {MODEL_URL}...")
    try:
        import urllib.request
        import shutil
        
        def download_with_progress(url, filepath):
            """Download with progress indicator"""
            with urllib.request.urlopen(url) as response:
                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0
                chunk_size = 1024 * 1024  # 1MB chunks
                
                with open(filepath, 'wb') as f:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size:
                            percent = (downloaded / total_size) * 100
                            mb_downloaded = downloaded / (1024 * 1024)
                            mb_total = total_size / (1024 * 1024)
                            print(f"Downloaded: {mb_downloaded:.0f}MB / {mb_total:.0f}MB ({percent:.1f}%)", end="\r")
        
        download_with_progress(MODEL_URL, model_path)
        print("\n✅ Download complete.")
        return model_path
    except Exception as e:
        print(f"\n❌ Error downloading model: {e}")
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
