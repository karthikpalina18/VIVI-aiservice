from flask import Flask, request, jsonify
from gpt4all import GPT4All
import os
import requests
from waitress import serve

app = Flask(__name__)

# Model info
MODEL_FILENAME = "ggml-nomic-ai-gpt4all-falcon-Q4_0.gguf"
MODEL_GDRIVE_ID = "1hGfz95mD6JqYML3x205qqGr3cJ4yNOvY"

def download_from_gdrive(file_id: str, dest_path: str, chunk_size: int = 32768):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    
    # First request to get any warning token
    response = session.get(URL, params={"id": file_id}, stream=True)
    token = None
    
    for k, v in response.cookies.items():
        if k.startswith("download_warning"):
            token = v
            break
    
    # For large files, use confirm=t to bypass the virus scan warning
    if token:
        params = {"id": file_id, "confirm": token}
    else:
        params = {"id": file_id, "confirm": "t"}  # Bypass large file warning
    
    response = session.get(URL, params=params, stream=True, timeout=60)
    response.raise_for_status()
    
    # Verify we're actually downloading a binary file, not HTML
    content_type = response.headers.get('content-type', '').lower()
    if 'text/html' in content_type:
        raise RuntimeError("Google Drive returned HTML. File may be inaccessible or sharing settings incorrect.")
    
    total_size = int(response.headers.get('content-length', 0))
    downloaded = 0
    
    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size:
                    percent = (downloaded / total_size) * 100
                    print(f"Downloaded: {percent:.1f}%", end="\r")

# Download to root workspace if not exists
if not os.path.exists(MODEL_FILENAME):
    print(f"Downloading model {MODEL_FILENAME} from Google Drive...")
    download_from_gdrive(MODEL_GDRIVE_ID, MODEL_FILENAME)
    print("✅ Download complete.")

# Set model_path to just the directory, not the full file path
model_path = os.getcwd()

# Load GPT4All - pass model_name instead of None
model = GPT4All(model_name=MODEL_FILENAME, model_path=model_path, allow_download=False)

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
