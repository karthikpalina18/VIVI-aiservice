# from flask import Flask, request, jsonify
# from gpt4all import GPT4All
# import os

# app = Flask(__name__)

# # Use the local model file only
# model_path = os.path.join(os.getcwd(), "ggml-nomic-ai-gpt4all-falcon-Q4_0.gguf")

# # Load the local model (prevent GPT4All from downloading)
# model = GPT4All(model_name=model_path, allow_download=False)

# @app.route("/")
# def home():
#     return jsonify({"message": "Local GPT4All API is running!"})

# @app.route("/chat", methods=["POST"])
# def chat():
#     data = request.get_json()
#     user_input = data.get("message", "")

#     if not user_input:
#         return jsonify({"error": "No message provided"}), 400

#     # Generate response from the model
#     with model.chat_session():
#         response = model.generate(user_input, max_tokens=200)

#     return jsonify({"response": response})

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000)

# from flask import Flask, request, jsonify
# from gpt4all import GPT4All
# import os
# import requests
# import sys

# app = Flask(__name__)

# MODEL_FILENAME = "ggml-nomic-ai-gpt4all-falcon-Q4_0.gguf"
# MODEL_GDRIVE_ID = "1hGfz95mD6JqYML3x205qqGr3cJ4yNOvY"

# def download_from_gdrive(file_id: str, dest_path: str, chunk_size: int = 32768):
#     """Download a Google Drive file given its file id."""
#     URL = "https://docs.google.com/uc?export=download"
#     session = requests.Session()

#     response = session.get(URL, params={"id": file_id}, stream=True)
#     token = None

#     # Handle Google Drive confirmation for large files
#     for k, v in response.cookies.items():
#         if k.startswith("download_warning"):
#             token = v

#     if token:
#         params = {"id": file_id, "confirm": token}
#         response = session.get(URL, params=params, stream=True)

#     with open(dest_path, "wb") as f:
#         for chunk in response.iter_content(chunk_size):
#             if chunk:
#                 f.write(chunk)


# def ensure_model():
#     """Make sure the model file exists locally."""
#     if os.path.exists(MODEL_FILENAME):
#         print(f"✅ Found model file: {MODEL_FILENAME}")
#         return

#     print(f"🔄 Downloading model {MODEL_FILENAME} from Google Drive…")
#     try:
#         download_from_gdrive(MODEL_GDRIVE_ID, MODEL_FILENAME)
#     except Exception as e:
#         print("❌ Error downloading model:", e)
#         sys.exit(1)
#     print("✅ Download complete.")


# # Ensure the model is downloaded before loading
# ensure_model()

# # Load the local model (prevent GPT4All from downloading)
# model = GPT4All(model_name=MODEL_FILENAME, allow_download=False)


# @app.route("/")
# def home():
#     return jsonify({"message": "Local GPT4All API is running!"})


# @app.route("/chat", methods=["POST"])
# def chat():
#     data = request.get_json()
#     user_input = data.get("message", "")

#     if not user_input:
#         return jsonify({"error": "No message provided"}), 400

#     # Generate response from the model
#     with model.chat_session():
#         response = model.generate(user_input, max_tokens=200)

#     return jsonify({"response": response})


# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 5000))
#     app.run(host="0.0.0.0", port=port)
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
    response = session.get(URL, params={"id": file_id}, stream=True)
    token = None
    for k, v in response.cookies.items():
        if k.startswith("download_warning"):
            token = v
    if token:
        response = session.get(URL, params={"id": file_id, "confirm": token}, stream=True)
    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size):
            if chunk:
                f.write(chunk)

# Download to root workspace if not exists
if not os.path.exists(MODEL_FILENAME):
    print(f"Downloading model {MODEL_FILENAME} from Google Drive...")
    download_from_gdrive(MODEL_GDRIVE_ID, MODEL_FILENAME)
    print("✅ Download complete.")

# Absolute path to the model
model_path = os.path.join(os.getcwd(), MODEL_FILENAME)

# Load GPT4All from root workspace
model = GPT4All(model_name=None, model_path=model_path, allow_download=False)

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
