"""
app.py
====================================================================
Email / SMS Spam Detection - Flask Web Server & API
====================================================================
Author: Student Project (B.E. Computer Science)
Objective: Serves the SpamShield web application and exposes API endpoints
           for real-time spam prediction and model performance metrics.
"""

import os
import json
import joblib
from flask import Flask, request, jsonify, send_from_directory

# Initialize Flask app
# static_folder="." allows serving files directly from the project root
app = Flask(__name__, static_folder=".")

# File paths
MODELS_DIR = "models"
VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
NAIVE_BAYES_PATH = os.path.join(MODELS_DIR, "naive_bayes_model.pkl")
SVM_PATH = os.path.join(MODELS_DIR, "svm_model.pkl")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")

# In-memory variables to store loaded models
vectorizer = None
naive_bayes_model = None
svm_model = None
model_metrics = None


def load_saved_models():
    """
    Loads trained models and the TF-IDF vectorizer into memory at startup.
    This avoids retraining or re-reading files on every incoming request.
    """
    global vectorizer, naive_bayes_model, svm_model, model_metrics

    # Check if model files exist
    required_files = [VECTORIZER_PATH, NAIVE_BAYES_PATH, SVM_PATH, METRICS_PATH]
    missing = [f for f in required_files if not os.path.exists(f)]

    if missing:
        print(f"Warning: The following model files are missing: {missing}")
        print("Please run 'python train_model.py' to generate model files.")
        return False

    try:
        # Load the fitted TF-IDF vectorizer
        vectorizer = joblib.load(VECTORIZER_PATH)

        # Load both trained machine learning models
        naive_bayes_model = joblib.load(NAIVE_BAYES_PATH)
        svm_model = joblib.load(SVM_PATH)

        # Load real evaluation metrics
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            model_metrics = json.load(f)

        print("All models and vectorizer loaded successfully into memory.")
        return True
    except Exception as e:
        print(f"Error loading models: {e}")
        return False


# Attempt loading models when the server starts
load_saved_models()


# ==========================================
# Frontend Routes
# ==========================================

@app.route("/")
def home():
    """Serves the main web application page."""
    return send_from_directory(".", "index.html")


@app.route("/<path:filename>")
def serve_static_file(filename):
    """
    Serves static assets (style.css, script.js) safely from the root directory.
    """
    allowed_files = ["style.css", "script.js"]
    if filename in allowed_files:
        return send_from_directory(".", filename)
    return jsonify({"error": "Resource not found"}), 404


# ==========================================
# Machine Learning API Endpoints
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():
    """
    API endpoint: Classify an input message as Spam or Ham.
    Expects JSON payload:
        {
            "message": "Congratulations! You won a prize...",
            "model": "naive_bayes" or "svm"
        }
    """
    # 1. Verify models are loaded
    if vectorizer is None or naive_bayes_model is None or svm_model is None:
        return jsonify({
            "status": "error",
            "message": "Models are not loaded. Please run 'python train_model.py' first."
        }), 503

    # 2. Parse request JSON data
    data = request.get_json(silent=True)
    if not data:
        return jsonify({
            "status": "error",
            "message": "Invalid request. Please provide JSON data."
        }), 400

    message_text = data.get("message", "").strip()
    selected_model_key = data.get("model", "naive_bayes").strip().lower()

    # 3. Input validation: Check for empty input
    if not message_text:
        return jsonify({
            "status": "error",
            "message": "Message text cannot be empty. Please enter a message to classify."
        }), 400

    # 4. Model selection validation
    if selected_model_key not in ["naive_bayes", "svm"]:
        return jsonify({
            "status": "error",
            "message": "Invalid model choice. Supported options are 'naive_bayes' or 'svm'."
        }), 400

    try:
        # 5. Transform raw text using the fitted TF-IDF vectorizer
        message_vector = vectorizer.transform([message_text])

        # 6. Run prediction with selected model
        if selected_model_key == "naive_bayes":
            prediction_result = naive_bayes_model.predict(message_vector)[0]
            model_display_name = "Multinomial Naive Bayes"
        else:
            prediction_result = svm_model.predict(message_vector)[0]
            model_display_name = "Linear Support Vector Machine (SVM)"

        # 7. Return friendly JSON response
        return jsonify({
            "status": "success",
            "prediction": prediction_result,   # "Spam" or "Ham"
            "model_used": model_display_name,
            "model_key": selected_model_key,
            "character_count": len(message_text),
            "explanation": (
                f"The message was classified as '{prediction_result}' by {model_display_name} "
                "based on vocabulary and word frequency patterns learned from the dataset."
            )
        }), 200

    except Exception as e:
        # Catch unexpected errors without crashing or leaking full stack trace
        return jsonify({
            "status": "error",
            "message": "An error occurred while processing the prediction request."
        }), 500


@app.route("/metrics", methods=["GET"])
def get_metrics():
    """
    API endpoint: Return real test accuracy and dataset metrics.
    """
    global model_metrics

    # Reload metrics from file if not loaded
    if model_metrics is None and os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                model_metrics = json.load(f)
        except Exception:
            pass

    if model_metrics is None:
        return jsonify({
            "status": "error",
            "message": "Model performance metrics not found. Train the models to view performance."
        }), 404

    return jsonify({
        "status": "success",
        "data": model_metrics
    }), 200


# ==========================================
# Application Entry Point
# ==========================================

if __name__ == "__main__":
    print("\nStarting SpamShield Flask Server...")
    print("Open http://127.0.0.1:5000 in your browser to use the application.\n")
    app.run(host="127.0.0.1", port=5000, debug=False)
