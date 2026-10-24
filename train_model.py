"""
train_model.py
====================================================================
Email / SMS Spam Detection - Machine Learning Training Script
====================================================================
Author: Student Project (B.E. Computer Science)
Objective: Train Multinomial Naive Bayes and Support Vector Machine (SVM)
           models to classify messages as Spam or Ham (Not Spam).
"""

import os
import json
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score


def load_and_prepare_dataset(file_path):
    """
    Loads the dataset from a CSV file and identifies the message and label columns.
    Supports both SMS Spam Collection format (v1, v2) and Email Spam format (label, text).
    """
    # 1. Check if the dataset file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Dataset file not found at: '{file_path}'. "
            "Please ensure the CSV file exists in the 'dataset/' folder."
        )

    print(f"Loading dataset from: {file_path}")

    # 2. Read the CSV file (handling possible encoding issues like latin-1 or utf-8)
    try:
        data = pd.read_csv(file_path, encoding="utf-8")
    except UnicodeDecodeError:
        data = pd.read_csv(file_path, encoding="latin-1")

    print(f"Original dataset shape: {data.shape} (rows, columns)")
    print(f"Original columns: {list(data.columns)}")

    # 3. Identify label and message columns dynamically
    label_col = None
    message_col = None

    # Check for SMS Spam Collection format: 'v1' is label, 'v2' is message
    if "v1" in data.columns and "v2" in data.columns:
        label_col = "v1"
        message_col = "v2"
    # Check for Email Spam dataset format: 'label' and 'text' or 'message'
    elif "label" in data.columns and "text" in data.columns:
        label_col = "label"
        message_col = "text"
    elif "label" in data.columns and "message" in data.columns:
        label_col = "label"
        message_col = "message"
    elif "Category" in data.columns and "Message" in data.columns:
        label_col = "Category"
        message_col = "Message"
    else:
        # Search for any column name containing 'label'/'class'/'type' and 'text'/'message'
        for col in data.columns:
            clean_col = col.strip().lower()
            if clean_col in ["label", "type", "category", "class", "target", "v1"]:
                label_col = col
            if clean_col in ["text", "message", "sms", "email", "content", "v2"]:
                message_col = col

    # Verify that both columns were found
    if label_col is None or message_col is None:
        raise ValueError(
            "Could not automatically detect label and message columns. "
            f"Columns in CSV: {list(data.columns)}. "
            "Expected 'v1' and 'v2' or 'label' and 'text'."
        )

    print(f"Identified label column: '{label_col}'")
    print(f"Identified message column: '{message_col}'")

    # 4. Keep only the label and message columns (ignores extra columns like Unnamed: 2, 3, etc.)
    df = data[[label_col, message_col]].copy()
    df.columns = ["label", "message"]

    # 5. Remove missing messages or unusable labels
    df = df.dropna(subset=["label", "message"])

    # Convert messages to strings (in case any numbers were parsed as numeric)
    df["message"] = df["message"].astype(str)

    # 6. Normalize labels to 'Ham' and 'Spam'
    def clean_label(val):
        val_str = str(val).strip().lower()
        if val_str in ["spam", "1", "1.0", "true"]:
            return "Spam"
        elif val_str in ["ham", "0", "0.0", "false", "not spam"]:
            return "Ham"
        return None

    df["label"] = df["label"].apply(clean_label)
    df = df.dropna(subset=["label"])

    # 7. Remove duplicate rows where appropriate
    initial_count = len(df)
    df = df.drop_duplicates(subset=["message"]).reset_index(drop=True)
    duplicates_removed = initial_count - len(df)

    if duplicates_removed > 0:
        print(f"Removed {duplicates_removed} duplicate messages.")

    # 8. Print dataset summary
    print(f"Total usable messages: {len(df)}")
    counts = df["label"].value_counts()
    ham_count = counts.get("Ham", 0)
    spam_count = counts.get("Spam", 0)
    print(f"  - Ham (Not Spam) messages: {ham_count}")
    print(f"  - Spam messages:           {spam_count}")

    if ham_count == 0 or spam_count == 0:
        raise ValueError("Dataset must contain both Ham and Spam examples to train.")

    return df


def train_models():
    """
    Main function to train TF-IDF, Multinomial Naive Bayes, and Linear SVM models.
    Saves the models and metrics to the 'models/' directory.
    """
    print("=" * 60)
    print("Email / SMS Spam Detection Model Training")
    print("=" * 60)

    # Dataset file path
    dataset_path = os.path.join("dataset", "spam.csv")
    if not os.path.exists(dataset_path):
        # Fallback check if the original extracted name is present
        alt_path = os.path.join("dataset", "spam_ham_dataset.csv")
        if os.path.exists(alt_path):
            dataset_path = alt_path

    # Step 1: Load and prepare dataset
    df = load_and_prepare_dataset(dataset_path)

    # Step 2: Separate features (X) and target labels (y)
    X = df["message"]
    y = df["label"]

    # Step 3: Split into 80% training data and 20% test data
    # We use random_state=42 for reproducibility.
    # stratify=y preserves the exact proportion of Spam and Ham in both sets.
    print("\nSplitting dataset into 80% training and 20% testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Training messages count: {len(X_train)}")
    print(f"Testing messages count:  {len(X_test)}")

    # Step 4: Convert raw text to numbers using TF-IDF (Term Frequency-Inverse Document Frequency)
    # Important: Fit ONLY on training data to prevent data leakage.
    print("\nExtracting text features using TF-IDF Vectorizer...")
    vectorizer = TfidfVectorizer(
        stop_words="english",
        lowercase=True,
        max_features=5000  # Keep top 5,000 important words for fast, accurate inference
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Step 5: Train Model 1 - Multinomial Naive Bayes
    print("\nTraining Model 1: Multinomial Naive Bayes (MultinomialNB)...")
    nb_model = MultinomialNB()
    nb_model.fit(X_train_tfidf, y_train)

    # Evaluate Naive Bayes on test data
    nb_predictions = nb_model.predict(X_test_tfidf)
    nb_accuracy = accuracy_score(y_test, nb_predictions)
    print(f"-> Naive Bayes Test Accuracy: {nb_accuracy * 100:.2f}%")

    # Step 6: Train Model 2 - Linear Support Vector Machine (LinearSVC)
    print("\nTraining Model 2: Linear Support Vector Machine (LinearSVC)...")
    svm_model = LinearSVC(random_state=42, dual="auto")
    svm_model.fit(X_train_tfidf, y_train)

    # Evaluate SVM on test data
    svm_predictions = svm_model.predict(X_test_tfidf)
    svm_accuracy = accuracy_score(y_test, svm_predictions)
    print(f"-> Linear SVM Test Accuracy:   {svm_accuracy * 100:.2f}%")

    # Step 7: Create 'models/' folder if it does not exist
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)

    # Step 8: Save vectorizer and trained models using joblib
    print("\nSaving trained models to disk...")
    vectorizer_path = os.path.join(models_dir, "tfidf_vectorizer.pkl")
    nb_model_path = os.path.join(models_dir, "naive_bayes_model.pkl")
    svm_model_path = os.path.join(models_dir, "svm_model.pkl")

    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(nb_model, nb_model_path)
    joblib.dump(svm_model, svm_model_path)

    print(f"Saved: {vectorizer_path}")
    print(f"Saved: {nb_model_path}")
    print(f"Saved: {svm_model_path}")

    # Step 9: Save real test metrics to JSON file
    metrics = {
        "dataset_total": int(len(df)),
        "train_count": int(len(X_train)),
        "test_count": int(len(X_test)),
        "ham_count": int(df["label"].value_counts().get("Ham", 0)),
        "spam_count": int(df["label"].value_counts().get("Spam", 0)),
        "models": {
            "naive_bayes": {
                "name": "Multinomial Naive Bayes",
                "accuracy": round(float(nb_accuracy), 4),
                "accuracy_percent": f"{nb_accuracy * 100:.2f}%"
            },
            "svm": {
                "name": "Linear Support Vector Machine (SVM)",
                "accuracy": round(float(svm_accuracy), 4),
                "accuracy_percent": f"{svm_accuracy * 100:.2f}%"
            }
        }
    }

    metrics_path = os.path.join(models_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    print(f"Saved metrics to: {metrics_path}")
    print("\n" + "=" * 60)
    print("Training Complete! Models are ready to use in the web app.")
    print("=" * 60)


if __name__ == "__main__":
    train_models()
