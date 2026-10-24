# SpamShield: Email Spam Detection Using Machine Learning

An academic mini-project for **B.E. Computer Science & Engineering (Semester 1)**.

SpamShield is a machine-learning web application that classifies incoming messages and emails as either **Spam** or **Ham (Legitimate)** using Natural Language Processing (NLP) with **TF-IDF Vectorization**, **Multinomial Naive Bayes**, and **Support Vector Machine (Linear SVM)**.

---

## 1. Project Overview & Problem Statement

Unsolicited spam messages and phishing emails clog inboxes, waste productive time, and present cybersecurity hazards. Traditional keyword filters fail when spammers alter wording. Machine learning addresses this problem by statistically learning vocabulary distributions and word importance scores from thousands of labelled messages.

### Project Objective
1. Extract and preprocess text from raw datasets.
2. Convert text into numeric vectors using **TF-IDF (Term Frequency-Inverse Document Frequency)**.
3. Train and compare two classification algorithms:
   - **Multinomial Naive Bayes (MultinomialNB)**
   - **Support Vector Machine (LinearSVC)**
4. Measure evaluation metrics using holdout **test accuracy**.
5. Serve the trained models through a **Flask web application** with an interactive user interface.

---

## 2. Dataset Information & Clarification

- **Source:** Kaggle SMS Spam Collection / Enron Spam Dataset (`spam.csv`).
- **File Location:** `dataset/spam.csv`
- **Usable Records:** 4,993 clean messages (after removing duplicates).
- **Class Balance:**
  - **Ham (Legitimate):** 3,531 messages (~70.7%)
  - **Spam:** 1,462 messages (~29.3%)
- **Dataset Limitation:** The dataset contains short messages and email bodies. While it represents realistic spam patterns, it does not include full multi-part RFC 822 email headers (e.g., DKIM, SPF, IP hops). This project demonstrates NLP spam classification on message content for educational evaluation.

---

## 3. Technology Stack & Machine Learning Workflow

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **Language** | Python 3.10+ (Tested on Python 3.14) | Core backend and data processing |
| **Data Processing** | `pandas`, `numpy` | CSV parsing, cleaning, normalization |
| **Machine Learning** | `scikit-learn` | TF-IDF feature extraction, Naive Bayes, Linear SVM |
| **Model Persistence** | `joblib` | Serializing and loading `.pkl` models |
| **Backend Web Server** | `Flask` | Serving frontend and REST API (`/predict`, `/metrics`) |
| **Frontend UI** | HTML5, CSS3, Vanilla JavaScript | Responsive dark navy user interface |

### 5-Step ML Pipeline
```
[ Raw Message ]
       ↓
[ Preprocessing: Lowercasing, English Stopwords Removal ]
       ↓
[ TF-IDF Feature Extraction (5,000 top n-grams) ]
       ↓
[ Model Classification: Naive Bayes or Linear SVM ]
       ↓
[ JSON Prediction Output: "Spam" or "Ham" ]
```

---

## 4. Model Evaluation & Accuracy Results

Both models were trained using an **80% training split (3,994 samples)** and evaluated on an unseen **20% test split (999 samples)** with stratified sampling (`random_state=42`):

| Algorithm | Model Type | Test Set Accuracy | Training Time |
| :--- | :--- | :--- | :--- |
| **Multinomial Naive Bayes** | Probabilistic (Bayes' Theorem) | **94.89%** | < 0.2 seconds |
| **Linear SVM (LinearSVC)** | Margin-Maximizing Hyperplane | **98.70%** | < 0.5 seconds |

> **Note on Accuracy Calculation:**
> $$\text{Accuracy} = \frac{\text{Correct Predictions (True Positives + True Negatives)}}{\text{Total Test Messages}} \times 100\%$$
> All metric values are retrieved directly from `models/metrics.json` via the Flask `/metrics` endpoint.

---

## 5. Project Directory Structure

```text
spam-detection/
│
├── dataset/
│   └── spam.csv               # Kaggle dataset (extracted)
│
├── models/
│   ├── tfidf_vectorizer.pkl   # Fitted TF-IDF vectorizer (5,000 features)
│   ├── naive_bayes_model.pkl  # Trained MultinomialNB model
│   ├── svm_model.pkl          # Trained LinearSVC model
│   └── metrics.json           # Real test accuracy numbers and dataset statistics
│
├── app.py                     # Flask web server & API routes (/predict, /metrics)
├── train_model.py             # Script to load dataset, preprocess, train & save models
├── test_app.py                # Automated test script verifying API endpoints & sample inputs
├── requirements.txt           # Minimal project dependencies
├── index.html                 # Complete web application user interface (SpamShield)
├── style.css                  # Dark navy styling with purple/blue accents
├── script.js                  # Frontend async API integration and UI interactions
└── README.md                  # Comprehensive documentation and viva guide
```

---

## 6. Windows Setup & Run Instructions

Follow these exact steps in your **Windows Command Prompt (`cmd`)** or **VS Code Terminal**:

### Step 1: Open the Project Directory
```cmd
cd "c:\Users\Prushutam\OneDrive\Desktop\spam detection"
```

### Step 2: Install Dependencies
```cmd
python -m pip install -r requirements.txt
```

### Step 3: Train the Machine Learning Models
```cmd
python train_model.py
```
*Output will show dataset statistics, TF-IDF vector extraction, and test accuracy for both models.*

### Step 4: Run the Flask Web Application
```cmd
python app.py
```
*You will see:*
```
Starting SpamShield Flask Server...
Open http://127.0.0.1:5000 in your browser to use the application.
 * Running on http://127.0.0.1:5000
```

### Step 5: Open in Your Browser
Open your web browser (Chrome, Edge, Firefox) and navigate to:
```
http://127.0.0.1:5000
```

### Step 6: Run Automated Tests (Optional)
In a new terminal window:
```cmd
python test_app.py
```

---

## 7. Sample Test Cases

You can test these pre-configured messages directly in the web app:

1. **Spam Sample 1:**  
   `"Congratulations! You have won a free $1,000 Walmart Gift Card. Claim your cash reward today by calling now!"`  
   *Result:* **Spam**

2. **Spam Sample 2:**  
   `"URGENT! Your account access has been restricted. Verify your security credentials immediately to prevent suspension."`  
   *Result:* **Spam**

3. **Ham Sample 1:**  
   `"Hey, are we meeting at the college library tomorrow to review our project presentation slides?"`  
   *Result:* **Ham**

4. **Ham Sample 2:**  
   `"I will reach the campus in ten minutes. Let's grab lunch at the canteen afterwards. See you soon!"`  
   *Result:* **Ham**

---

## 8. College Viva Preparation (Questions & Answers)

Here are student-friendly answers for college lab viva and examinations:

#### Q1: What is Spam Detection?
> **Answer:** Spam detection is the automated process of identifying and filtering out unwanted, malicious, or commercial messages (spam) from genuine, desired communications (ham).

#### Q2: What is NLP (Natural Language Processing)?
> **Answer:** NLP is a branch of Artificial Intelligence that enables computers to understand, interpret, and process human text and spoken language.

#### Q3: What is Classification in Machine Learning?
> **Answer:** Classification is a supervised learning task where an algorithm learns from labelled past data to assign new, unseen observations into predefined categories (in our case: Spam or Ham).

#### Q4: What is the difference between Spam and Ham?
> **Answer:** "Spam" refers to unsolicited, unwanted, or harmful messages such as lottery scams and phishing attempts. "Ham" is a widely adopted industry term referring to legitimate, wanted messages from friends, colleagues, or services.

#### Q5: What is TF-IDF?
> **Answer:** TF-IDF stands for **Term Frequency-Inverse Document Frequency**. It converts text into numbers by evaluating how frequently a word appears in a message (TF) balanced against how common that word is across the entire dataset (IDF). Rare words specific to spam (like "claim", "winner", "prize") receive higher weights than universally common words.

#### Q6: How does Multinomial Naive Bayes work?
> **Answer:** Multinomial Naive Bayes is a probabilistic classifier based on **Bayes' Theorem**. It computes the probability that a message belongs to a class given its words, assuming that words occur independently of one another.

#### Q7: What is Support Vector Machine (SVM)?
> **Answer:** SVM is a supervised learning algorithm that finds an optimal boundary line (or hyperplane) in multi-dimensional space to separate two classes with the maximum possible margin between the closest data points (support vectors).

#### Q8: Why are two models used in this project?
> **Answer:** Using both Naive Bayes and Linear SVM allows us to compare a fast **probabilistic** model with a **margin-maximizing** geometric model, showing the trade-offs in accuracy and decision boundaries.

#### Q9: What is training data?
> **Answer:** Training data is the portion (80% in our project, 3,994 messages) of the labelled dataset given to the machine learning algorithms so they can learn patterns and associate words with Spam or Ham.

#### Q10: What is testing data?
> **Answer:** Testing data is a separate holdout portion (20%, 999 messages) that the model has never seen during training. It is used to fairly evaluate how accurately the model performs on new real-world data.

#### Q11: What is accuracy?
> **Answer:** Accuracy is the ratio of correct predictions to total predictions: $\frac{\text{Correct}}{\text{Total}} \times 100\%$. In our project, Linear SVM achieved 98.70% and Naive Bayes achieved 94.89%.

#### Q12: Why can accuracy be misleading for imbalanced datasets?
> **Answer:** In datasets where 90% of messages are Ham, a dummy model that predicts "Ham" for everything would still be 90% accurate while failing to catch any spam at all. For imbalanced datasets, metrics like Precision, Recall, and F1-score provide deeper insights.

#### Q13: What does Flask do in this project?
> **Answer:** Flask is a lightweight Python web framework. It acts as the bridge: it serves the HTML/CSS/JS frontend to the web browser and provides API endpoints (`/predict` and `/metrics`) that accept user messages, run the Python ML model, and return JSON results.

#### Q14: What is the purpose of `index.html`?
> **Answer:** `index.html` defines the semantic structure and user interface of the SpamShield web application, containing the input form, model selection dropdown, buttons, and result displays.

#### Q15: How does the frontend communicate with Python?
> **Answer:** The frontend uses the JavaScript `fetch()` API to send an asynchronous HTTP POST request with JSON data (`{"message": "...", "model": "..."}`) to the Flask backend endpoint `/predict`, and renders the JSON response dynamically without reloading the page.

#### Q16: Why is the TF-IDF vectorizer saved to disk?
> **Answer:** The vectorizer must learn the exact same vocabulary and word IDF weights that the model was trained on. If we don't save and reload the fitted vectorizer, incoming messages would be transformed inconsistently, causing completely wrong predictions.

#### Q17: What are the limitations of this project?
> **Answer:**
> 1. It only inspects text content and does not analyze email headers, IP sender reputations, or attachment files.
> 2. Highly disguised spam with misspelled words (e.g., "fr33 pr1ze") might evade the vocabulary.
> 3. Very short ambiguous messages may lack sufficient context for high confidence.

---

## 9. Academic Disclaimer

This project is created strictly for educational purposes as part of a Bachelor of Engineering (B.E.) Computer Science curriculum demonstration.
