/**
 * script.js
 * ====================================================================
 * SpamShield - Frontend Logic & API Integration
 * ====================================================================
 * Handles user interactions, input validation, character counting,
 * async fetch calls to Flask /predict and /metrics, and live UI updates.
 */

document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements - Form & Controls
    const spamForm = document.getElementById("spamForm");
    const messageInput = document.getElementById("messageInput");
    const modelSelect = document.getElementById("modelSelect");
    const charCounter = document.getElementById("charCounter");
    const submitBtn = document.getElementById("submitBtn");
    const clearBtn = document.getElementById("clearBtn");
    const btnSpinner = document.getElementById("btnSpinner");
    const btnText = document.getElementById("btnText");

    // DOM Elements - Feedback & Results
    const errorAlert = document.getElementById("errorAlert");
    const errorMessage = document.getElementById("errorMessage");
    const loadingIndicator = document.getElementById("loadingIndicator");
    const resultCard = document.getElementById("resultCard");
    const resultBadge = document.getElementById("resultBadge");
    const verdictIcon = document.getElementById("verdictIcon");
    const verdictHeadline = document.getElementById("verdictHeadline");
    const resultModelName = document.getElementById("resultModelName");
    const resultExplanation = document.getElementById("resultExplanation");

    // DOM Elements - Metrics Display
    const nbAccuracyValue = document.getElementById("nbAccuracyValue");
    const svmAccuracyValue = document.getElementById("svmAccuracyValue");
    const nbStatusText = document.getElementById("nbStatusText");
    const svmStatusText = document.getElementById("svmStatusText");
    const datasetTotalCount = document.getElementById("datasetTotalCount");
    const trainRecordsCount = document.getElementById("trainRecordsCount");
    const testRecordsCount = document.getElementById("testRecordsCount");
    const metricsAlert = document.getElementById("metricsAlert");
    const metricsAlertText = document.getElementById("metricsAlertText");
    const nbMetricCard = document.getElementById("nbMetricCard");
    const svmMetricCard = document.getElementById("svmMetricCard");
    const nbSelectedBadge = document.getElementById("nbSelectedBadge");
    const svmSelectedBadge = document.getElementById("svmSelectedBadge");

    // DOM Elements - Quick Sample Chips
    const sampleSpamBtn1 = document.getElementById("sampleSpamBtn1");
    const sampleSpamBtn2 = document.getElementById("sampleSpamBtn2");
    const sampleHamBtn1 = document.getElementById("sampleHamBtn1");
    const sampleHamBtn2 = document.getElementById("sampleHamBtn2");

    // Sample Texts for Demonstration
    const sampleTexts = {
        spam1: "Congratulations! You have won a free $1,000 Walmart Gift Card. Claim your cash reward today by calling now!",
        spam2: "URGENT! Your account access has been restricted. Verify your security credentials immediately to prevent suspension.",
        ham1: "Hey, are we meeting at the college library tomorrow to review our project presentation slides?",
        ham2: "I will reach the campus in ten minutes. Let's grab lunch at the canteen afterwards. See you soon!"
    };

    // State Tracking
    let isSubmitting = false;

    // ==========================================
    // 1. Character Counter Function
    // ==========================================
    function updateCharCounter() {
        const length = messageInput.value.length;
        charCounter.textContent = `${length.toLocaleString()} character${length === 1 ? '' : 's'}`;
    }

    messageInput.addEventListener("input", () => {
        updateCharCounter();
        // Clear error as user types
        if (errorAlert.style.display !== "none") {
            hideError();
        }
    });

    // ==========================================
    // 2. Sample Chip Handlers
    // ==========================================
    if (sampleSpamBtn1) {
        sampleSpamBtn1.addEventListener("click", () => {
            messageInput.value = sampleTexts.spam1;
            updateCharCounter();
            hideError();
            hideResult();
            messageInput.focus();
        });
    }

    if (sampleSpamBtn2) {
        sampleSpamBtn2.addEventListener("click", () => {
            messageInput.value = sampleTexts.spam2;
            updateCharCounter();
            hideError();
            hideResult();
            messageInput.focus();
        });
    }

    if (sampleHamBtn1) {
        sampleHamBtn1.addEventListener("click", () => {
            messageInput.value = sampleTexts.ham1;
            updateCharCounter();
            hideError();
            hideResult();
            messageInput.focus();
        });
    }

    if (sampleHamBtn2) {
        sampleHamBtn2.addEventListener("click", () => {
            messageInput.value = sampleTexts.ham2;
            updateCharCounter();
            hideError();
            hideResult();
            messageInput.focus();
        });
    }

    // ==========================================
    // 3. Clear Button Handler
    // ==========================================
    clearBtn.addEventListener("click", () => {
        messageInput.value = "";
        updateCharCounter();
        hideError();
        hideResult();
        hideLoading();
        messageInput.focus();
    });

    // ==========================================
    // 4. Helper UI Functions
    // ==========================================
    function showError(msg) {
        errorMessage.textContent = msg;
        errorAlert.style.display = "flex";
    }

    function hideError() {
        errorAlert.style.display = "none";
        errorMessage.textContent = "";
    }

    function showLoading() {
        loadingIndicator.style.display = "block";
        submitBtn.disabled = true;
        btnSpinner.style.display = "inline-block";
        btnText.textContent = "Classifying...";
        isSubmitting = true;
    }

    function hideLoading() {
        loadingIndicator.style.display = "none";
        submitBtn.disabled = false;
        btnSpinner.style.display = "none";
        btnText.textContent = "Check Message";
        isSubmitting = false;
    }

    function hideResult() {
        resultCard.style.display = "none";
        resultCard.classList.remove("is-spam", "is-ham");
    }

    // ==========================================
    // 5. Display Prediction Result
    // ==========================================
    function displayResult(data) {
        const isSpam = data.prediction.toLowerCase() === "spam";

        // Remove old classes
        resultCard.classList.remove("is-spam", "is-ham");
        resultBadge.classList.remove("badge-spam", "badge-ham");

        if (isSpam) {
            resultCard.classList.add("is-spam");
            resultBadge.classList.add("badge-spam");
            resultBadge.textContent = "Spam";
            if (verdictIcon) verdictIcon.textContent = "🚨";
            if (verdictHeadline) {
                verdictHeadline.textContent = "Result: Spam Message";
                verdictHeadline.style.color = "#cf222e";
            }
        } else {
            resultCard.classList.add("is-ham");
            resultBadge.classList.add("badge-ham");
            resultBadge.textContent = "Ham";
            if (verdictIcon) verdictIcon.textContent = "✅";
            if (verdictHeadline) {
                verdictHeadline.textContent = "Result: Legitimate Message (Ham)";
                verdictHeadline.style.color = "#1a7f37";
            }
        }

        if (resultModelName) resultModelName.textContent = data.model_used;
        if (resultExplanation) {
            resultExplanation.textContent = data.explanation || (
                `The message was classified as '${data.prediction}' by ${data.model_used} based on patterns learned during training.`
            );
        }

        resultCard.style.display = "block";
        resultCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }

    // ==========================================
    // 6. Predict Message (Submit Form)
    // ==========================================
    async function handlePredict(e) {
        if (e) e.preventDefault();

        // Prevent concurrent requests
        if (isSubmitting) return;

        const messageText = messageInput.value.trim();
        const selectedModel = modelSelect.value;

        // Validation: Empty message check
        if (!messageText) {
            hideResult();
            showError("Please enter or paste a message before clicking 'Check Message'.");
            messageInput.focus();
            return;
        }

        hideError();
        hideResult();
        showLoading();

        try {
            const response = await fetch("/predict", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    message: messageText,
                    model: selectedModel
                })
            });

            const data = await response.json();

            if (!response.ok) {
                // Server returned an error (400, 500, 503)
                showError(data.message || "Failed to classify message. Please check the server status.");
            } else {
                displayResult(data);
            }
        } catch (networkError) {
            // Network failure or server unreachable
            console.error("Prediction network error:", networkError);
            showError("Could not connect to the Python backend server. Make sure 'python app.py' is running.");
        } finally {
            hideLoading();
        }
    }

    spamForm.addEventListener("submit", handlePredict);

    // ==========================================
    // Synchronize Model Selection & Performance Cards
    // ==========================================
    function updateActiveModelUI() {
        const selected = modelSelect.value;
        if (selected === "naive_bayes") {
            if (nbMetricCard) {
                nbMetricCard.classList.add("active-row", "active-model");
            }
            if (svmMetricCard) {
                svmMetricCard.classList.remove("active-row", "active-model");
            }
            if (nbSelectedBadge) nbSelectedBadge.style.display = "inline-block";
            if (svmSelectedBadge) svmSelectedBadge.style.display = "none";
        } else {
            if (svmMetricCard) {
                svmMetricCard.classList.add("active-row", "active-model");
            }
            if (nbMetricCard) {
                nbMetricCard.classList.remove("active-row", "active-model");
            }
            if (svmSelectedBadge) svmSelectedBadge.style.display = "inline-block";
            if (nbSelectedBadge) nbSelectedBadge.style.display = "none";
        }
    }

    modelSelect.addEventListener("change", () => {
        updateActiveModelUI();
        // Automatically re-classify with the new model if a message is already tested!
        if (resultCard.style.display !== "none" && messageInput.value.trim().length > 0) {
            handlePredict();
        }
    });

    if (nbMetricCard) {
        nbMetricCard.addEventListener("click", () => {
            modelSelect.value = "naive_bayes";
            updateActiveModelUI();
            if (resultCard.style.display !== "none" && messageInput.value.trim().length > 0) {
                handlePredict();
            }
        });
    }

    if (svmMetricCard) {
        svmMetricCard.addEventListener("click", () => {
            modelSelect.value = "svm";
            updateActiveModelUI();
            if (resultCard.style.display !== "none" && messageInput.value.trim().length > 0) {
                handlePredict();
            }
        });
    }

    // ==========================================
    // 7. Load Real Model Metrics on Page Load
    // ==========================================
    async function fetchModelMetrics() {
        try {
            const response = await fetch("/metrics");
            const result = await response.json();

            if (response.ok && result.status === "success" && result.data) {
                const metrics = result.data;

                // Naive Bayes Accuracy
                if (metrics.models && metrics.models.naive_bayes && nbAccuracyValue) {
                    nbAccuracyValue.textContent = metrics.models.naive_bayes.accuracy_percent;
                }
                if (nbStatusText) {
                    nbStatusText.textContent = "Live scikit-learn model";
                }

                // SVM Accuracy
                if (metrics.models && metrics.models.svm && svmAccuracyValue) {
                    svmAccuracyValue.textContent = metrics.models.svm.accuracy_percent;
                }
                if (svmStatusText) {
                    svmStatusText.textContent = "Live scikit-learn model";
                }

                // Dataset counts
                if (metrics.dataset_total && datasetTotalCount) {
                    datasetTotalCount.textContent = `${metrics.dataset_total.toLocaleString()} records`;
                }
                if (metrics.train_count && trainRecordsCount) {
                    trainRecordsCount.textContent = `${metrics.train_count.toLocaleString()} rows (80%)`;
                }
                if (metrics.test_count && testRecordsCount) {
                    testRecordsCount.textContent = `${metrics.test_count.toLocaleString()} rows (20%)`;
                }

                if (metricsAlert) {
                    metricsAlert.style.display = "none";
                }
            } else {
                showMetricsAlert("Train the models to view performance. Run 'python train_model.py'.");
            }
        } catch (err) {
            console.warn("Metrics could not be fetched automatically:", err);
            showMetricsAlert("Train the models to view performance (python train_model.py).");
        }
    }

    function showMetricsAlert(msg) {
        if (metricsAlert && metricsAlertText) {
            metricsAlertText.textContent = msg;
            metricsAlert.style.display = "flex";
        }
        if (nbAccuracyValue) nbAccuracyValue.textContent = "N/A";
        if (svmAccuracyValue) svmAccuracyValue.textContent = "N/A";
    }

    // Initialize metrics fetch and character count on page load
    updateCharCounter();
    updateActiveModelUI();
    fetchModelMetrics();
});
