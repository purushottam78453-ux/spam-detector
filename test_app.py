"""
test_app.py
Comprehensive test suite for the SpamShield Flask Backend.
"""
from app import app

def run_tests():
    client = app.test_client()

    print("--- Test 1: GET / (index.html) ---")
    res = client.get('/')
    print("Status:", res.status_code, "Bytes:", len(res.data))
    assert res.status_code == 200, "Root route failed!"

    print("\n--- Test 2: GET /style.css ---")
    res = client.get('/style.css')
    print("Status:", res.status_code, "Bytes:", len(res.data))
    assert res.status_code == 200, "style.css route failed!"

    print("\n--- Test 3: GET /script.js ---")
    res = client.get('/script.js')
    print("Status:", res.status_code, "Bytes:", len(res.data))
    assert res.status_code == 200, "script.js route failed!"

    print("\n--- Test 4: GET /metrics ---")
    res = client.get('/metrics')
    data = res.get_json()
    print("Status:", res.status_code)
    print("Naive Bayes Accuracy:", data["data"]["models"]["naive_bayes"]["accuracy_percent"])
    print("SVM Accuracy:        ", data["data"]["models"]["svm"]["accuracy_percent"])
    assert res.status_code == 200, "/metrics endpoint failed!"

    print("\n--- Test 5: POST /predict (empty input validation) ---")
    res = client.post('/predict', json={'message': '   ', 'model': 'naive_bayes'})
    print("Status:", res.status_code, "Response:", res.get_json())
    assert res.status_code == 400, "Empty message should return 400!"

    print("\n--- Test 6: POST /predict (invalid model validation) ---")
    res = client.post('/predict', json={'message': 'Hello friend', 'model': 'unsupported_model'})
    print("Status:", res.status_code, "Response:", res.get_json())
    assert res.status_code == 400, "Invalid model should return 400!"

    print("\n--- Test 7: POST /predict (Five Sample Messages from prompt) ---")
    test_cases = [
        ("Congratulations! You have won a free prize. Claim now!", "naive_bayes"),
        ("Congratulations! You have won a free prize. Claim now!", "svm"),
        ("URGENT! Claim your reward today!", "svm"),
        ("Hey, are we meeting at college tomorrow?", "naive_bayes"),
        ("I will reach in ten minutes.", "svm"),
        ("Ok, see you later.", "naive_bayes")
    ]

    for msg, model_key in test_cases:
        res = client.post('/predict', json={'message': msg, 'model': model_key})
        assert res.status_code == 200, f"Prediction failed for: {msg}"
        result_json = res.get_json()
        pred = result_json["prediction"]
        model_name = result_json["model_used"]
        print(f"[{pred.upper():4}] via {model_name:30} -> \"{msg}\"")

    print("\n==========================================")
    print("ALL TESTS PASSED SUCCESSFULLY (7/7)!")
    print("==========================================")

if __name__ == "__main__":
    run_tests()
