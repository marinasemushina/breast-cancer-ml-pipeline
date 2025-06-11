import argparse
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
import os

def evaluate(model_path, metrics_path):
    data = joblib.load(model_path)
    model = data['model']
    X_test = data['X_test']
    y_test = data['y_test']
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred)
    }
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Saved metrics to {metrics_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model")
    parser.add_argument("--model", required=True, help="Path to model and test data")
    parser.add_argument("--metrics", required=True, help="Path to save metrics JSON")
    args = parser.parse_args()
    evaluate(args.model, args.metrics)
