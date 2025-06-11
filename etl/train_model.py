import argparse
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import os

def train(input_path, model_path, test_size=0.2, random_state=42):
    df = pd.read_csv(input_path)
    X = df.drop(columns=['diagnosis'])
    y = df['diagnosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump({'model': model, 'X_test': X_test, 'y_test': y_test}, model_path)
    print(f"Model and test data saved to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Logistic Regression model")
    parser.add_argument("--input", required=True, help="Path to preprocessed CSV")
    parser.add_argument("--model", required=True, help="Path to save model and test data")
    args = parser.parse_args()
    train(args.input, args.model)
