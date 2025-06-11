import argparse
import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess(input_path, output_path):
    df = pd.read_csv(input_path)
    # Удаляем пустой столбец
    if 'Unnamed: 32' in df.columns:
        df.drop(columns=['Unnamed: 32'], inplace=True)
    # Кодируем diagnosis: M=1, B=0
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
    # Отделяем признаки и целевую
    X = df.drop(columns=['id', 'diagnosis'])
    y = df['diagnosis']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    df_processed = pd.DataFrame(X_scaled, columns=X.columns)
    df_processed['diagnosis'] = y.values
    df_processed.to_csv(output_path, index=False)
    print(f"Saved preprocessed data to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess data")
    parser.add_argument("--input", required=True, help="Path to raw data CSV")
    parser.add_argument("--output", required=True, help="Path to save preprocessed CSV")
    args = parser.parse_args()
    preprocess(args.input, args.output)
