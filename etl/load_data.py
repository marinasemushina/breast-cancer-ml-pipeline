import argparse
import pandas as pd

def load_and_analyze(input_path, output_path):
    df = pd.read_csv(input_path)
    print(f"Data shape: {df.shape}")
    print(df.info())
    print(df.describe())
    df.to_csv(output_path, index=False)
    print(f"Saved raw data to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and analyze data")
    parser.add_argument("--input", required=True, help="Path to input CSV")
    parser.add_argument("--output", required=True, help="Path to save raw data CSV")
    args = parser.parse_args()
    load_and_analyze(args.input, args.output)
