import argparse
import shutil
import os

def save_results(model_path, metrics_path, storage_dir):
    os.makedirs(storage_dir, exist_ok=True)
    shutil.copy(model_path, os.path.join(storage_dir, os.path.basename(model_path)))
    shutil.copy(metrics_path, os.path.join(storage_dir, os.path.basename(metrics_path)))
    print(f"Copied model and metrics to {storage_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save results to storage")
    parser.add_argument("--model", required=True, help="Path to model file")
    parser.add_argument("--metrics", required=True, help="Path to metrics file")
    parser.add_argument("--storage", required=True, help="Path to storage directory (local or mounted cloud)")
    args = parser.parse_args()
    save_results(args.model, args.metrics, args.storage)
