import numpy as np
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

# Paths
BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

MODEL_PATHS = {
    "SVM": os.path.join(BASE_PATH, "models/svm/svm_predictions.npy"),
    "QSVM": os.path.join(BASE_PATH, "models/qsvm/qsvm_predictions.npy"),
    "VQC": os.path.join(BASE_PATH, "models/vqc/vqc_predictions.npy"),
}

Y_TEST_PATH = os.path.join(BASE_PATH, "data/processed/y_test.npy")


def load_data():
    y_true = np.load(Y_TEST_PATH)

    predictions = {}
    for model_name, path in MODEL_PATHS.items():
        if not os.path.exists(path):
            print(f"[WARNING] Missing predictions for {model_name}")
            continue
        predictions[model_name] = np.load(path)

    return y_true, predictions


def evaluate_model(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def compare_models():
    y_true, predictions = load_data()

    results = {}

    print("\n=== Model Comparison ===\n")

    for model_name, y_pred in predictions.items():
        metrics = evaluate_model(y_true, y_pred)
        results[model_name] = metrics

        print(f"{model_name}:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        print()

    plot_results(results)
    save_results(results)

    return results


def plot_results(results):
    metrics = ["accuracy", "precision", "recall", "f1_score"]
    model_names = list(results.keys())

    for metric in metrics:
        values = [results[m][metric] for m in model_names]

        plt.figure()
        plt.bar(model_names, values)
        plt.title(f"{metric.upper()} Comparison")
        plt.xlabel("Models")
        plt.ylabel(metric)
        plt.ylim(0, 1)

        save_path = os.path.join(BASE_PATH, f"results/graphs/{metric}_comparison.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()


def save_results(results):
    save_path = os.path.join(BASE_PATH, "results/reports/performance_summary.txt")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w") as f:
        for model, metrics in results.items():
            f.write(f"{model}:\n")
            for k, v in metrics.items():
                f.write(f"  {k}: {v:.4f}\n")
            f.write("\n")

if __name__ == "__main__":
    compare_models()