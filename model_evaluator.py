# model_evaluator.py
# Load saved model results and return for Flask template or CLI report

import json
import os


def load_results():
    path = "models/model_results.json"
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def get_results_for_template():
    """Returns (results_dict, best_model_name) for use in Flask templates."""
    data = load_results()
    if data:
        return data.get("results", {}), data.get("best_model", "")
    return {}, ""


def print_report():
    data = load_results()
    if not data:
        print("❌ No results found. Run model_trainer.py first.")
        return

    best    = data["best_model"]
    results = data["results"]

    print("\n" + "=" * 70)
    print("  MODEL EVALUATION REPORT — Resume Screening System")
    print("=" * 70)
    print(f"\n{'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1 Score':>10}")
    print("-" * 70)

    for name, m in results.items():
        marker = "  ← BEST" if name == best else ""
        print(f"{name:<25} {m['accuracy']:>9}% {m['precision']:>9}% {m['recall']:>9}% {m['f1']:>9}%{marker}")

    print("-" * 70)
    print(f"\n🏆 Selected model: {best}")
    print("=" * 70)


if __name__ == "__main__":
    print_report()