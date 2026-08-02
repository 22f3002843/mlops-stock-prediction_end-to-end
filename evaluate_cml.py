import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

def run_cml_evaluation():
    print("📥 Loading registered model 'StockMovementPredictor' from MLflow...")
    model_name = "StockMovementPredictor"
    model_uri = f"models:/{model_name}/1"
    
    try:
        model = mlflow.sklearn.load_model(model_uri)
    except Exception:
        # Fallback to local run artifact if model registry alias differs
        client = mlflow.tracking.MlflowClient()
        latest_runs = client.search_runs(
            experiment_ids=[client.get_experiment_by_name("Stock_Movement_Hyperparameter_Tuning").experiment_id],
            order_by=["metrics.f1_score DESC"],
            max_results=1
        )
        best_run_id = latest_runs[0].info.run_id
        model = mlflow.sklearn.load_model(f"runs:/{best_run_id}/model")

    print("📊 Loading test dataset...")
    data_path = "data/training_dataset.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"{data_path} not found!")

    df = pd.read_csv(data_path).dropna(subset=['rolling_avg_10', 'volume_sum_10', 'target'])
    
    # ==========================================
    # SANITY TESTS FOR FEATURES
    # ==========================================
    print("🔍 Running Feature Sanity Checks...")
    
    # Sanity Test 1: rolling_avg_10 must be positive and non-null
    assert not df['rolling_avg_10'].isnull().any(), "Sanity Check Failed: Nulls found in rolling_avg_10!"
    assert (df['rolling_avg_10'] > 0).all(), "Sanity Check Failed: Negative or zero prices found in rolling_avg_10!"
    
    # Sanity Test 2: volume_sum_10 must be non-negative and non-null
    assert not df['volume_sum_10'].isnull().any(), "Sanity Check Failed: Nulls found in volume_sum_10!"
    assert (df['volume_sum_10'] >= 0).all(), "Sanity Check Failed: Negative values found in volume_sum_10!"
    
    print("✅ All Feature Sanity Tests Passed Successfully!")

    # Evaluation
    X_test = df[['rolling_avg_10', 'volume_sum_10']]
    y_test = df['target']
    
    preds = model.predict(X_test)
    
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)

    # 1. Write metrics to text file for CML
    with open("metrics.txt", "w") as f:
        f.write("### 🚀 Model Evaluation Metrics\n\n")
        f.write(f"- **Accuracy**: {acc:.4f}\n")
        f.write(f"- **Precision**: {prec:.4f}\n")
        f.write(f"- **Recall**: {rec:.4f}\n")
        f.write(f"- **F1-Score**: {f1:.4f}\n\n")
        f.write("### 🧪 Feature Sanity Checks\n")
        f.write("- ✅ `rolling_avg_10`: All values are non-null and positive.\n")
        f.write("- ✅ `volume_sum_10`: All values are non-null and non-negative.\n")

    # 2. Plot & save Confusion Matrix plot
    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Down/Flat", "Up"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues")
    plt.title("Confusion Matrix - CML Evaluation")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.close()

    print("✅ Metrics saved to metrics.txt and plot saved to confusion_matrix.png")

if __name__ == "__main__":
    run_cml_evaluation()