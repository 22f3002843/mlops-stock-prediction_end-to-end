import os
import sys
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def run_iteration(iteration_tag):
    mlflow.set_experiment("Stock_Movement_Incremental_Iterations")
    
    data_path = "data/training_dataset.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"{data_path} not found!")
        
    df = pd.read_csv(data_path)
    df = df.dropna(subset=['rolling_avg_10', 'volume_sum_10', 'target'])
    
    feature_cols = ['rolling_avg_10', 'volume_sum_10']
    X = df[feature_cols]
    y = df['target']
    
    # 80/20 Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    with mlflow.start_run(run_name=f"Iteration_{iteration_tag}") as run:
        mlflow.log_param("iteration", iteration_tag)
        mlflow.log_param("num_samples", len(df))
        
        clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        clf.fit(X_train, y_train)
        
        preds = clf.predict(X_test)
        
        acc = float(accuracy_score(y_test, preds))
        prec = float(precision_score(y_test, preds, zero_division=0))
        rec = float(recall_score(y_test, preds, zero_division=0))
        f1 = float(f1_score(y_test, preds, zero_division=0))
        
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        
        mlflow.sklearn.log_model(clf, artifact_path="model")
        
        print("\n==========================================")
        print(f"📊 EVALUATION RESULTS: Iteration {iteration_tag}")
        print("==========================================")
        print(f"Total Dataset Size : {len(df)} records")
        print(f"Accuracy           : {acc:.4f}")
        print(f"Precision          : {prec:.4f}")
        print(f"Recall             : {rec:.4f}")
        print(f"F1 Score           : {f1:.4f}")
        print(f"MLflow Run ID      : {run.info.run_id}")
        print("==========================================\n")

if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "v0"
    run_iteration(tag)