import os
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def run_hyperparameter_sweep():
    # Set experiment name for tracking
    experiment_name = "Stock_Movement_Hyperparameter_Tuning"
    mlflow.set_experiment(experiment_name)
    
    # 1. Load dataset from Feast output
    data_path = "data/training_dataset.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"{data_path} not found! Run get_training_data.py first.")
        
    df = pd.read_csv(data_path)
    df = df.dropna(subset=['rolling_avg_10', 'volume_sum_10', 'target'])
    
    feature_cols = ['rolling_avg_10', 'volume_sum_10']
    X = df[feature_cols]
    y = df['target']
    
    # Stratified train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 2. Simple hyperparameter grid
    param_grid = [
        {"n_estimators": 25,  "max_depth": 5,  "min_samples_split": 2},
        {"n_estimators": 50,  "max_depth": 10, "min_samples_split": 2},
        {"n_estimators": 100, "max_depth": 15, "min_samples_split": 5},
        {"n_estimators": 150, "max_depth": 20, "min_samples_split": 10},
    ]
    
    best_f1 = -1.0
    best_run_id = None
    best_model = None
    best_params = None
    
    print("\n==========================================")
    print("🚀 STARTING HYPERPARAMETER TUNING SWEEP")
    print("==========================================\n")
    
    # 3. Sweep over parameter grid and log every run
    for i, params in enumerate(param_grid, start=1):
        with mlflow.start_run(run_name=f"Sweep_Run_{i}") as run:
            # Log hyperparameters
            mlflow.log_params(params)
            
            # Train model
            clf = RandomForestClassifier(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                min_samples_split=params["min_samples_split"],
                random_state=42
            )
            clf.fit(X_train, y_train)
            
            # Evaluate predictions
            preds = clf.predict(X_test)
            acc = float(accuracy_score(y_test, preds))
            prec = float(precision_score(y_test, preds, zero_division=0))
            rec = float(recall_score(y_test, preds, zero_division=0))
            f1 = float(f1_score(y_test, preds, zero_division=0))
            
            # Log metrics
            mlflow.log_metrics({
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1_score": f1
            })
            
            # Log model artifact for every individual run
            mlflow.sklearn.log_model(clf, artifact_path="model")
            
            print(f"Run {i} [ID: {run.info.run_id[:8]}] -> "
                  f"n_estimators: {params['n_estimators']}, max_depth: {params['max_depth']} | "
                  f"F1 Score: {f1:.4f} | Accuracy: {acc:.4f}")
            
            # Track the winning run based on primary metric (F1 Score)
            if f1 > best_f1:
                best_f1 = f1
                best_run_id = run.info.run_id
                best_model = clf
                best_params = params

    print("\n==========================================")
    print("🏆 SWEEP RESULTS & MODEL PROMOTION")
    print("==========================================")
    print(f"Best Run ID  : {best_run_id}")
    print(f"Best Params  : {best_params}")
    print(f"Best F1 Score: {best_f1:.4f}")
    
    # 4. Register the best model into the MLflow Model Registry
    model_name = "StockMovementPredictor"
    model_uri = f"runs:/{best_run_id}/model"
    
    result = mlflow.register_model(model_uri=model_uri, name=model_name)
    
    print(f"\n✅ Promoted best model to MLflow Registry!")
    print(f"Registered Name : {result.name}")
    print(f"Model Version   : {result.version}")
    print("==========================================\n")

if __name__ == "__main__":
    run_hyperparameter_sweep()