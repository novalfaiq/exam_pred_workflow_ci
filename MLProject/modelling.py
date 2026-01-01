import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import argparse
import os

def run_training(n_estimators, max_depth):
    # Mengacu pada data hasil preprocessing (Kriteria 1)
    data_path = 'exam_score_prediction_preprocessing'
    
    # Load Data
    X_train = pd.read_csv(os.path.join(data_path, 'X_train.csv'))
    y_train = pd.read_csv(os.path.join(data_path, 'y_train.csv'))
    X_test = pd.read_csv(os.path.join(data_path, 'X_test.csv'))
    y_test = pd.read_csv(os.path.join(data_path, 'y_test.csv'))

    with mlflow.start_run():
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train.values.ravel())
        
        # Evaluasi
        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        # Manual Logging (Kriteria Skilled)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2_score", r2)
        
        mlflow.sklearn.log_model(model, "model")
        print(f"Training Selesai (Python 3.12.7). R2 Score: {r2:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=10)
    args = parser.parse_args()
    run_training(args.n_estimators, args.max_depth)