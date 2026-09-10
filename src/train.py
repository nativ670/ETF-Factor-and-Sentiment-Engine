from pathlib import Path
from pyexpat import model
import mlflow
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    )
import xgboost as xgb
from data_loader import load_etf_data

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "mlflow.db"
TRACKING_URI = f"sqlite:///{DB_PATH.as_posix()}"

def main(ticker="SPY"):
    # Load ETF data for the specified ticker
    X_train, X_test, y_train, y_test = load_etf_data(ticker)

    # Set up MLflow tracking
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("ETF_Price_Prediction")

    mlflow.xgboost.autolog()  # Enable automatic logging for XGBoost

    with mlflow.start_run():
        mlflow.log_param("ticker", ticker)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))

        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()        # Train the XGBoost model
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            eval_metric="logloss",
            early_stopping_rounds=20,
            random_state=42,
            scale_pos_weight=scale_pos_weight,
        )
        model.fit(X_train, y_train, eval_set=[(X_train, y_train),(X_test, y_test)], verbose=False)

        y_prob = model.predict_proba(X_test)[:, 1]  # Get probabilities for the positive class
        y_pred = (y_prob >= 0.5).astype(int)  # Convert probabilities to binary predictions

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "brier_score_loss": brier_score_loss(y_test, y_prob),
            "log_loss": log_loss(y_test, y_prob),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_prob),
        }
        # Log the metrics to MLflow
        mlflow.log_metrics(metrics)

        print(f"Metrics for {ticker}:")
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}: {metric_value:.4f}")

if __name__ == "__main__":
    main()