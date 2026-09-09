from data_loader import load_etf_data
import xgboost as xgb
import mlflow
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test =  load_etf_data("SPY")

remote_server_uri = "sqlite:///mlflow.db"  # Replace with your MLflow server URI

mlflow.set_tracking_uri(remote_server_uri)
mlflow.set_experiment("ETF_Price_Prediction")
mlflow.xgboost.autolog()  # Enable automatic logging for XGBoost

with mlflow.start_run():
    model = xgb.XGBClassifier()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    mlflow.log_metric("custom_test_accuracy", accuracy)
        
    print(f"Model accuracy: {accuracy}")