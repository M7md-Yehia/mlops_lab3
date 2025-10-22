import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import lightgbm as lgb
import mlflow
import dagshub
dagshub.init(repo_owner='M7md-Yehia', repo_name='mlops-lab3', mlflow=True)
mlflow.lightgbm.autolog() 
mlflow.set_experiment("Telco Churn Prediction")

df = pd.read_csv('data/preprocessed_churn.csv')
X = df.drop('Churn', axis=1); y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run(run_name="LightGBM Autolog"):
    
    model = lgb.LGBMClassifier(random_state=42)
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    mlflow.log_metric("precision", precision_score(y_test, y_pred))
    
    mlflow.log_metric("recall", recall_score(y_test, y_pred))
    
    mlflow.log_metric("f1_score", f1_score(y_test, y_pred))
    
    print(f"LightGBM run logged with Autologging and custom metrics.")