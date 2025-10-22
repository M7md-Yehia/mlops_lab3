import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
import dagshub
import mlflow
import os

dagshub.init(repo_owner='M7md-Yehia', repo_name='mlops-lab3', mlflow=True) 
mlflow.set_experiment("Telco Churn Prediction")
df = pd.read_csv('data/preprocessed_churn.csv')
X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


list_n_estimators = [50, 100]
list_max_depth = [5, 10]

with mlflow.start_run(run_name="RandomForest Tuning"):
    mlflow.set_tag("model_type", "RandomForest")
    print("Starting Random Forest Tuning experiments...")

    for n_estimators in list_n_estimators:
        for max_depth in list_max_depth:
            
            with mlflow.start_run(run_name=f"RF (n={n_estimators}, depth={max_depth})", nested=True):
                
                mlflow.log_param("n_estimators", n_estimators)
                mlflow.log_param("max_depth", max_depth)
                
                model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
                
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("auc", auc)
                
                mlflow.log_metric("precision", precision_score(y_test, y_pred))
                mlflow.log_metric("recall", recall_score(y_test, y_pred))
                mlflow.log_metric("f1_score", f1_score(y_test, y_pred))

                print(f"  > experiment (n={n_estimators}, depth={max_depth}) اكتملت. Accuracy: {accuracy:.4f}")

    print("done with Random Forest Tuning experiments.")