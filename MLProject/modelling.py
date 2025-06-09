import pandas as pd
import mlflow
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Load dataset
df = pd.read_csv('personality_dataset_preprocessing.csv')

# Split dataset
X = df.drop(["Personality"], axis=1)
y = df["Personality"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)

# Set experiment (local tracking by default)
mlflow.set_experiment("Personality Classification Non Tuning")

# Start MLflow run
with mlflow.start_run() as run:

    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)

    # Manual metrics logging
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # Save and log model
    joblib.dump(rf, "model.pkl")
    mlflow.sklearn.log_model(rf, "model")

    print(f"Akurasi: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
