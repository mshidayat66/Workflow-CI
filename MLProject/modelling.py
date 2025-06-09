import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from mlflow.models.signature import infer_signature

df = pd.read_csv('personality_dataset_preprocessing.csv')

X = df.drop(["Personality"], axis=1)
y = df["Personality"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)

mlflow.set_experiment("Personality Classification Non Tuning")

with mlflow.start_run() as run:
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # Buat signature model menggunakan infer_signature
    signature = infer_signature(X_train, rf.predict(X_train))

    # Contoh input (ambil 5 baris dari X_train)
    input_example = X_train.head(5)

    # Log model dengan signature dan input_example
    mlflow.sklearn.log_model(
        sk_model=rf,
        artifact_path="model",
        signature=signature,
        input_example=input_example
    )

    print(f"Akurasi: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

    print("Run ID:", run.info.run_id)
