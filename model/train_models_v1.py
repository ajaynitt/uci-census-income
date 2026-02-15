import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import os

def preprocess_data(df):
    df = df.replace("?", np.nan)
    df = df.dropna()

    for col in df.select_dtypes(include=['object']).columns:
        df[col] = LabelEncoder().fit_transform(df[col])

    return df


def train_models():
    # 1. Define the column names (from adult.names file)
    column_names = [
        "age", "workclass", "fnlwgt", "education", "education-num",
        "marital-status", "occupation", "relationship", "race", "sex",
        "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
    ]

    # 2. Use a robust path
    base_dir = os.path.dirname(__file__)
    data_path = os.path.join(base_dir, "..", "data", "adult.data")  # or "adult.csv" if renamed

    # 3. Load the data, telling pandas there is no header row
    df = pd.read_csv(data_path, names=column_names, sep=',', skipinitialspace=True)
    df = preprocess_data(df)

    X = df.drop("income", axis=1)
    y = df["income"]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        # "Logistic_Regression": LogisticRegression(max_iter=1000,class_weight='balanced'),
        "Logistic_Regression": LogisticRegression(max_iter=1000),
        "Decision_Tree": DecisionTreeClassifier(),
        "KNN": KNeighborsClassifier(),
        "Naive_Bayes": GaussianNB(),
        "Random_Forest": RandomForestClassifier(),
        "XGBoost": XGBClassifier(eval_metric='logloss')
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        results[name] = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1": f1_score(y_test, y_pred),
            "MCC": matthews_corrcoef(y_test, y_pred),
        }

        try:
            y_prob = model.predict_proba(X_test)[:, 1]
            results[name]["AUC"] = roc_auc_score(y_test, y_prob)
        except:
            results[name]["AUC"] = "N/A"

        joblib.dump(model, f"{name}.pkl")

    return results


if __name__ == "__main__":
    results = train_models()
    print(results)
