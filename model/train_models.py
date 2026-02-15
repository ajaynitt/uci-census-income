import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def train_models():

    column_names = [
        "age", "workclass", "fnlwgt", "education", "education-num",
        "marital-status", "occupation", "relationship", "race", "sex",
        "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
    ]

    base_dir = os.path.dirname(__file__)
    data_path = os.path.join(base_dir, "..", "data", "adult.data")

    df = pd.read_csv(data_path, names=column_names, sep=',', skipinitialspace=True)

    df["income"] = df["income"].str.replace(".", "", regex=False)

    X = df.drop("income", axis=1)
    # y = df["income"]
    y = df["income"].map({"<=50K": 0, ">50K": 1})

    categorical_cols = X.select_dtypes(include=["object"]).columns
    numerical_cols = X.select_dtypes(exclude=["object"]).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            # ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols) #Issue with #TypeError: Sparse data was passed for X, but dense data is required.
            #GaussianNB() (Naive Bayes) does NOT accept sparse input.
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)

        ]
    )

    models = {
        "Logistic_Regression": LogisticRegression(max_iter=1000),
        "Decision_Tree": DecisionTreeClassifier(),
        "KNN": KNeighborsClassifier(),
        "Naive_Bayes": GaussianNB(),
        "Random_Forest": RandomForestClassifier(),
        "XGBoost": XGBClassifier(eval_metric="logloss")
    }

    results = {}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    for name, model in models.items():

        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", model)
        ])

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)

        results[name] = {
            "Accuracy": accuracy_score(y_test, y_pred),
            # "Precision": precision_score(y_test, y_pred, pos_label=">50K"),
            "Precision": precision_score(y_test, y_pred, pos_label=1),

            "Recall": recall_score(y_test, y_pred, pos_label=1),
            "F1": f1_score(y_test, y_pred, pos_label=1),
            "MCC": matthews_corrcoef(y_test, y_pred),
        }

        try:
            y_prob = pipeline.predict_proba(X_test)[:, 1]
            # results[name]["AUC"] = roc_auc_score(
            #     y_test.map({"<=50K": 0, ">50K": 1}),
            #     y_prob
            # )
            results[name]["AUC"] = roc_auc_score(y_test, y_prob)
        except:
            results[name]["AUC"] = "N/A"

        joblib.dump(pipeline, os.path.join(base_dir, f"{name}.pkl"))

    print(results)


if __name__ == "__main__":
    train_models()
