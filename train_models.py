"""Train two simple models from local Kaggle CSV files."""
import csv
import json
from pathlib import Path

import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

BASE = Path(__file__).resolve().parent


def main():
    # PART 1: Predict the tip using only the restaurant bill.
    with (BASE / "data/tips.csv").open() as file:
        tips = list(csv.DictReader(file))
    X = [[float(row["total_bill"])] for row in tips]
    y = [float(row["tip"]) for row in tips]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    regression = LinearRegression()
    regression.fit(X_train, y_train)
    r2 = r2_score(y_test, regression.predict(X_test))
    (BASE / "models").mkdir(exist_ok=True)
    joblib.dump(regression, BASE / "models/regression.joblib")
    regression_train_rows, regression_test_rows = len(X_train), len(X_test)

    # PART 2: Identify the flower using only its petal length and width.
    with (BASE / "data/Iris.csv").open() as file:
        flowers = list(csv.DictReader(file))
    X = [[float(row["PetalLengthCm"]), float(row["PetalWidthCm"])] for row in flowers]
    y = [row["Species"] for row in flowers]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    classification = DecisionTreeClassifier(max_depth=2, random_state=42)
    classification.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, classification.predict(X_test))
    joblib.dump(classification, BASE / "models/classification.joblib")

    metrics = {
        "regression_r2": float(r2),
        "classification_accuracy": float(accuracy),
        "regression_train_rows": regression_train_rows,
        "regression_test_rows": regression_test_rows,
        "classification_train_rows": len(X_train),
        "classification_test_rows": len(X_test),
        "regression_intercept": float(regression.intercept_),
        "regression_slope": float(regression.coef_[0])
    }
    (BASE / "models/metrics.json").write_text(json.dumps(metrics, indent=2))
    rules = export_text(classification, feature_names=["petal_length", "petal_width"])
    (BASE / "models/tree_rules.txt").write_text(rules)
    print(json.dumps(metrics, indent=2))
    print("Decision tree rules:\n" + rules)


if __name__ == "__main__":
    main()
