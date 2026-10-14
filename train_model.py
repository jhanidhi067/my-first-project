
"""
Student Success AI
Machine learning training pipeline.

Uses fictional synthetic data for development.
Predictions are experimental and not validated for real students.
"""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# Project folders
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

for folder in [DATA_DIR, MODEL_DIR, RESULTS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "study_hours",
    "attendance",
    "assignment_completion",
    "previous_score",
    "sleep_hours",
]


def create_dataset(number_of_students=1200):
    """Generate fictional data for developing the application."""
    rng = np.random.default_rng(42)

    study_hours = rng.uniform(0.5, 8, number_of_students)
    attendance = rng.uniform(40, 100, number_of_students)
    assignments = rng.uniform(30, 100, number_of_students)
    previous_score = rng.uniform(25, 100, number_of_students)
    sleep_hours = rng.uniform(4, 10, number_of_students)

    # Illustrative rule for creating a synthetic target.
    risk_score = (
        2.0
        - 0.28 * study_hours
        - 0.035 * (attendance - 70)
        - 0.025 * (assignments - 65)
        - 0.035 * (previous_score - 60)
        - 0.08 * (sleep_hours - 7)
        + rng.normal(0, 0.9, number_of_students)
    )

    return pd.DataFrame({
        "study_hours": study_hours.round(2),
        "attendance": attendance.round(2),
        "assignment_completion": assignments.round(2),
        "previous_score": previous_score.round(2),
        "sleep_hours": sleep_hours.round(2),
        "at_risk": (risk_score > 0).astype(int),
    })


def build_models():
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=1000, random_state=42
            )),
        ]),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6,
            min_samples_leaf=10,
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=8,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        ),
    }


def main():
    print("STUDENT SUCCESS AI: MODEL TRAINING")

    df = create_dataset()
    df.to_csv(DATA_DIR / "student_data.csv", index=False)

    print("\nDataset shape:", df.shape)
    print("Missing values:", int(df.isnull().sum().sum()))
    print("\nTarget counts:")
    print(df["at_risk"].value_counts().rename(
        index={0: "Not flagged", 1: "Support flag"}
    ))

    X = df[FEATURES]
    y = df["at_risk"]

    # Reserve a final test set that is not used for model selection.
    X_development, X_test, y_development, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    X_train, X_validation, y_train, y_validation = train_test_split(
        X_development,
        y_development,
        test_size=0.25,
        random_state=42,
        stratify=y_development,
    )

    # Compare models on validation data, not the final test set.
    models = build_models()
    validation_results = []
    fitted_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        prediction = model.predict(X_validation)

        score = f1_score(
            y_validation, prediction, zero_division=0
        )

        validation_results.append({
            "model": name,
            "validation_f1": round(float(score), 4),
        })
        fitted_models[name] = model

    validation_df = pd.DataFrame(validation_results).sort_values(
        "validation_f1", ascending=False
    )

    best_name = validation_df.iloc[0]["model"]
    print("\nValidation comparison:")
    print(validation_df.to_string(index=False))
    print("\nSelected model:", best_name)

    # Refit the selected model on all development data.
    best_model = build_models()[best_name]
    best_model.fit(X_development, y_development)

    # Evaluate the selected model once on the held-out test set.
    prediction = best_model.predict(X_test)

    test_metrics = {
        "model": best_name,
        "accuracy": round(float(
            accuracy_score(y_test, prediction)
        ), 4),
        "precision": round(float(
            precision_score(y_test, prediction, zero_division=0)
        ), 4),
        "recall": round(float(
            recall_score(y_test, prediction, zero_division=0)
        ), 4),
        "f1_score": round(float(
            f1_score(y_test, prediction, zero_division=0)
        ), 4),
    }

    joblib.dump(best_model, MODEL_DIR / "best_model.joblib")

    # Save test evaluation for the dashboard.
    pd.DataFrame([test_metrics]).to_csv(
        RESULTS_DIR / "test_metrics.csv", index=False
    )

    cm = confusion_matrix(y_test, prediction, labels=[0, 1])
    pd.DataFrame(
        cm,
        index=["Actual_Not_Flagged", "Actual_Support_Flag"],
        columns=["Predicted_Not_Flagged", "Predicted_Support_Flag"],
    ).to_csv(RESULTS_DIR / "confusion_matrix.csv")

    report = classification_report(
        y_test,
        prediction,
        labels=[0, 1],
        target_names=["Not flagged", "Support flag"],
        output_dict=True,
        zero_division=0,
    )

    with open(
        RESULTS_DIR / "classification_report.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(report, file, indent=4)

    metadata = {
        "project": "Student Success AI",
        "selected_model": best_name,
        "features": FEATURES,
        "target": {
            "0": "Not flagged",
            "1": "May benefit from additional support",
        },
        "dataset_type": "Synthetic fictional data",
        "test_metrics": test_metrics,
        "warning": (
            "Experimental prototype. Synthetic-data results do not "
            "establish real-world accuracy or fairness."
        ),
    }

    with open(
        RESULTS_DIR / "model_metadata.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metadata, file, indent=4)

    print("\nFinal held-out test results:")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value}")

    print("\nConfusion matrix:")
    print(cm)

    print("\nTraining completed successfully.")
    print("Saved model:", MODEL_DIR / "best_model.joblib")


if __name__ == "__main__":
    main()
