"""Preprocessing for experiment cli, exported by ML-BenchX.

Runnable as-is. It reproduces exactly the pipeline whose decisions were shown to you in
the app: the same columns, the same transformers, the same seed, and the same ordering —
split first, then fit on the training rows only.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

CSV_PATH = '/Users/rohit/Downloads/student_dropout_dataset_v3.csv'
TARGET = 'Dropout'
TASK_TYPE = 'classification'
TEST_SIZE = 0.2
SEED = 42

# Columns dropped before modelling, and why.
DROPPED = {
    'Student_ID': 'row identifier',
}

NUMERIC = [
    'Age',
    'Family_Income',
    'Study_Hours_per_Day',
    'Attendance_Rate',
    'Assignment_Delay_Days',
    'Travel_Time_Minutes',
    'Stress_Index',
    'GPA',
    'Semester_GPA',
    'CGPA',
]
BINARY = [
    'Gender',
    'Internet_Access',
    'Part_Time_Job',
    'Scholarship',
]
LOW_CARD = [
    'Semester',
    'Department',
    'Parental_Education',
]
HIGH_CARD = []


def build_pipeline() -> ColumnTransformer:
    """The same ColumnTransformer the app built for this experiment."""
    blocks = []
    blocks.append(("numeric", Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
    ]), NUMERIC))
    blocks.append(("binary", Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
        ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ]), BINARY))
    blocks.append(("low_card", Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]), LOW_CARD))
    return ColumnTransformer(
        transformers=blocks,
        remainder="drop",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    y = df[TARGET]
    X = df.drop(columns=[TARGET])
    X = X.drop(columns=[c for c in DROPPED if c in X.columns])

    stratify = y if TASK_TYPE == "classification" else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, shuffle=True, stratify=stratify
    )

    pipeline = build_pipeline()
    # --- Fitted on the TRAINING partition only. Everything the transformers learn
    # --- (medians, means, categories, frequencies) comes from these rows alone.
    X_train_out = pipeline.fit_transform(X_train)
    X_test_out = pipeline.transform(X_test)

    print("train:", X_train_out.shape, " test:", X_test_out.shape)
    print("features:", list(X_train_out.columns))


if __name__ == "__main__":
    main()
