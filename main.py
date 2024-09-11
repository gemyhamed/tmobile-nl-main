import pandas as pd
import numpy as np
import argparse
import os
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib


# Function to extract date features from 'lastVisit' column
def extract_date_features(df):
    if "lastVisit" in df.columns:
        df["lastVisit"] = pd.to_datetime(df["lastVisit"], errors="coerce")
        df["visit_year"] = df["lastVisit"].dt.year
        df["visit_month"] = df["lastVisit"].dt.month
        df["visit_day"] = df["lastVisit"].dt.day
        df["visit_dayofweek"] = df["lastVisit"].dt.dayofweek
        df = df.drop(columns=["lastVisit"])
    return df


# Function to create income buckets using quantiles
def create_income_buckets(df):
    if "income" in df.columns:
        # Creating 4 buckets based on quantiles (quartiles)
        df["income_bucket"] = pd.qcut(
            df["income"], q=4, labels=["Low", "Medium", "High", "Very High"]
        )
    return df


# Function to create age buckets
def create_age_buckets(df):
    if "age" in df.columns:
        # Define the age ranges and corresponding labels
        bins = [0, 25, 50, 100]  # Young: 0-25, Middle: 25-50, Senior: 50+
        labels = ["Young", "Middle-aged", "Senior"]
        df["age_bucket"] = pd.cut(
            df["age"], bins=bins, labels=labels, include_lowest=True
        )
    return df


# Apply all feature engineering transformations
def feature_engineering(df):
    df = extract_date_features(df)
    df = create_income_buckets(df)
    df = create_age_buckets(df)
    return df


# Common Preprocessing Pipeline Builder
def build_preprocessor():
    # Separate numerical and categorical columns
    numerical_columns = ["income", "age", "var1"]  # Keep these as numerical
    categorical_columns = [
        "subscriber",
        "gender",
        "house_type",
        "income_bucket",
        "age_bucket",
    ]

    # Preprocessing steps for numerical columns
    numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Preprocessing steps for categorical columns
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # Combine preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_columns),
            ("cat", categorical_transformer, categorical_columns),
        ]
    )

    return preprocessor


# Preprocess the dataset for both training and prediction
def preprocess_data(df, preprocessor):
    # Apply feature engineering before preprocessing
    df = feature_engineering(df)
    X_processed = (
        preprocessor.fit_transform(df)
        if "product02" not in df
        else preprocessor.transform(df)
    )
    return X_processed


# Hyperparameter tuning using GridSearchCV
def tune_hyperparameters(X_train, y_train):
    # RandomForestClassifier with class_weight='balanced' to handle class imbalance
    model = RandomForestClassifier(class_weight="balanced", random_state=42)

    # Define hyperparameter grid
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [10, 20],
        "min_samples_split": [5, 10],
    }

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring="f1_weighted",
        verbose=1,
        n_jobs=-1,
    )

    try:
        grid_search.fit(X_train, y_train)
    except ValueError as e:
        print(f"Error during GridSearchCV: {e}")
        return None, None

    # Return the best model after hyperparameter tuning
    return grid_search.best_estimator_, grid_search.best_params_


# Train the model with hyperparameter tuning and class imbalance handling
def train_model(training_data_path):
    data = pd.read_csv(training_data_path)

    # Remove rows with more than 5 nulls and preprocess data
    data_clean = data[data.isnull().sum(axis=1) <= 5]

    X = data_clean.drop(columns=["product02"])
    y = data_clean["product02"]

    # Build preprocessor and preprocess the data
    preprocessor = build_preprocessor()
    X_processed = preprocess_data(X, preprocessor)

    # Split data into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42
    )

    # Hyperparameter tuning with GridSearchCV
    print("Starting hyperparameter tuning with GridSearchCV...")
    best_model, best_params = tune_hyperparameters(X_train, y_train)

    if best_model is None:
        print("Hyperparameter tuning failed.")
        return

    print(f"Best hyperparameters: {best_params}")

    # Evaluate the best model on the test set
    accuracy = best_model.score(X_test, y_test)
    print(f"Model accuracy after hyperparameter tuning: {accuracy:.4f}")

    # Print classification report
    y_pred = best_model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Save the best model and preprocessor
    joblib.dump(best_model, "model.pkl")
    joblib.dump(preprocessor, "preprocessor.pkl")
    print("Model and preprocessor saved.")


# Predict on new data using the saved model and preprocessor
def predict_on_new_data(new_data_path):
    if not os.path.exists("model.pkl") or not os.path.exists("preprocessor.pkl"):
        print("Model or preprocessor not found. Please train the model first.")
        return

    # Load new data
    new_data = pd.read_csv(new_data_path)

    # Load pre-trained model and preprocessor
    model = joblib.load("model.pkl")
    preprocessor = joblib.load("preprocessor.pkl")

    # Preprocess new data
    X_new_processed = preprocess_data(new_data, preprocessor)

    # Predict with the model
    predictions = model.predict(X_new_processed)
    print("Predictions for new data:")
    print(predictions)


# Main function to control train and predict modes
def main():
    parser = argparse.ArgumentParser(description="Train or predict using the pipeline.")
    parser.add_argument(
        "mode", choices=["train", "predict"], help="Mode to run: 'train' or 'predict'."
    )
    parser.add_argument(
        "--data",
        type=str,
        help="Path to the dataset for training or predicting.",
        required=True,
    )

    args = parser.parse_args()

    if args.mode == "train":
        print(f"Training mode selected. Data: {args.data}")
        train_model(args.data)
    elif args.mode == "predict":
        print(f"Prediction mode selected. Data: {args.data}")
        predict_on_new_data(args.data)


if __name__ == "__main__":
    main()
