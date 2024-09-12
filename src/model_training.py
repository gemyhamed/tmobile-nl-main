import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from src.data_preprocessing import build_preprocessor, FeatureEngineering
from src.utils import plot_original_feature_importance
import os


# Custom function to drop rows with more than 2 missing values (used only during training)
def drop_rows_with_missing_values(df, threshold=2):
    print(f"Dropping rows with more than {threshold} missing values...")
    return df.dropna(thresh=df.shape[1] - threshold)


# Train the model and save the pipeline as a .pkl file
def train_model(data):

    Target = "product02"
    data = drop_rows_with_missing_values(data, threshold=2)
    X = data.drop(columns=[Target])
    y = data[Target]

    print("Splitting the data into training and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=60
    )

    print(
        "Building the full pipeline (feature engineering, preprocessing, and classifier)..."
    )
    # Build the full pipeline (feature engineering + preprocessing + classifier)
    preprocessor = build_preprocessor()

    full_pipeline = Pipeline(
        steps=[
            ("feature_engineering", FeatureEngineering()),  # Custom feature engineering
            ("preprocessor", preprocessor),  # Preprocessing step
            (
                "classifier",
                RandomForestClassifier(random_state=60, class_weight="balanced"),
            ),  # Classifier
        ]
    )

    # Reduce the hyperparameter grid for faster search
    param_grid = {
        "classifier__n_estimators": [40, 50],
        "classifier__max_depth": [15, 20],
        "classifier__min_samples_split": [1, 2, 3],
    }

    print("Starting hyperparameter tuning with GridSearchCV...")
    # Perform grid search with cross-validation
    grid_search = GridSearchCV(
        estimator=full_pipeline,
        param_grid=param_grid,
        cv=3,
        scoring="f1_weighted",
        n_jobs=-1,
    )

    print("Fitting the model on the training data...")
    grid_search.fit(X_train, y_train)

    # Get the best pipeline (full pipeline with best hyperparameters)
    best_pipeline = grid_search.best_estimator_
    best_params = grid_search.best_params_

    print(f"Best hyperparameters found: {best_params}")

    # Generate classification report
    y_pred = best_pipeline.predict(X_test)
    print("Training Classification Report (Test Set):")
    print(classification_report(y_test, y_pred))

    # Save the entire pipeline (feature engineering, preprocessing, and model) as .pkl
    model_dir = "models"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    print("Saving the trained pipeline to 'models/train_pipeline.pkl'...")
    joblib.dump(best_pipeline, os.path.join(model_dir, "train_pipeline.pkl"))

    print("Plotting Feature Importance..")
    plot_original_feature_importance(best_pipeline)

    return best_pipeline
