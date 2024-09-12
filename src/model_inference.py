import joblib
from sklearn.metrics import classification_report


# Load the trained pipeline
def load_pipeline():
    pipeline = joblib.load("models/train_pipeline.pkl")
    return pipeline


# Predict using the loaded pipeline
def predict_new_data(X_new, pipeline):
    # Make a copy of the input data
    data = X_new.copy()

    # Drop the 'subscriber' column for prediction
    x_predict = data.drop("product02", axis=1)

    y_true = X_new["product02"].copy()

    # Generate predictions
    predictions = pipeline.predict(x_predict)

    # Add predictions to the dataframe
    data["prediction"] = predictions

    # Generate and print the classification report
    report = classification_report(y_true, predictions)
    print("Classification Report:")
    print(report)

    return data
