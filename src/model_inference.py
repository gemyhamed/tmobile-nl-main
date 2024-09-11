import joblib
from sklearn.metrics import classification_report


# Load the trained pipeline
def load_pipeline():
    pipeline = joblib.load("models/train_pipeline.pkl")
    return pipeline


# Predict using the loaded pipeline
def predict_new_data(X_new, pipeline):

    data = X_new.copy()
    predictions = pipeline.predict(X_new)

    data["prediction"] = predictions

    return data, predictions
