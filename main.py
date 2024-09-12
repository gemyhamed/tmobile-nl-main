import argparse
import os
import pandas as pd
from src.model_training import train_model
from src.model_inference import load_pipeline, predict_new_data


def main():

    parser = argparse.ArgumentParser(description="ML Pipeline Project")
    parser.add_argument(
        "mode", choices=["train", "predict"], help="Choose train or predict mode"
    )
    parser.add_argument("--data", type=str, required=True, help="Path to the dataset")
    args = parser.parse_args()

    # Load the dataset
    data = pd.read_csv(args.data)

    ################################ Training mode ################################
    if args.mode == "train":

        # Training mode
        train_model(data)
        print("Model training completed!")
    ################################ Prediction mode ################################
    elif args.mode == "predict":

        # Check if the pipeline pickle file exists
        pipeline_path = "models/train_pipeline.pkl"
        if not os.path.exists(pipeline_path):
            print(
                f"Model file not found at '{pipeline_path}'. Please train the model first using 'train' mode."
            )
            return

        # Load the trained pipeline
        pipeline = load_pipeline()

        # Make predictions
        data = predict_new_data(data, pipeline)

        # Save to disk
        prediction_file_path = "./data/output/predictions.csv"
        data.to_csv(prediction_file_path)
        print(f"Predictions Saved to file {prediction_file_path}!")


if __name__ == "__main__":
    main()
