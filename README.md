# tmobile-nl

This project is designed to train and predict using a machine learning model that leverages feature engineering, preprocessing, and hyperparameter tuning. The project allows both training and prediction workflows, and it can be containerized using Docker for reproducibility.

### Task 1 
Build a predictive model to predict product02

### Task 2 (Bonus)
Push your source code into Git hosting platform of your choice

### Task 3 (Bonus)
Create a simple Dockerfile for running your scripts

.
├── models/
│   └── train_pipeline.pkl          # Saved trained pipeline (including preprocessing, feature engineering, and the model)
├── src/
│   ├── data_preprocessing.py       # Contains the logic for handling numerical and categorical data preprocessing
│   ├── model_inference.py          # Loads the trained pipeline and performs predictions
│   ├── model_training.py           # Trains the model, applies feature engineering, saves the trained pipeline
│   ├── utils.py                    # Utility functions (can include feature importance or helper functions)
├── .gitignore                      # Lists files and folders to be ignored by Git
├── data-Set.csv                    # Example dataset (for training or prediction)
├── dataReport.html                 # Data exploration or report output
├── Dockerfile                      # Docker configuration file for containerizing the application
├── fakeTest.csv                    # Example test dataset for predictions
├── main.py                         # Main entry point for the project (handles train and predict modes)
├── playground.ipynb                # Jupyter notebook for testing and experimentation
├── predictions.csv                 # Output of the predictions from the model
├── README.md                       # Project README file
├── requirements.txt                # List of Python dependencies for the project
