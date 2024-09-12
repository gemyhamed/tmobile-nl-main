import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_original_feature_importance(pipeline):
    # Get the feature names from the preprocessor
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    # Get Importance values
    importance_values = pipeline["classifier"].feature_importances_

    # Clean feature names by removing 'num__' and 'cat__'
    clean_feature_names = [name.split("__")[-1] for name in feature_names]
    # Remove OHE suffix from categorical variables
    clean_feature_names = [x.split("_")[0] for x in clean_feature_names]

    # Create a dictionary to accumulate importance values for OHE features
    feature_importance_dict = {}

    for feature, importance in zip(clean_feature_names, importance_values):
        # Aggregate OHE feature importances
        if feature in feature_importance_dict:
            feature_importance_dict[feature] += importance
        else:
            feature_importance_dict[feature] = importance

    # Create a DataFrame for plotting
    feature_importance_df = pd.DataFrame(
        list(feature_importance_dict.items()), columns=["Feature", "Importance"]
    )

    # Sort the features by importance
    feature_importance_df = feature_importance_df.sort_values(
        by="Importance", ascending=False
    )

    # Plot the feature importances
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance_df["Feature"], feature_importance_df["Importance"])
    plt.xlabel("Feature Importance")
    plt.ylabel("Features")
    plt.title("Feature Importance Plot")
    plt.gca().invert_yaxis()  # Invert y-axis to have the most important at the top
    plt.show()


def check_for_invalid_values(df):
    numerical_columns = df.select_dtypes(include=[np.number])
    if np.any(np.isnan(numerical_columns)) or np.any(np.isinf(numerical_columns)):
        raise ValueError("NaNs or infinite values found in the numerical data.")
