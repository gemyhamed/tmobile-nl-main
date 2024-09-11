import numpy as np


def check_for_invalid_values(df):
    numerical_columns = df.select_dtypes(include=[np.number])
    if np.any(np.isnan(numerical_columns)) or np.any(np.isinf(numerical_columns)):
        raise ValueError("NaNs or infinite values found in the numerical data.")
