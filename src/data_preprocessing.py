from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Apply the custom feature engineering steps
        X = self._extract_date_features(X)
        X = self._create_income_buckets(X)
        X = self._create_age_buckets(X)
        return X

    def _extract_date_features(self, df):
        if "lastVisit" in df.columns:
            df["lastVisit"] = pd.to_datetime(df["lastVisit"], errors="coerce")
            df["visit_year"] = df["lastVisit"].dt.year
            df["visit_month"] = df["lastVisit"].dt.month
            df["visit_day"] = df["lastVisit"].dt.day
            df["visit_dayofweek"] = df["lastVisit"].dt.dayofweek
            df = df.drop(columns=["lastVisit"])
        return df

    def _create_income_buckets(self, df):
        if "income" in df.columns:
            df["income_bucket"] = pd.qcut(
                df["income"], q=4, labels=["Low", "Medium", "High", "Very High"]
            )
        return df

    def _create_age_buckets(self, df):
        if "age" in df.columns:
            bins = [0, 18, 30, 60, 100]
            labels = ["Minor", "Youth", "Middle-aged", "Senior"]
            df["age_bucket"] = pd.cut(
                df["age"], bins=bins, labels=labels, include_lowest=True
            )
        return df


def build_preprocessor():
    # Numerical and categorical columns
    numerical_columns = ["income", "age", "var1"]
    categorical_columns = [
        "gender",
        "house_type",
        "income_bucket",
        "age_bucket",
    ]

    # Preprocessing for numerical columns
    numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),  # not actually needed for a tree model
        ]
    )

    # Preprocessing for categorical columns
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
