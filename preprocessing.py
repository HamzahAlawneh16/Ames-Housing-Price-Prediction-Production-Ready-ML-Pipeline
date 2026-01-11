import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.preprocessing import TargetEncoder

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to add 'House_Age' and 'Total_SF'.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        # Ensure correct types for calculation
        # YearBuilt and YrSold might be read as objects, so feature engineer logic needs to be robust
        # But usually in Ames they are numeric.
        
        # House Age
        if 'YearBuilt' in X.columns and 'YrSold' in X.columns:
            X['House_Age'] = X['YrSold'] - X['YearBuilt']
            # handle potential negative age if data is noisy
            X['House_Age'] = X['House_Age'].apply(lambda x: max(0, x))
        
        # Total Square Footage
        # Summing up basement, 1st floor, and 2nd floor
        # Check column names carefully as they valid in Ames dataset
        sf_cols = ['TotalBsmtSF', '1stFlrSF', '2ndFlrSF']
        available_cols = [c for c in sf_cols if c in X.columns]
        
        if available_cols:
            X['Total_SF'] = X[available_cols].sum(axis=1)
            
        return X

def get_preprocessor(numeric_features, categorical_features):
    """
    Returns a ColumnTransformer for preprocessing.
    
    Args:
        numeric_features (list): List of numeric column names.
        categorical_features (list): List of categorical column names.
    """
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Using OneHotEncoder for categorical features. 
    # handle_unknown='ignore' is crucial for production pipelines.
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop' # Drop columns not specified (like original date columns if not used)
    )
    
    return preprocessor
