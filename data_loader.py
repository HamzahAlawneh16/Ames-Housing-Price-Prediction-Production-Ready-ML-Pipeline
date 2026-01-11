import pandas as pd
from sklearn.datasets import fetch_openml

def load_data():
    """
    Fetches the Ames Housing dataset from OpenML.
    
    Returns:
        X (pd.DataFrame): Features
        y (pd.Series): Target variable (SalePrice)
    """
    print("Loading Ames Housing dataset from OpenML...")
    # Ames Housing dataset ID is 42165
    housing = fetch_openml(name="house_prices", as_frame=True, parser='auto')
    
    X = housing.data
    y = housing.target
    
    print(f"Data loaded successfully. Shape: {X.shape}")
    return X, y

if __name__ == "__main__":
    X, y = load_data()
    print(X.head())
