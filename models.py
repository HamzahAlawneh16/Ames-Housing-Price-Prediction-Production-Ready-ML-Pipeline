from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

def get_models_and_params():
    """
    Returns a dictionary of models and their hyperparameter grids for RandomizedSearchCV.
    """
    
    models_config = {
         'XGBoost': {
            'model': XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1),
            'params': {
                'model__n_estimators': [100, 300, 500],
                'model__learning_rate': [0.01, 0.05, 0.1],
                'model__max_depth': [3, 5, 7],
                'model__subsample': [0.7, 0.8, 0.9],
                'model__colsample_bytree': [0.7, 0.8, 0.9]
            }
        },
        'RandomForest': {
            'model': RandomForestRegressor(random_state=42, n_jobs=-1),
            'params': {
                'model__n_estimators': [100, 200, 300],
                'model__max_depth': [None, 10, 20, 30],
                'model__min_samples_split': [2, 5, 10],
                'model__min_samples_leaf': [1, 2, 4]
            }
        },
        'Ridge': {
            'model': Ridge(),
            'params': {
                'model__alpha': [0.1, 1.0, 10.0, 100.0]
            }
        }
    }
    
    return models_config
