import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_selector

# Import local modules
from src.data_loader import load_data
from src.preprocessing import FeatureEngineer, get_preprocessor
from src.models import get_models_and_params
from src.evaluation import evaluate_model, plot_feature_importance

def main():
    # 1. Load Data
    X, y = load_data()
    
    # 2. Initial Feature Engineering 
    # note: We fit transform here to get the columns populated for the column selector.
    # In a strict pipeline, we would put this inside, but to dynamically determine numeric/cat columns
    # for the ColumnTransformer WITHOUT hardcoding, it's easier to apply this first or use a selector that works on the fly.
    # However, Scikit-learn Pipeline with ColumnTransformer + ColumnSelector works lazily at fit time.
    # So we CAN put FeatureEngineer in the main pipeline *before* Preprocessor.
    # BUT, Preprocessor (ColumnTransformer) takes 'columns' argument.
    # If we use make_column_selector(dtype_include=...), it applies to the input of the transformer (which would be FE output).
    # This is the robust "Production" way.
    
    # Let's Split Data first
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Setup Preprocessing
    # We define selectors
    numeric_selector = make_column_selector(dtype_include=np.number)
    categorical_selector = make_column_selector(dtype_include=object)
    
    # Get preprocessor using selectors
    preprocessor = get_preprocessor(numeric_selector, categorical_selector)
    
    # Feature Engineering Step
    fe = FeatureEngineer()
    
    # 4. Model Training & Tuning
    models_config = get_models_and_params()
    results = {}
    
    best_model_overall = None
    best_score = float('-inf') # R2 maximization
    
    for name, config in models_config.items():
        print(f"\nTraining {name}...")
        
        # Construct Full Pipeline
        # FE -> Preprocessor -> Model
        pipeline = Pipeline(steps=[
            ('fe', fe),
            ('preprocessor', preprocessor),
            ('model', config['model'])
        ])
        
        # Tuning
        # Note: params keys in models.py were like 'model__n_estimators'.
        # Since 'model' is the step name in THIS pipeline, it matches.
        search = RandomizedSearchCV(
            pipeline,
            param_distributions=config['params'],
            n_iter=10, # Limiting to 10 for speed in this demo
            cv=3,
            scoring='neg_root_mean_squared_error',
            verbose=1,
            random_state=42,
            n_jobs=-1
        )
        
        search.fit(X_train, y_train)
        
        print(f"Best params for {name}: {search.best_params_}")
        best_estimator = search.best_estimator_
        
        # Evaluate
        metrics = evaluate_model(best_estimator, X_test, y_test, name)
        results[name] = metrics
        
        if metrics['R2'] > best_score:
            best_score = metrics['R2']
            best_model_overall = best_estimator
            best_model_name = name

    # 5. Explainability (Feature Importance)
    # We need to extract feature names from the preprocessor to map them back
    # This is tricky with pipelines.
    print(f"\nGeneratring Feature Importance for Best Model: {best_model_name}")
    
    # Retrieve feature names
    # Fit the FE and Preprocessor separately on X_train just to get names
    # (The pipeline has already fitted them, we can access them from best_estimator)
    
    try:
        # Step 1: Feature Engineer names
        # Our custom FE preserves pandas, so it returns DF if we configured it (but standard transformer usually returns numpy array unless set_output is used).
        # We didn't set_output(transform='pandas') in preprocessing.py. Let's assume standard behavior.
        # Ideally we use get_feature_names_out().
        
        # Access preprocessor from pipeline
        prep_step = best_model_overall.named_steps['preprocessor']
        
        # For sklearn < 1.1 get_feature_names_out might be get_feature_names.
        # Assuming modern sklearn.
        feature_names = prep_step.get_feature_names_out()
        
        # Clean up names (e.g., "num__YearBuilt")
        clean_names = [f.split('__')[-1] for f in feature_names]
        
        plot_feature_importance(best_model_overall, clean_names, filename='feature_importance.png')
        
    except Exception as e:
        print(f"Could not extract feature names for plotting: {e}")
        # Fallback to index based plotting inside the function
        plot_feature_importance(best_model_overall, None, filename='feature_importance.png')

if __name__ == "__main__":
    main()
