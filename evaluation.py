import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_model(model, X_test, y_test, model_name):
    """
    Calculates and prints RMSE, MAE, and R^2.
    """
    y_pred = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"--- {model_name} Performance ---")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R^2:  {r2:.4f}")
    print("-" * 30)
    
    return {'RMSE': rmse, 'MAE': mae, 'R2': r2}

def plot_feature_importance(model, feature_names, filename='feature_importance.png'):
    """
    Plots feature importance for tree-based models.
    """
    # Check if model has feature_importances_ (RandomForest, XGBoost)
    # If the model is a Pipeline, we need to access the final step
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model.named_steps['model'], 'feature_importances_'):
        importances = model.named_steps['model'].feature_importances_
    elif hasattr(model.named_steps['model'], 'coef_'):
        # For linear models like Ridge
        importances = np.abs(model.named_steps['model'].coef_)
    else:
        print("Model does not provide feature importances.")
        return

    # Create DataFrame
    # If feature_names is None, we can't label effectively
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(len(importances))]
        
    df_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    df_imp = df_imp.sort_values('Importance', ascending=False).head(20) # Top 20
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=df_imp)
    plt.title('Top 20 Feature Importances')
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Feature importance plot saved to {filename}")
