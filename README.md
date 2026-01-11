## Ames Housing Price Prediction: Production-Ready ML Pipeline

##  Project Overview

This project implements a robust, end-to-end machine learning pipeline designed to predict residential home prices in Ames, Iowa. Beyond predictive accuracy, the architecture focuses on **Software Engineering best practices** for ML, utilizing Scikit-Learn Pipelines to ensure modularity, reproducibility, and the prevention of data leakage.

##  Problem Statement

Estimating real estate value is a complex task influenced by dozens of variables. Manual appraisal is often subjective and slow. This project solves this by:

*   **Handling High-Dimensional Data**: Systematically processing over 70 features, including numerical, categorical, and temporal data.
*   **Automating Feature Engineering**: Transforming raw data into predictive signals (e.g., converting "Year Built" into "House Age").
*   **Ensuring Pipeline Integrity**: Encapsulating all transformations within a unified `Pipeline` to ensure that preprocessing logic is identical during both training and inference.

##  System Architecture

The repository is structured following **Modular Programming** principles to separate concerns:

### 1\. Preprocessing & Feature Engineering

*   **Custom Transformers**: A specialized `FeatureEngineer` class (inheriting from `BaseEstimator` and `TransformerMixin`) calculates domain-specific metrics like `House_Age` and `Total_SF` (Total Square Footage).
*   **Automated Column Selection**: Uses `make_column_selector` to dynamically identify dtypes, making the pipeline resilient to schema changes.
*   **Scalable Transformations**:
    *   **Numerical**: Median imputation followed by `StandardScaler`.
    *   **Categorical**: Most-frequent imputation and `OneHotEncoder` with `handle_unknown='ignore'` to prevent crashes on unseen labels in production.

### 2\. Model Selection & Hyperparameter Tuning

The system compares three diverse algorithms through `RandomizedSearchCV` to find the optimal balance between bias and variance:

*   **XGBoost Regressor**: Optimized for capturing non-linear relationships and interactions.
*   **Random Forest Regressor**: Used for its robustness and ensemble stability.
*   **Ridge Regression**: Serves as a regularized linear baseline to prevent overfitting.

### 3\. Evaluation & Explainability

*   **Performance Metrics**: Models are evaluated using $RMSE$, $MAE$, and $R^2$ to provide a comprehensive view of error distribution.
*   **XAI (Explainable AI)**: The pipeline extracts and visualizes **Feature Importances**, allowing stakeholders to understand which factors (e.g., Overall Quality, Ground Living Area) most significantly drive property value.

##  Getting Started

1.  **Install Dependencies**:
    
    Bash
    
    ```plaintext
    pip install scikit-learn xgboost pandas numpy matplotlib seaborn
    ```
    
2.  **Run the Pipeline**:
    
    Bash
    
    ```plaintext
    python main.py
    ```
