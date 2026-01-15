# Titanic Survival Prediction: From Baseline to Optimization

## Project Overview
This project demonstrates the complete evolution of a machine learning pipeline using **XGBoost** to predict passenger survival on the Titanic. 

I uploaded two versions of the code to show the difference between a raw baseline model and an optimized model with feature engineering.

## Files Description

### 1. `run_model.py` (The Baseline)
* **Goal:** Establish a Minimum Viable Product (MVP).
* **Features Used:** Raw Pclass, Sex, Age, SibSp, Parch, Fare.
* **Method:** Basic data cleaning (median imputation) + Default XGBoost parameters.
* **Performance:** Achieved **~80% accuracy**.
* **Limitation:** Missed hidden social patterns in text data (Names).

### 2. `run_model_updated.py` (The Optimized Version)
* **Goal:** Break the performance bottleneck through deep feature mining.
* **Key Improvements:**
    * **Feature Engineering:** Extracted **Titles** (e.g., Mr, Miss, Master) from names and created a **FamilySize** feature. This revealed that social status and family structure are critical survival factors.
    * **Hyperparameter Tuning:** Implemented **GridSearchCV** with 5-fold cross-validation to automatically find the best tree depth and learning rate.
* **Performance:** Improved accuracy to **~82-84%** (Top tier stability).

## Key Learnings
Comparing the two models confirms that **Feature Engineering** often yields higher ROI (Return on Investment) than model complexity alone. While the basic XGBoost is powerful, "feeding" it with socially meaningful features (like Titles) significantly improves its predictive power.

## Tech Stack
* **Python**
* **Pandas** (Data Manipulation)
* **XGBoost** (Gradient Boosting Decision Trees)
* **Scikit-Learn** (GridSearch, Preprocessing)

## How to Run
1.  Install dependencies:
    ```bash
    pip install pandas xgboost scikit-learn
    ```
2.  Run the baseline model:
    ```bash
    python run_model.py
    ```
3.  Run the optimized model:
    ```bash
    python run_model_updated.py
    ```
