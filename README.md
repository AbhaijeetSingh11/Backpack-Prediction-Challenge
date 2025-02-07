# Kaggle Problem - Regression Model with LightGBM

## Overview
This project builds a regression model to predict target values using CatBoost, LightGBM, and XGBoost. It preprocesses categorical features, optimizes model hyperparameters, and evaluates performance.

## Dataset
- The dataset contains both numerical and categorical features.
- Key categorical features: `Brand`, `Material`, `Size`, `Laptop Compartment`, `Waterproof`, `Style`, `Color`.
- The target variable is a continuous numeric value.

## Dependencies
Ensure you have the required libraries installed:
```sh
pip install pandas numpy scikit-learn lightgbm
```

## Preprocessing Steps
1. Convert categorical features to the appropriate format:
   - **LightGBM** requires categorical features to be converted to `category` dtype.
2. Split the dataset into training and testing sets.

## Model Implementation
### **LightGBM Regressor**
```python
from lightgbm import LGBMRegressor
X_train[cat_features] = X_train[cat_features].astype("category")
lgbm = LGBMRegressor(n_estimators=1000, learning_rate=0.1, max_depth=3, boosting_type='gbdt', device="gpu")
lgbm.fit(X_train, y_train, categorical_feature=cat_features)
y_pred = lgbm.predict(X_test)
```

## Common Issues & Fixes
### **1. "No OpenCL device found" (LightGBM GPU issue)**
- Ensure you have an NVIDIA/AMD GPU with OpenCL installed.
- Run LightGBM in CPU mode if GPU is unavailable:
  ```python
  lgbm = LGBMRegressor(device="cpu")
  ```

### **2. "pandas dtypes must be int, float or bool" (LightGBM & XGBoost categorical issue)**
- Convert categorical columns:
  ```python
  for col in cat_features:
      X_train[col] = X_train[col].astype("category")
  ```

## Evaluation
- Use **Mean Squared Error (MSE)** to compare models:
```python
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
print("MSE:", mse)
```

## Future Improvements
- Hyperparameter tuning with GridSearchCV or Optuna.
- Feature engineering to improve model performance.
- Experimenting with deep learning models like TensorFlow/Keras.

## Author
- Developed by Abhaijeet Singh

