# SmartBuild Manufacturing - Data Analysis Pipeline

**IMPORT Libraries, Packages**

# Data Manipulation
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import zscore

# Visualisation
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing
from sklearn.preprocessing import LabelEncoder, minmax_scale, PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Model Selection & Evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, balanced_accuracy_score, f1_score,
    confusion_matrix, roc_curve, roc_auc_score)

# Models
import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor, plot_importance

# Feature Selection
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

import os

**DATA IMPORT**
  
path = input("Enter input file path:")
file = "SmartBuild_Manufacturing.csv"
path_file = os.path.join(path, file)
data = pd.read_csv(path_file)

print(data.head(2))
print(data.dtypes)


**DATA CLEANING**

# Check for missing values 
print(data.isnull().sum())

# Drop duplicate and uninformative columns 
# 'weight_in_g' is a duplicate of 'weight_in_kg' 

print("Correlation between weight columns:", data['weight_in_kg'].corr(data['weight_in_g']).round(2))
data = data.drop(columns = ['id', 'weight_in_g'])

# Visualise distributions 
data.hist(alpha = 0.7, bins = 50, figsize = (14, 10))
plt.subplots_adjust(hspace = 0.6)
plt.suptitle("Feature Distributions", fontsize = 14)
plt.show()

# Remove outliers using Z-score (threshold = 3) 
outlier_cols = ['width', 'weight_in_kg', 'nicesness']
z_cols = [f"{col}_z" for col in outlier_cols]
data[z_cols] = zscore(data[outlier_cols])

outliers = (
    (data['width_z'].abs() > 3) |
    (data['weight_in_kg_z'].abs() > 3) |
    (data['nicesness_z'].abs() > 3))

data = data[~outliers].drop(columns=z_cols)

# Normalise numeric columns with MinMax scaling
numeric_cols = data.select_dtypes(np.number).columns
data[numeric_cols] = minmax_scale(data[numeric_cols])

**Feature Engineering**

# Encode boolean columns 
for col in ['error', 'multideminsionality']:
    data[col] = data[col].replace({'yes': 1, 'no': 0})

# Label encode categorical columns 
le_ionization = LabelEncoder()
le_flux = LabelEncoder()
data['ionizationclass'] = le_ionization.fit_transform(data['ionizationclass'])
data['fluxcompensation'] = le_flux.fit_transform(data['fluxcompensation'])


# FEATURE SPLIT: RAW MATERIAL vs OUTPUT

input_data = data[['width', 'height', 'ionizationclass', 'fluxcompensation',
                      'pressure', 'karma', 'modulation', 'weight_in_kg']]

output_data = data[['error', 'error_type', 'quality', 'reflectionscore',
                     'distortion', 'nicesness', 'multideminsionality']]

# Correlation Matrix: Raw Material Features 
matrix = input_data.corr().round(2)
mask = np.triu(np.ones_like(matrix, dtype=bool))
sns.heatmap(matrix, annot=True, vmax=1, vmin=-1, cmap='vlag', mask=mask)
plt.title("Correlation Matrix - Input Data Features")
plt.show()
# Result: High correlations between weight_in_kg, width, and height

# Prepare data 
model_data = pd.concat([input_data, output_data], axis = 1)
model_data = model_data.drop(columns = ['error', 'error_type', 'quality',
                                       'reflectionscore', 'distortion',
                                       'nicesness', 'multideminsionality'])

# One-hot encode categorical columns
ionization_dummies = pd.get_dummies(model_data['ionizationclass'])
flux_dummies = pd.get_dummies(model_data['fluxcompensation'])
model_data = model_data.drop(columns=['ionizationclass', 'fluxcompensation'])
model_data = pd.concat([model_data, ionization_dummies, flux_dummies], axis=1)

# Correlation matrix for model data
plt.figure(figsize = (12, 8))
sns.heatmap(model_data.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix - Manufacturing Data")
plt.show()

**MODEL 1 — POLYNOMIAL REGRESSION (Predicting Weight)**
# Train/Test Split

x = model_data[['width']]
y = model_data['weight_in_kg']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

# Linear Regression 
linear_model = LinearRegression()
linear_model.fit(x_train, y_train)
y_pred_linear = linear_model.predict(x_test)

# Polynomial Regression (Degree 3) 
poly_converter = PolynomialFeatures(degree = 3, include_bias = False)
x_train_poly = poly_converter.fit_transform(x_train)
x_test_poly = poly_converter.transform(x_test)

poly_model = LinearRegression()
poly_model.fit(x_train_poly, y_train)
y_pred_poly = poly_model.predict(x_test_poly)

# Visualise: Actual vs Predicted 
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, y_pred, color, label, title in zip(
    axes,
    [y_pred_linear, y_pred_poly],
    ['red', 'yellow'],
    ['Predicted - Linear', 'Predicted - Polynomial'],
    ['Linear Model: Actual vs Predicted', 'Polynomial Model: Actual vs Predicted']):
    ax.scatter(x_test['width'], y_test, color='blue', alpha=0.5, label='Actual')
    ax.scatter(x_test['width'], y_pred, color=color, alpha=0.5,
               label=label, edgecolors='black')
    ax.set_title(title)
    ax.set_xlabel('Width')
    ax.set_ylabel('Weight in kg')
    ax.legend()
    ax.set_facecolor('lightgray')
    ax.grid(True)

plt.tight_layout()
plt.show()

# Residual Analysis 
residuals_linear = y_test - y_pred_linear
residuals_poly = y_test - y_pred_poly

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, residuals, color, title in zip(
    axes,
    [residuals_linear, residuals_poly],
    ['blue', 'green'],
    ['Residuals - Linear Model', 'Residuals - Polynomial Model']):
    ax.scatter(y_pred_linear if color == 'blue' else y_pred_poly,
               residuals, color=color, alpha=0.5, edgecolors='black')
    ax.axhline(y=0, color='red', linestyle='--')
    ax.set_title(title)
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Residuals')
    ax.set_facecolor('lightgray')
    ax.grid(True)

plt.tight_layout()
plt.show()

# Model Evaluation 
for label, y_pred, mse in [
    ("Linear Regression", y_pred_linear, mean_squared_error(y_test, y_pred_linear)),
    ("Polynomial Regression", y_pred_poly, mean_squared_error(y_test, y_pred_poly))]:
    mae = mean_absolute_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)
    print(f"\n--- {label} ---")
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {np.sqrt(mse):.4f}")
    print(f"MAE  : {mae:.4f}")
    print(f"R²   : {r2:.5f}")

# Polynomial equation
c3, c2, c1 = poly_model.coef_
intercept = poly_model.intercept_
print(f"\nPolynomial Function: y = {round(c3,4)}x³ + {round(c2,4)}x² + {round(c1,4)}x + {round(intercept,4)}")

**MODEL 2 — XGBOOST CLASSIFIER (Predicting Error)**
# Prepare Data 
clf_data = pd.concat([input_data, output_data], axis = 1)
clf_data = clf_data.drop(columns = ['error_type', 'weight_in_kg', 'quality',
                                   'reflectionscore', 'distortion',
                                   'nicesness', 'multideminsionality'])

x = clf_data.drop(columns = ['error'])
y = LabelEncoder().fit_transform(clf_data['error'])

# Train/Test Split 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)

# Train XGBoost Classifier 
model = XGBClassifier(max_depth = 4, random_state = 42)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
y_pred_proba = model.predict_proba(x_test)[:, 1]

# Model Evaluation 
print("\n--- XGBoost Classifier ---")
print(f"Accuracy          : {accuracy_score(y_test, y_pred):.3f}")
print(f"Balanced Accuracy : {balanced_accuracy_score(y_test, y_pred):.3f}")
print(f"F1 Score          : {f1_score(y_test, y_pred):.2f}")

# Feature Importance Plot (replaces Decision Tree visual) 
fig, ax = plt.subplots(figsize = (10, 6))
plot_importance(model, ax = ax, importance_type = 'weight')
plt.title("XGBoost Feature Importance")
plt.tight_layout()
plt.show()

# Confusion Matrix 
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize = (10, 8))
ax = sns.heatmap(conf_matrix, annot = True, fmt = 'd', cmap = 'viridis',
                 cbar = False, annot_kws = {"size": 16})
ax.set_xticklabels(['Negative (0)', 'Positive (1)'], fontsize = 14)
ax.set_yticklabels(['Negative (0)', 'Positive (1)'], fontsize = 14, rotation = 0)
plt.title('Confusion Matrix', fontsize = 20)
plt.ylabel('True Label', fontsize = 16)
plt.xlabel('Predicted Label', fontsize = 16)
plt.tight_layout()
plt.show()

# ROC Curve 
fpr_vals, tpr_vals, _ = roc_curve(y_test, y_pred_proba)
auc_score = roc_auc_score(y_test, y_pred_proba)

plt.figure(figsize = (10, 8))
plt.plot(fpr_vals, tpr_vals, color = 'darkorange', lw = 2,
         label = f'ROC Curve (AUC = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], color = 'navy', lw = 2, linestyle = '--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize = 16)
plt.ylabel('True Positive Rate', fontsize = 16)
plt.title('Receiver Operating Characteristic (ROC)', fontsize = 20)
plt.legend(loc = 'lower right', fontsize = 16)
plt.gca().set_facecolor('lightgray')
plt.grid(True)
plt.tight_layout()
plt.show()

# TPR & FPR Table 
tn, fp, fn, tp = conf_matrix.ravel()
tpr_val = round(tp / (tp + fn), 2)
fpr_val = round(fp / (fp + tn), 2)

results_df = pd.DataFrame({
    "Metric": ["True Positive Rate (TPR)", "False Positive Rate (FPR)"],
    "Value":  [tpr_val, fpr_val]})

fig, ax = plt.subplots(figsize = (5, 2))
ax.axis('tight')
ax.axis('off')
ax.table(cellText = results_df.values, colLabels = results_df.columns,
         cellLoc = 'center', loc = 'center',
         colColours = ["palegreen", "paleturquoise"])
plt.title("TPR and FPR", fontsize = 16, color = "darkblue")
plt.tight_layout()
plt.show()








