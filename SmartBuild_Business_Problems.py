#!/usr/bin/env python
# coding: utf-8

# # Import Dataset

# In[250]:


pip install feature_engine


# In[251]:


# Data Manipulation
import pandas as pd
import numpy as np
import statsmodels.api as sm

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn import datasets
import xgboost as xgb
from xgboost import XGBRegressor, plot_tree, XGBClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree  # Added plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score, f1_score, accuracy_score, classification_report, confusion_matrix, balanced_accuracy_score, mean_squared_error, mean_absolute_error, r2_score

# Feature Selection
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

# Feature Engineering
#from feature_engine.encoding import RareLabelEncoder, OrdinalEncoder
from sklearn.preprocessing import LabelEncoder

# Miscellaneous
from sklearn.preprocessing import minmax_scale
from scipy.stats import zscore

# Data Import
data = pd.read_csv("SmartBuild_Manufacturing.csv")


# In[252]:


data.head(2)


# In[253]:


data.dtypes


# # Cleaning

# In[254]:


# Check if there are any NULL values
data.isnull().sum()


# In[255]:


# Use correlation to check if weight columns are duplicates of each other
print(data['weight_in_kg'].corr(data['weight_in_g']).round(2))


# In[256]:


columns_to_drop = ['id', 'weight_in_g']
data = data.drop(columns = columns_to_drop, axis = 1)


# In[257]:


# Visualise data with histogram
data.hist(alpha=0.7, bins=50, figsize=(14, 10))
plt.subplots_adjust(hspace=0.6)
plt.show()


# ## Information gained:
# ### 1) Column **"ID**" could be dropped as it doesn't bring any information.
# ### 2) Column **"weight_in_g"** could be dropped as it's a duplicate for "weight_in_kg" attribute.
# ### 3) Values in the column **"error"** could be replaced from yes: 1, no: 0.
# ### 4) Values in the columns **"ionizationclass"** and **"FluxCompensation"** could be labeled.
# ### 5) Values in columns **"width", "weight_in_kg", "nicesness"** are unusual.

# ## Removing outliers with Z-score

# In[258]:


# Calculate z-scores for relevant columns
data[['width_z', 'weight_in_kg_z', 'nicesness_z']] = zscore(data[['width', 'weight_in_kg', 'nicesness']])

# Identify rows with outliers
outliers = ((data['width_z'] < -3) | (data['width_z'] > 3) |
            (data['weight_in_kg_z'] < -3) | (data['weight_in_kg_z'] > 3) |
            (data['nicesness_z'] < -3) | (data['nicesness_z'] > 3))

# Drop rows with outliers
data = data.drop(data[outliers].index)

# Drop the "z_score" column
data = data.drop(['width_z', 'weight_in_kg_z', 'nicesness_z'], axis=1)


# In[259]:


# MinMax normalise all numeric columns
cols = data.select_dtypes(np.number).columns
data[cols] = minmax_scale(data[cols])


# In[260]:


# Replace boolean yes/no to 1/0 for "error" and "multideminsionality" columns
data['error'].replace(['yes', 'no'], [1,0], inplace=True)
data['multideminsionality'].replace(['yes', 'no'], [1,0], inplace=True)

# Create LabelEncoder instances
le_ionizationclass = LabelEncoder()
le_FluxCompensation = LabelEncoder()

# Fit and transform each column
data['ionizationclass'] = le_ionizationclass.fit_transform(data['ionizationclass'])
data['fluxcompensation'] = le_FluxCompensation.fit_transform(data['fluxcompensation'])


# ## Check for obvious correlations with matrix to know if the properties can be predicted at the input to optimize the production

# In[261]:


raw_material = data[['width', 'height', 'ionizationclass', 'fluxcompensation', 'pressure', 'karma', 'modulation', 'weight_in_kg']]
output_data = data[['error', 'error_type', 'quality', 'reflectionscore', 'distortion', 'nicesness', 'multideminsionality']]

# Prepare mask
matrix = raw_material.corr().round(2)
mask = np.triu(np.ones_like(matrix, dtype=bool))

# Build
sns.heatmap(matrix, annot = True, vmax = 1, vmin = -1, cmap = 'vlag', mask = mask)


# ## Result: high correlations between weight in kg and width / height features

# # Polynomial Model

# In[262]:


#Q1 Multiple Polinomial Model: What will be weight of potential product ?

#concat raw_material and output
data = pd.concat([raw_material, output_data], axis =1)

#drop output data and id
data = data.drop(['error', 'error_type', 'quality', 'reflectionscore', 'distortion', 'nicesness', 'multideminsionality'], axis =1)

# Convert categorical data to numerical
ionizationclass = pd.get_dummies(data['ionizationclass'])
FluxCompensation = pd.get_dummies(data['fluxcompensation'])

data.drop(['ionizationclass', 'fluxcompensation'], axis = 1, inplace = True)

data = pd.concat([data, ionizationclass, FluxCompensation], axis =1)

# Calculating correlation coefficients to select input variables
correlation_matrix = data.corr()

# Plotting the correlation matrix
plt.figure(figsize = (12, 8))
sns.heatmap(correlation_matrix, annot = True, cmap = 'coolwarm')
plt.title("Correlation Matrix for Manufacturing Data")
plt.show()

#Assign x and y
x = data[['width']]
y = data['weight_in_kg']

# Splitting the data for 'width'
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)


# Linear Regression Model
linear_model = LinearRegression()
linear_model.fit(x_train, y_train)
y_pred_linear = linear_model.predict(x_test)


# Polynomial Regression Model (Degree 3)
polynomial_converter = PolynomialFeatures(degree = 3, include_bias = False)
x_train_poly = polynomial_converter.fit_transform(x_train)
x_test_poly = polynomial_converter.transform(x_test)

poly_model = LinearRegression()
poly_model.fit(x_train_poly, y_train)
y_pred_poly = poly_model.predict(x_test_poly)


# Visualizing the performance of Linear and Polynomial Models
plt.figure(figsize = (14, 6))

# Linear Model
plt.subplot(1, 2, 1)
plt.scatter(x_test['width'], y_test, color = 'blue', alpha = 0.5, label = 'Actual')
plt.scatter(x_test['width'], y_pred_linear, color = 'red', alpha = 0.5, label = 'Predicted - Linear', edgecolors = 'black')
plt.title('Linear Model: Actual vs Predicted')
plt.xlabel('Width')
plt.ylabel('Weight in kg')
plt.legend()
plt.gca().set_facecolor('lightgray')
plt.grid(True)

# Polynomial Model
plt.subplot(1, 2, 2)
plt.scatter(x_test['width'], y_test, color = 'blue', alpha = 0.5, label = 'Actual')
plt.scatter(x_test['width'], y_pred_poly, color = 'yellow', alpha = 0.5, label = 'Predicted - Polynomial', edgecolors = 'black')
plt.title('Polynomial Model: Actual vs Predicted')
plt.xlabel('Width')
plt.ylabel('Weight in kg')
plt.legend()
plt.gca().set_facecolor('lightgray')
plt.grid(True)

plt.tight_layout()
plt.show()


# Residual Analysis
residuals_linear = y_test - y_pred_linear
residuals_poly = y_test - y_pred_poly

# Plotting residuals for both models
plt.figure(figsize = (14, 6))

# Linear Model Residuals
plt.subplot(1, 2, 1)
plt.scatter(y_pred_linear, residuals_linear, color = 'blue', alpha = 0.5, edgecolors = 'black')
plt.title('Residuals of Linear Model')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.axhline(y = 0, color = 'red', linestyle = '--')
plt.gca().set_facecolor('lightgray')
plt.grid(True)
plt.tight_layout()

# Polynomial Model Residuals
plt.subplot(1, 2, 2)
plt.scatter(y_pred_poly, residuals_poly, color = 'green', alpha = 0.5, edgecolors = 'black')
plt.title('Residuals of Polynomial Model')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.axhline(y = 0, color = 'red', linestyle = '--')
plt.gca().set_facecolor('lightgray')
plt.grid(True)
plt.tight_layout()
plt.show()

#Error Check for Linear Model
mse_linear = mean_squared_error(y_test, y_pred_linear)
mae_linear = mean_absolute_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)

# Print ERRORs
print("MSE - Mean Squared Error of Linear Regression: " + str(round(mse_linear, 4)))
print("RMSE - Root Mean Square Error of Linear Regression: " + str(round(np.sqrt(mse_linear), 4)))
print("MAE - Mean Absolute Error of Linear Regression: " + str(round(mae_linear, 4)))
print("R Square of Linear Regression: " + str(round(r2_linear, 5)))
print('------------------------------------------------------------------------------')

#Error Check for Polynomial Model
mse_poly = mean_squared_error(y_test, y_pred_poly)
mae_poly = mean_absolute_error(y_test, y_pred_poly)
r2_poly = r2_score(y_test, y_pred_poly)

print("MSE - Mean Squared Error of Polynomial: " + str(round(mse_poly, 4)))
print("RMSE - Root Mean Square Error of Polynomial: " + str(round(np.sqrt(mse_poly), 4)))
print("MAE - Mean Absolute Error of Polynomial: " + str(round(mae_poly, 4)))
print("R Square of of Polynomial: " + str(round(r2_poly, 5)))

# Coefficients
x3_coefficients = round(poly_model.coef_[0], 4)
x2_coefficients = round(poly_model.coef_[1], 4)
x_coefficients = round(poly_model.coef_[2], 4)
intercept = round(poly_model.intercept_, 4)

print("Polinomial Function is " 'y = '+ str(x3_coefficients) + 'x^3 ' + '+ ' + str(x2_coefficients) + 'x^2 ' + str(x_coefficients) + 'x ' + '+ ' + str(intercept))


# # XGBOOST_CLASSIFIER

# In[263]:


#Q2 XGBOOST_CLASSIFIER width, height, ionization class and etc... (Error)

#concat raw_material and output
data = pd.concat([raw_material, output_data], axis =1)

#drop output values
new_data = data.drop(['error_type' ,'weight_in_kg', 'quality', 'reflectionscore', 'distortion', 'nicesness', 'multideminsionality'], axis = 1)

#define x and y values
x = new_data.drop(['error'], axis =1)
y = new_data[['error']]

le = LabelEncoder()
y = le.fit_transform(y)

#define train, test dataset, create model
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)

model = XGBClassifier(max_depth = 4)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

#Measure Accuracy
balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', round(accuracy, 3))
print('Balanced Accuracy:', round(balanced_accuracy, 3))

# Calculating the F1 Score
f1 = round(f1_score(y_test, y_pred),2)
print("F1 Score is " + str(f1))

#Visualise Tree
fig, ax = plt.subplots(figsize=(30, 30))
xgb.plot_tree(model, num_trees = 0, ax = ax, class_names = data['error'].unique())
plt.show()

# Calculation of Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Visualise Confusion Matrix
plt.figure(figsize=(10, 8))
ax = sns.heatmap(conf_matrix, annot = True, fmt = 'd', cmap = 'viridis', cbar = False, annot_kws = {"size": 16})
plt.title('Confusion Matrix', fontsize = 20)
plt.ylabel('True Label', fontsize = 16)
plt.xlabel('Predicted Label', fontsize = 16)
ax.set_xticklabels(['Negative (0)', 'Positive (1)'], fontsize=14)
ax.set_yticklabels(['Negative (0)', 'Positive (1)'], fontsize=14, rotation=0)
plt.show()

# Calculate the ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred)

# Calculate the Area Under Curve
auc_score = roc_auc_score(y_test, y_pred)

# Plotting the ROC curve
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate', fontsize=16)
plt.title('Receiver Operating Characteristic (ROC)', fontsize=20)
plt.legend(loc="lower right", fontsize=16)
plt.gca().set_facecolor('lightgray')
plt.grid(True)
plt.show()

# TPR FPR
true_negative, false_positive, false_negative, true_positive = confusion_matrix(y_test, y_pred).ravel()

# Calculating True Positive Rate (TPR) and False Positive Rate (FPR)
tpr = round(true_positive / (true_positive + false_negative),2)  # TPR = TP / (TP + FN)
fpr = round(false_positive / (false_positive + true_negative),2) # FPR = FP / (FP + TN)

# Creating a table for display
results = {
    "Metrics": ["True Positive Rate (TPR)", "False Positive Rate (FPR)"],
    "Values": [tpr, fpr]
}

results_df = pd.DataFrame(results)

# Plotting the table
fig, ax = plt.subplots(figsize=(5, 2))  # set size frame
ax.axis('tight')
ax.axis('off')
ax.table(cellText=results_df.values, colLabels=results_df.columns, cellLoc = 'center', loc='center',
         colColours=["palegreen", "paleturquoise"])

plt.title("Calculated TPR and FPR", fontsize=16, color="darkblue")
plt.gca().set_facecolor('lightgray')
plt.show()


# In[ ]:





# In[ ]:




