**Introduction:**

Smart Build is a manufacturing company that faces issues with product defects and difficulty of predicting weight of future products. Errors in products can be influenced by factors such as weight, ionization class, width, and various other characteristics.
The uncertainty of products’ weight cause damages on the Quality Control, Planning in Transport, Storage and Warehousing, Packaging processes.

By leveraging machine learning models on historical data, these errors can be predicted in advance, allowing the company to proactively address potential defects and improve product quality.

**Objective:** The goal of this project is to address the following business issues: -What will be weight of the future products ? -How can we predict errors in advance ? 

**Data Collection:** The dataset is synthetic.

**1st Model:** Polynomial Model:

The below correlation matrix displays the relationships between various variables in a dataset. It shows correlation coefficients that measure the strength and direction of these relationships. There is a very high relationship between width and weight_in_kg (0.97), indicating that weight_in_kg increase as the width increase. The relationships between other variables are very weak.

![image](https://github.com/user-attachments/assets/4e8b45ac-674a-4088-b8fd-6b826826d148)

**Figure_1.** "Correlation Matrix of all variables"

![image](https://github.com/user-attachments/assets/9ef67098-99e4-48aa-ade7-3d2600e25c07)

**Figure_2.** "Linear Model vs Polynomial Model"

![image](https://github.com/user-attachments/assets/1abcb9d0-5dff-4ba0-b1cb-cf9a9e8e99e4)

**Figure_3.** "Residual Distribution of Linear and Polynomial Models"

The below performance measurements confirm the strength of this relationship:
**Linear Model's Perforance:**
MSE - Mean Squared Error of Linear Regression: 0.0021
RMSE - Root Mean Square Error of Linear Regression: 0.0457
MAE - Mean Absolute Error of Linear Regression: 0.0352
R Square of Linear Regression: 0.95073

**Polynomial Model's Perforance:**
MSE - Mean Squared Error of Polynomial: 0.0011
RMSE - Root Mean Square Error of Polynomial: 0.0334
MAE - Mean Absolute Error of Polynomial: 0.0241
R Square of of Polynomial: 0.97365
**Polinomial Function** is y = 0.353x^3 + 0.9171x^2 -0.2201x + 0.0213

**2nd Model:** XGBoost Classifier:

![image](https://github.com/user-attachments/assets/225b3648-66fa-432d-8d97-926eb5a0b7d5)


**Figure_4.** "Decision Tree"










