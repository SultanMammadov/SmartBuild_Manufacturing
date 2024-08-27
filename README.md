**Introduction:**

Smart Build is a manufacturing company that faces issues with product defects and difficulty of predicting weight of future products. Errors in products can be influenced by factors such as weight, ionization class, width, and various other characteristics.
The uncertainty of products’ weight cause damages on the Quality Control, Planning in Transport, Storage and Warehousing, Packaging processes.

By leveraging machine learning models on historical data, these errors can be predicted in advance, allowing the company to proactively address potential defects and improve product quality.

**Objective:** The goal of this project is to address the following business issues: -What will be weight of the future products ? -How can we predict errors in advance ? 

**Data Collection:** The dataset is synthetic.

**1st Model:** Polynomial Model:
The dataset was split into 80% for training and 20% for testing.
The below correlation matrix displays the relationships between various variables in a dataset. It shows correlation coefficients that measure the strength and direction of these relationships. There is a very high relationship between width and weight_in_kg (0.97), indicating that weight_in_kg increase as the width increase. The relationships between other variables are very weak.

![image](https://github.com/user-attachments/assets/4e8b45ac-674a-4088-b8fd-6b826826d148)

**Figure_1.** "Correlation Matrix of all variables"

![image](https://github.com/user-attachments/assets/9ef67098-99e4-48aa-ade7-3d2600e25c07)

**Figure_2.** "Linear Model vs Polynomial Model"

![image](https://github.com/user-attachments/assets/1abcb9d0-5dff-4ba0-b1cb-cf9a9e8e99e4)

**Figure_3.** "Residual Distribution of Linear and Polynomial Models"

The residuals appear to be randomly scattered around 0 without any systematic pattern. This indicates that the model is well-suited for the data, and there is no obvious sign of non-linearity that the model is not capturing.


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

The dataset was split into 70% for training and 30% for testing.
The model is predicting Error in future products by using main factors as input.

![image](https://github.com/user-attachments/assets/225b3648-66fa-432d-8d97-926eb5a0b7d5)

**Figure_4.** "Decision Tree"

The Confusion Matrix below shows slightly lower performance than Confusion Matrix of 1st model with the following values: True Negatives (TN): 1783, False Positives (FP): 186, False Negatives (FN): 48, and True Positives (TP): 951.

![image](https://github.com/user-attachments/assets/a889ace5-0817-4213-8563-07fb94d4e14c)

**Figure_5.** "Confusion Matrix"

The ROC (Receiver Operating Characteristic) graph below illustrates the relationship between the True Positive Rate (TPR) and the False Positive Rate (FPR).
![image](https://github.com/user-attachments/assets/a588b78d-abbe-4bc0-839c-98fa8472b38a)

**Figure_6.** "ROC (Receiver Operating Characteristic)"

The following performance results were achieved using the XGBoost Classifier model, demonstrating the model's high reliability and effectiveness.
Accuracy: 0.921
Balanced Accuracy: 0.905
F1 Score is 0.94


**Why the Analysis Was Done This Way ?**
The XGBClassifier predicted Errors much more accurately than Decision Tree.
Relationship is complex and non-linear between features which XGBoost can capture better.  
Comprehensive Evaluation: Using metrics like F1, TPR, FPR, Accuracy, and AUC prove the model's performance and the way of analysis.
Balancing Sensitivity and Specificity: TPR and FPR are crucial for evaluating the trade-offs between correctly identifying positive cases and avoiding false positives.

**What value can we derive from the insights?**
92% correct prediction in defective products, just 8% can’t be detected.
Model can help to save 12k eur (1000*0.08*150) per 1k products by predicting defective products in advance. 
1200kg (12000 eur/ 10) raw materials could be bought with the saving.
Fast integration: Our reports including visualizations can be quickly integrated into existing IT systems using the Python programming, as well as streams for real-time analysis
Prediction of errors with this level of accuracy can be valuable in manufacturing to solve problems in the below areas: 
Cost Reduction
Customer Satisfaction and Trust
Legal Compliance and Safety Standards

**What could be improved ?**
Feature Engineering:
Investigate additional provided features that could improve model performance.

Model Tuning:
Fine-tuning hyperparameters of the XGBClassifier could further optimize performance.

Advanced Validation Techniques:
Implementing k-fold cross-validation for a more robust evaluation.
Using stratified sampling to ensure balanced representation of classes.

Addressing Class Imbalance:
Techniques like SMOTE or adjusting class weights in the XGBClassifier to deal with imbalanced datasets.

Alternative Models:
Comparing results with other algorithms like Random Forest, SVM, or neural networks to find the best-performing model.













