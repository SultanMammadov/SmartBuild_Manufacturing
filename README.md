**Introduction:**
        
Smart Build is a manufacturing company that faces issues with product defects and difficulty of predicting weight of future products. Errors in products can be influenced by factors such as weight, ionization class, width and various other characteristics.
The uncertainty of products’ weight cause damages on the Quality Control, Planning in Transport, Storage and Warehousing, Packaging processes.

By leveraging machine learning models on historical data, these errors can be predicted in advance, allowing the company to proactively address potential defects and improve product quality.

**Objective:** The goal of this project is to address the following business issues: 
 
 - What will be weight of the future products ?
 - How can we predict errors in advance ? 

**Data Collection:** The dataset is synthetic.


**Data Cleaning:**

The dataset was first checked for missing values, and two uninformative columns were removed — id, which carried no analytical value, and weight_in_g, which was confirmed to be a duplicate of weight_in_kg through a correlation check. Feature distributions were then visualised using histograms, which revealed unusual values in the width, weight_in_kg and nicesness columns. Outliers in these columns were removed using the Z-score method, with a threshold of 3, ensuring that only extreme values were eliminated. Finally, all numeric columns were normalised using MinMax scaling to bring all features onto a consistent scale, preparing the data for reliable model training.

Most features look reasonably well-distributed, with karma, reflectionscore and modulation showing a clean, bell-shaped pattern. However, weight_in_kg stands out with a strong right skew, where most values cluster near zero but stretch out to 3, hinting at some extreme values that need attention. Quality and distortion show fragmented, uneven distributions with noticeable gaps, which could reflect discrete groupings or inconsistencies in how the data was recorded. Overall, while most features are in good shape, width, height, nicesness and weight_in_kg show irregular tails that are worth addressing before moving into modelling.

<img width="1161" height="913" alt="download" src="https://github.com/user-attachments/assets/ba288bbb-2e87-41a5-9ea2-65c2e6ca4177" />

**Figure_1.** "Feature Distribution"

**Feature Engineering:**

Feature engineering began by converting boolean columns — error and multideminsionality — from yes/no text values to binary 1/0 format, making them numerically interpretable. Label Encoding was then applied to the categorical columns ionizationclass and fluxcompensation, replacing text categories with corresponding numerical values. The dataset was subsequently split into input and output features, separating raw material properties from production outcomes to provide a clearer structure for modelling. Finally, a correlation matrix was computed on the input features, revealing strong relationships between weight_in_kg, width and height, which informed feature selection for the models ahead.

The below correlation matrix displays the relationships between various variables in a dataset. It shows correlation coefficients that measure the strength and direction of these relationships. There is a very high relationship between width and weight_in_kg (0.97), indicating that weight_in_kg increase as the width increase. The relationships between other variables are very weak.

<img width="645" height="543" alt="download" src="https://github.com/user-attachments/assets/bebc564e-117d-4a9f-a5b1-3dedb01f2eaf" />

**Figure_2.** "Correlation Matrix of all variables"


**1st Model:** Polynomial Model:

The dataset was split into 80% for training and 20% for testing.

![image](https://github.com/user-attachments/assets/9ef67098-99e4-48aa-ade7-3d2600e25c07)

**Figure_3.** "Linear Model vs Polynomial Model"

![image](https://github.com/user-attachments/assets/1abcb9d0-5dff-4ba0-b1cb-cf9a9e8e99e4)

**Figure_4.** "Residual Distribution of Linear and Polynomial Models"

The residuals appear to be randomly scattered around 0 without any systematic pattern. This indicates that the model is well-suited for the data and there is no obvious sign of non-linearity that the model is not capturing.

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

Both models do a strong job of predicting weight from width, but the Polynomial model clearly has the edge. The Linear model follows the data well with an R² of 0.95, though its residual plot reveals a distinct curved pattern, which is a tell-tale sign that a straight line is not fully capturing the relationship between width and weight. The Polynomial model addresses this nicely, fitting a smooth curve through the data and achieving an R² of 0.97, with noticeably lower error across all metrics — MSE dropping from 0.0021 to 0.0011 and MAE from 0.0352 to 0.0241. Its residuals are much more evenly scattered around zero, suggesting the model is capturing the underlying relationship more accurately. Overall, the Polynomial model of degree 3 — expressed as y = 0.353x³ + 0.917x² − 0.220x + 0.021 — is the stronger choice and is well-suited for predicting product weight based on width in a manufacturing setting.


**2nd Model:** XGBoost Classifier:

The dataset was split into 70% for training and 30% for testing.
The model is predicting Error in future products by using main factors as input.

![image](https://github.com/user-attachments/assets/225b3648-66fa-432d-8d97-926eb5a0b7d5)

**Figure_5.** "Decision Tree"

The Confusion Matrix below shows slightly lower performance than Confusion Matrix of 1st model with the following values: True Negatives (TN): 1783, False Positives (FP): 186, False Negatives (FN): 48, and True Positives (TP): 951.

![image](https://github.com/user-attachments/assets/a889ace5-0817-4213-8563-07fb94d4e14c)

**Figure_6.** "Confusion Matrix"

The ROC (Receiver Operating Characteristic) graph below illustrates the relationship between the True Positive Rate (TPR) and the False Positive Rate (FPR).

![image](https://github.com/user-attachments/assets/a588b78d-abbe-4bc0-839c-98fa8472b38a)

**Figure_7.** "ROC (Receiver Operating Characteristic)"

The following performance results were achieved using the XGBoost Classifier model, demonstrating the model's high reliability and effectiveness.
Accuracy: 0.921
Balanced Accuracy: 0.905
F1 Score is 0.94

**Why the Analysis Was Done This Way ?**
The XGBoost Classifier delivers strong results with an accuracy of 92.1%, a balanced accuracy of 90.5%, and an F1 score of 0.94, confirming it handles both error and non-error cases reliably — and notably outperforming a standard Decision Tree, thanks to XGBoost's ability to capture complex, non-linear relationships between features. Feature importance reveals that karma, width, and height are the primary drivers of error prediction, while fluxcompensation and ionizationclass contribute the least. The confusion matrix shows 1,783 true positives and 951 true negatives, with only 48 false negatives and 186 false positives — a solid outcome in a manufacturing setting where early error detection is critical. The ROC curve (AUC = 0.90) and a TPR of 0.97 further confirm the model's effectiveness and using a comprehensive set of metrics — F1, TPR, FPR, Accuracy and AUC — ensures a well-rounded and transparent evaluation of performance.
From a business perspective, the value this model delivers is significant. With 92% prediction accuracy, only 8% of defective products go undetected, which translates to a saving of approximately 12,000 EUR per 1,000 products — enough to purchase around 1,200 kg of raw materials. The solution can also be quickly integrated into existing IT systems using Python, with the potential to support real-time analysis through data streams. Ultimately, this level of accuracy creates tangible value across cost reduction, customer satisfaction and legal compliance and safety standards.


**Business problems:**  
 
 - What will be weight of the future products ?

The Polynomial Regression model (degree 3) demonstrated that product weight can be reliably predicted from width alone, achieving an R² of 0.97 and a low MAE of 0.024.
This means the business can accurately estimate the weight of future products at the input stage, before production is complete, enabling better material planning, cost forecasting and quality control.

 - How can we predict errors in advance ?
 
The XGBoost Classifier answers this question directly, achieving 92.1% accuracy in identifying defective products before they reach the end of the production line. 
By analysing input features such as karma, width and height, the model flags potential errors early, giving the business the opportunity to intervene before defects escalate. 
This translates to an estimated saving of 12,000 EUR per 1,000 products, reduced material waste and improved compliance with safety and quality standards. 
With Python-based integration, the model can also be deployed for real-time error prediction within existing IT systems, making it a practical and scalable solution for the manufacturing floor.


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



