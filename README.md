
# SECOM FAULT PREDICTION ANALYSIS

Fault Detection Using Machine Learning in Semiconductor Manufacturing.


## Problem Statement



Semiconductor manufacturing processes involve hundreds of sensors generating high-dimensional data. Detecting faulty products early in the production pipeline is critical to avoid:

Financial losses
Product recalls
Reduced manufacturing efficiency

However, the dataset presents two major challenges:

High Dimensionality (590+ sensor features)  
Severe Class Imbalance (very few failure cases)

The objective of this project is to:

Build a machine learning model capable of accurately detecting failure cases (class = 1), prioritizing recall to minimize missed defects.


#### Key Challenges
Large number of missing values across features  
Presence of low-variance and redundant features  
Highly imbalanced target variable (~7% failures)




### Dataset Summary

*Dataset Name:* SECOM Dataset  

*Domain:* Semiconductor Manufacturing

*Dataset Source:*
[UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/179/secom) 

*Number of Samples:* 1567

*Number of Features:* 590 sensor measurements

*Target Variable:*
1 → Failure
-1 → Normal






## Repository Structure


SECOM-Fault-Prediction-analysis/
│
├── data/
│   ├── raw/                # Original dataset 
│   └── processed/          # Cleaned dataset
│
├── notebooks/
│   └── UCI SECOM.ipynb  # EDA and initial experimentation
│
├── README.md
│
└── .gitignore









## Approach

1. Data Cleaning  
* Removed features with excessive missing values
* Imputed remaining missing values using mean
2. Feature Selection  
* Removed low-variance features
* Eliminated highly correlated features
3. Data Preprocessing  
* Scaled numerical features
* Handled class imbalance using class weighting
4. Model Training  
* Logistic Regression
* Random Forest
* Support Vector Classifier (SVC)
5. Evaluation  
Accuracy, Confusion Matrix, Recall (primary metric)  
and ROC-AUC



### Key Findings
Most models achieved high accuracy but failed to detect failure cases
Logistic Regression performed best in terms of failure detection
Accuracy alone was found to be misleading due to class imbalance

### Conclusions

Logistic Regression with class weighting is currently the best tested model for this problem because it achieved the highest recall for failure detection and
Successfully identified a significant portion of failure cases.
Hence, it provides a better balance between interpretability and performance

### Future Improvements/Recommendation
* Apply other resampling techniques (such as SMOTE, oversampling)  
* Optimize decision threshold for better recall  
* Perform hyperparameter tuning  
* Implement advanced models with imbalance-aware training
* Add model explainability (e.g., SHAP values)
## Author

Chigozie Okonkwo  
Machine Learning | Data Science | Predictive Systems

