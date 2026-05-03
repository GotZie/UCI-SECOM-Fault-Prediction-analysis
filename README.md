
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
Small Sample

The objective of this project is to:

Build a machine learning model capable of accurately detecting failure cases (class = 1), prioritizing recall to minimize missed defects.


#### Key Challenges
Large number of missing values across features  
Many low-variance and redundant features  
Highly imbalanced target variable (104 failures out of 1,567 samples)
Small Sample size


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
│   └── Initial Analysis.ipynb  # EDA and initial analyis
│   └── Optimised Modeling.ipynb  # Model Optimization 
│
├── requirements.txt
│
├── README.md
│
└── .gitignore





## Approach

1. Data Exploration and PreProcessing  
* Removed features with excessive missing values
* Imputed remaining missing values using mean
* Removed low-variance features
* Eliminated highly correlated features 
* Scaled numerical features
* Handled class imbalance using 'class weight'

2. Model Training  
* Logistic Regression   * Random Forest
* Support Vector Classifier (SVC)   * XGBoost
* LightGBM

3. Optimization Techniques
* Random Forest Feature importance
* Further class imbalance handling using SMOTENN
* Hyperparameter Tuning
* Threshold Optimization

4. Evaluation Metrics
Recall (primary metric), Confusion Matrix, F1_Score, ROC-AUC and SHAP


### Key Assumptions
A "Missed Failure" is significantly more costly than a "False Alarm".


### Key Findings

Correlation analysis revealed maximum correlation values to be around ±0.15, meaning changes in the sensor have no predictable impact on whether the product passes or fails.  

The Logistic Regression is the worst performing model failling to show improvements to optimization techniques. It maintained a 0.54 Recall score throughout the analysis

The XGBoost is the best performing model catching nearly 80% of failures while maintaining a high Recall and Precision level that prevents the quality control team from being overwhelmed by false reports.  

SHAP analysis revealed that Feature 0 is the primary driver of failure. However, it isn't the only cause of failure,  
secondary drivers like Feature 7 and Feature 40 helped maintain high detection rates.

Accuracy alone was found to be misleading due to class imbalance


### Confusion Matrix Comparison
The following plots illustrates the behaciour of all 5 models. By lowering the decision threshold, we successfully minimized the bottom-left quadrant (**Missed Fails**) across all models.

![Confusion Matrix Comparison](final_confu_matrix.png)

*Figure 1: Comparison of model performances.*


### Final Results


| Model | Recall (Failures Caught) | False Alarms | Verdict |
| :--- | :--- | :--- | :--- |
| **XGBoost** | 79.2% | 92 | Best balance of high detection with moderate inspection overhead. |
| **LightGBM** | 75.0% | 79 | Lowest Operational Cost with Lowest false-alarm rate. Keeps the production line moving fastest. |
| **SVM** | 83.3% | 132 | Best at catching defects. Suitable for high-stakes scenarios where missing a defect is catastrophic. |
| **Random Forest** | 70.8% | 80 | Reliable performance, but outclassed by XGBoost in detection sensitivity.  |
| **Logistic Regression** | 54.2% | 73 |	Inadequate. Failed to capture nearly half of the defects. |



### Conclusions

No single sensor strongly determines failure. Correlation results suggests that failure is driven by complex interactions between multiple process variables rather than individual sensor readings.

The failure of Logistic Regression (missing 46% of defects) proves that sensor failures are not caused by single-variable shifts. Instead, they are caused by complex interactions between multiple sensors. Therefore, ensemble methods (XGBoost, Random Forest) are mandatory for this dataset because they can "see" these non-linear relationships.

Deploying the XGBoost model with a 0.047 threshold provides the most professional balance, catching 4 out of every 5 defects while keeping false inspections at a manageable level for the engineering team.

However, If the factory's priority is Zero Defects, then the SVM is the superior choice (at 83.3% recall) but the XGBoost will reduce the inspection workload (False Alarms) by 30% compared to the SVM.

Though the LightGBM has the best F1-Score and Precision, the XGBoost remains superior especially because of the extremely low threshold (0.004) of the LightGBM model. Such hyper-sensitivity can cause model instability, a tiny bit of sensor noise could cause the model to fluctuate wildly.  

The XGBoost’s threshold of 0.047 is more robust and reliable for a real production line.




### Recommendation
* A two-tier inspection process is recommended to mitigate the cost of false alarms while maintaining high safety standards. 
* Adoption of Model Cascading for possible improved results.
* Improve Data acquisition over a longer duration (e.g., 6–12 months) for a wider sample size and to capture seasonal factory shifts, sensor wear-and-tear, and different material batches.
* Prioritize the maintenance and calibration of sensors identified by SHAP (especially Feature 0, 59 and 33). They provide the signal that the best models rely on. If they drift or fail, the entire predictive system will collapse.




## Author

Chigozie Okonkwo  
Machine Learning | Data Science | Predictive Systems












