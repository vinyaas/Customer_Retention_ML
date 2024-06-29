# Customer Retention Machine Learning Model

This repository contains the source code for a machine learning model developed to predict customer retention. The model was built using a Decision Tree Classifier and achieved an accuracy of 98%.

## Project Overview

The project involved several stages:

1. **Exploratory Data Analysis (EDA)**: Initial exploration of the dataset to understand the features and data distribution.

2. **Visualization of Features**: Graphical representation of data for better understanding and insight generation.

3. **Feature Engineering**: This included:
   - Label Encoding
   - Principal Component Analysis (PCA)
   - Forward Feature Selection
   - Pearson Correlation Coefficient

4. **Model Building**: Post feature engineering, a Decision Tree Classifier was used to build the model. GridSearchCV was used for hyperparameter tuning.

5. **Handling Imbalanced Data**: The SMOTEENN technique was used to handle imbalanced data and improve the model's performance.

## Results

The initial model achieved an accuracy of 80%. After resampling the data using SMOTEENN to handle data imbalance, the model's accuracy improved to 98%.

## Deployment

After achieving the desired accuracy, a Flask API was built with a frontend to collect data and predict the outcome. The API predicts whether a customer will churn or not, and displays the probability of churning.

## Libraries Used

The following libraries were used in this project:

- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- imbalanced-learn
- Flask

## Source Code

All the source code for the model can be found in the `src` directory of this repository.

## Conclusion

This project successfully predicts whether the customer will is likely to churn or not .This project also demonstrates the effectiveness of decision trees and resampling techniques in handling imbalanced data and improving model performance. The model has been deployed using a Flask API, providing a user-friendly interface for making predictions.


![image](https://github.com/vinyaas/Customer-Retention-ML-End-to-End-/assets/124361378/d9d4b3ea-6004-45e1-9f36-d135e9ffea5f)
