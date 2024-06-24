# Boston Housing Principal Component Analysis and Linear Regression

This repository contains a PySpark script for performing Principal Component Analysis (PCA) and Linear Regression on the Boston Housing dataset.

## Overview

The script performs the following steps:

1. Sets up a PySpark session in Google Colab.
2. Reads the Boston Housing dataset into a PySpark DataFrame.
3. Combines all feature columns into a single vector column using `VectorAssembler`.
4. Standardizes the features for PCA using `StandardScaler`.
5. Trains a PCA model with two principal components and transforms the data.
6. Collects the PCA features and target column as numpy arrays, and converts them to a pandas DataFrame.
7. Plots the two principal components.
8. Splits the dataset into a training set and a testing set.
9. Fits a Linear Regression model to the training data.
10. Evaluates the model on the test data, printing the Root Mean Squared Error (RMSE) and R-squared value.
11. Performs further analysis by examining the residuals of the model and calculating the correlation between the predicted and actual values.
12. Performs hyperparameter tuning using `CrossValidator` and `ParamGridBuilder`.

## Requirements

- PySpark
- Numpy
- Pandas
- Matplotlib

## Usage

To run the script, simply open it in Google Colab and execute the cells in order.

## Results

The script outputs the coefficients and intercept of the Linear Regression model, the RMSE and R-squared value of the model on the test data, a histogram of the residuals, and the correlation between the predicted and actual values. It also performs hyperparameter tuning and outputs the coefficients and intercept of the best model found.
