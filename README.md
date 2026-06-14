# Linear Regression from Scratch: Salary Prediction

## Overview
This repository contains a custom, from-scratch Python implementation of a Univariate Linear Regression model. Using only core numerical libraries (NumPy and Pandas), the model is trained using Batch Gradient Descent to predict an employee's salary based on their years of experience.

## Dataset
The project utilizes the **Salary Data Dataset** sourced from Kaggle.
* **Link:** [Salary Data Dataset for Linear Regression](https://www.kaggle.com/datasets/shubham47/salary-data-dataset-for-linear-regression)
* **Features:**
  * `YearsExperience` (Independent Variable / X): The number of years the employee has worked.
  * `Salary` (Target Variable / y): The employee's annual salary in USD.

## Insights from the Dataset
By analyzing and visualizing the data prior to and during training, several key insights were uncovered:
1. **Strong Positive Correlation:** There is a highly predictable, linear relationship between years of experience and salary. As an employee accumulates more experience, their salary increases proportionally.
2. **Disparate Scales Require Normalization:** The input feature (`YearsExperience`) ranges roughly from 1 to 10, while the target (`Salary`) ranges from 30,000 to over 120,000. This massive difference in scale means that without feature scaling (like the Z-normalization implemented here), a gradient descent algorithm would struggle to converge and could easily overshoot the minimum loss.
3. **Smooth Convergence:** Tracking the loss history shows that the model learns rapidly in the initial iterations. The Mean Squared Error (MSE) drops drastically from over 3.2 billion down to approximately 15.6 million, proving the custom gradient calculations accurately guide the weights to their optimal values.

## Implementation Details
The project intentionally avoids high-level machine learning libraries like `scikit-learn` to demonstrate the underlying mathematics of linear regression.

### 1. Data Preprocessing
Z-normalization (standardization) is implemented from scratch to scale the input feature $X$ so it has a mean of 0 and a standard deviation of 1.

### 2. Custom `LinearRegression` Class
The core model features:
* **Hyperparameters:** A learning rate of `0.1`, a maximum of `2000` iterations, and an early stopping threshold of `1e-6`.
* **Gradient Descent:** Iteratively updates the weight and bias by calculating the gradients of the Mean Squared Error (MSE) loss function.
* **Early Stopping:** The algorithm monitors the change in loss between iterations and automatically halts training once the loss stabilizes below the defined threshold, saving computational resources.

## Conclusion
Building this Linear Regression model from scratch successfully demonstrates the mechanics of machine learning algorithms under the hood. The custom Batch Gradient Descent algorithm effectively minimized the Mean Squared Error, converging smoothly to find the optimal line of best fit. Ultimately, the model proves highly capable of predicting salaries based on years of experience, validating both the mathematical implementation and the linear nature of the dataset.

## Requirements
To run this notebook, you need the following Python libraries installed:
* Python 3.x
* NumPy
* Pandas
* Matplotlib

## How to Run
1. Clone this repository to your local machine.
2. Install the required dependencies: `pip install numpy pandas matplotlib`
3. Download the `Salary_Data.csv` file from Kaggle and place it in the same directory as the notebook.
4. Open the Jupyter Notebook (`linear_regression_scratch.ipynb`).
5. Run all cells sequentially to view the data preprocessing, model training loss history, and the final plotted regression line.