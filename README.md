# Student Grade Prediction

## Overview

This project aims to predict students' final grades based on academic performance indicators such as attendance, study hours, number of assignments, and semester using a Linear Regression machine learning model.

---

## Dataset

The dataset contains 200 student records with key features including:

- Standard (grade level)
- Attendance percentage
- Study hours per week
- Number of assignments completed
- Semester
- Final grade (target variable)

---
## Usage

- Run the Jupyter notebook `student_grade_prediction.ipynb` to train the model, evaluate performance, and visualize insights.
- The trained model and scaler are saved as `linear_regression_model.pkl` and `scaler.pkl` respectively.
- Use these saved files to predict grades on new student data.

---

## Model Performance

The Linear Regression model is evaluated with:

- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- R² Score

These metrics provide insight into prediction accuracy and model fit.

---
## Future Work

- Experiment with additional regression models like Random Forest or Gradient Boosting.
- Implement hyperparameter tuning to improve model accuracy.
- Develop a web-based interface to allow easy grade predictions.
