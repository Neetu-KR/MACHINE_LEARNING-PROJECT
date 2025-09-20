# ■ Student Final Grade Prediction
This project uses **machine learning** to predict students' final grades based on academic and
behavioral features such as attendance, study hours, assignments, and semester.
We compare multiple models (Linear Regression and Random Forest) and save the best-performing
one for future predictions.
---
## ■ Features - Load and preprocess student dataset (`CSV` file). - Train/test split for model
evaluation. - Feature scaling using **StandardScaler**. - Train and evaluate: - **Linear Regression** -
**Random Forest Regressor** - Select and save the **best model** using `joblib`. - Predict the final
grade for a **new student**.
---
## ■ Dataset The dataset used: `students_dataset_200_with_splitnames_reordered.csv`
### Example columns: - `standard` → Student’s class/grade level - `attendance` → Attendance
percentage - `study_hours` → Hours spent studying per week - `assignments` → Number of
assignments submitted - `semester` → Current semester - `final_grade` → Target variable (grade to
predict)
---
## ■■ Installation
Clone this repository and install dependencies:
```bash git clone https://github.com/yourusername/student-grade-prediction.git cd
student-grade-prediction pip install -r requirements.txt ```
### Requirements ``` pandas numpy scikit-learn joblib ```
---
## ■ Model Training & Evaluation
We train two models and evaluate them with **MSE** and **R²**.
■ The best model is saved automatically.
---
## ■ Saving Models The script saves: - `best_student_model.pkl` → Best model (by MSE) - `scaler.pkl`
→ Scaler for preprocessing new data
---
## ■ Predicting for a New Student
Example input: - Standard: 10 - Attendance: 85% - Study Hours: 12 - Assignments: 7 - Semester: 2

Output: ■ Predicted Final Grade: ~78.65
---
## ■ Next Steps - Add more models (XGBoost, Neural Networks). - Perform hyperparameter tuning. -
Build a simple **Flask/Streamlit web app** for user interaction.
---
## ■ License This project is licensed under the MIT License.

