import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# ==========================================
# Part 1: NumPy – Numerical Computing
# ==========================================
print("--- Part 1: NumPy ---")

# Task 1: Array Creation
arr_1d = np.arange(21)
arr_2d = np.random.randint(10, 100, size=(4, 5))
identity_matrix = np.eye(3)

# Operations
print(f"2D Array Mean: {np.mean(arr_2d)}")
print(f"2D Array Median: {np.median(arr_2d)}")
print(f"2D Array Std Dev: {np.std(arr_2d)}")
print(f"Second Row: {arr_2d[1, :]}")
print(f"Third Column: {arr_2d[:, 2]}")

# Task 2: Broadcasting & Reshaping
arr_16 = np.arange(16).reshape(4, 4)
broadcasted_arr = arr_16 + np.array([1, 2, 3, 4])
flattened_arr = broadcasted_arr.flatten()

# ==========================================
# Part 2: Pandas – Data Manipulation
# ==========================================
print("\n--- Part 2: Pandas ---")

# Task 3: DataFrame Creation
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Evan'],
    'Age': [25, 30, 28, 35, 40],
    'Department': ['IT', 'HR', 'IT', 'Marketing', 'HR'],
    'Salary': [50000, 45000, 72000, 58000, 60000]
}
df = pd.DataFrame(data)

# Operations
df['Bonus'] = df['Salary'] * 0.10
hr_employees = df[df['Department'] == 'HR']
avg_salary = df.groupby('Department')['Salary'].mean()

print("HR Employees:\n", hr_employees)
print("\nAverage Salary by Dept:\n", avg_salary)

# Save to CSV (VS Code এ এটি আপনার ফোল্ডারে সেভ হবে)
df.to_csv('employee_data.csv', index=False)

# ==========================================
# Part 3: Classification Evaluation
# ==========================================
print("\n--- Part 3: ML Metrics ---")

# Load & Prepare Iris Data
iris = load_iris()
df_iris = pd.DataFrame(iris.data, columns=iris.feature_names)
df_iris['binary_target'] = (iris.target == 1).astype(int)

X = df_iris[iris.feature_names]
y = df_iris['binary_target']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# Train Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Metrics Calculation
conf_matrix = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Confusion Matrix:\n", conf_matrix)
print(f"Accuracy: {acc:.2f}")
print(f"Precision: {prec:.2f}")
print(f"Recall: {rec:.2f}")
print(f"F1-Score: {f1:.2f}")

# Explanations
print("\n--- Metric Explanations ---")
print("Accuracy: Overall percentage of correct guesses.")
print("Precision: How many of the predicted 'Versicolor' were actually 'Versicolor'.")
print("Recall: How many of the actual 'Versicolor' were correctly identified.")
print("F1-Score: Balance between Precision and Recall.")
