# Import libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib

# Load dataset
data = pd.read_csv("student_data.csv")

# Input features
X = data[[
    "Study_Hours",
    "Attendance",
    "Previous_Score"
]]

# Output column
y = data["Result"]

# Create model
model = DecisionTreeClassifier()

# Train model
model.fit(X, y)

# Save model
joblib.dump(
    model,
    "student_model.pkl"
)

print("Model Saved Successfully")