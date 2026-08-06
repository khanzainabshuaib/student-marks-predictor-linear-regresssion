import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Sample Dataset: Study Hours, Attendance (%), Sleep Hours -> Marks (Out of 100)
data = {
    'study_hours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'attendance':  [50, 60, 65, 70, 75, 80, 85, 90, 95, 98],
    'sleep_hours': [5, 6, 6, 7, 7, 8, 8, 7, 8, 8],
    'marks':        [35, 45, 52, 60, 68, 75, 82, 88, 94, 99]
}

df = pd.DataFrame(data)

# Features (X) & Target (y)
X = df[['study_hours', 'attendance', 'sleep_hours']]
y = df['marks']

# Train Model
model = LinearRegression()
model.fit(X, y)

# Save Model File
joblib.dump(model, 'student_model.pkl')
print("✅ Student model save ho gaya hai ('student_model.pkl')!")