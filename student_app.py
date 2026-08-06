from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Model Load Karo
model = joblib.load('student_model.pkl')

@app.route('/')
def home():
    return render_template('student_index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        study_hours = float(data['study_hours'])
        attendance = float(data['attendance'])
        sleep_hours = float(data['sleep_hours'])

        # Predict Marks
        features = np.array([[study_hours, attendance, sleep_hours]])
        predicted_marks = model.predict(features)[0]
        
        # Marks ko 0-100% ke beech rakho
        predicted_marks = round(min(100, max(0, predicted_marks)), 2)

        # Grade logic
        if predicted_marks >= 90:
            grade, color = "A+ (Outstanding)", "#10b981"
        elif predicted_marks >= 75:
            grade, color = "A (Excellent)", "#3b82f6"
        elif predicted_marks >= 60:
            grade, color = "B (Good)", "#f59e0b"
        elif predicted_marks >= 40:
            grade, color = "C (Pass)", "#8b5cf6"
        else:
            grade, color = "F (Needs Work)", "#ef4444"

        return jsonify({
            'success': True,
            'marks': f"{predicted_marks}%",
            'grade': grade,
            'color': color
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(port=5001, debug=True)