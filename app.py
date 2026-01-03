import os
import traceback
from flask import Flask, request, render_template, jsonify, session, redirect, url_for
from functools import wraps
from sqlalchemy import func
import torch
import torch.nn as nn
import numpy as np
from dotenv import load_dotenv
import mysql.connector
from mysql.connector import errorcode
from flask_sqlalchemy import SQLAlchemy
import torch.optim as optim
from flask import session, redirect, url_for
from functools import wraps

app = Flask(__name__)
app.secret_key = 'Admin@1234'  # Replace with a secure key

# Database setup
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///students.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Login protection decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('logged_in'):
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# User credentials (for demo purposes; replace with secure auth in production)
ADMIN_USERNAME = 'admin'
ADMIN_PASSWORD = 'adminpass'

# Routes for login/logout
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session['logged_in'] = True
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error='Invalid credentials')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))

# Main page (protected)
@app.route('/')
@login_required
def index():
    return render_template('index.html')

# Dashboard route (optional, can be same as index or separate)
@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('index.html')

# Data Model
class StudentData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    attendance_rate = db.Column(db.Float, nullable=False)
    previous_grade_avg = db.Column(db.Float, nullable=False)
    study_hours = db.Column(db.Float, nullable=False)
    assignments_completed = db.Column(db.Float, nullable=False)
    participation_score = db.Column(db.Float, nullable=False)
    predicted_performance = db.Column(db.Float, nullable=True)
    timestamp = db.Column(db.DateTime(timezone=True), server_default=func.now())

# ML Model Definition
class PerformanceModel(nn.Module):
    def __init__(self):
        super(PerformanceModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)

# Initialize model
model = PerformanceModel()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Generate training data function
def generate_training_data(n=1000):
    attendance = np.random.uniform(50, 100, n)
    prev_grade = np.random.uniform(50, 100, n)
    study_hours = np.random.uniform(0, 40, n)
    assignments = np.random.uniform(50, 100, n)
    participation = np.random.uniform(1, 10, n)

    X = np.stack([attendance, prev_grade, study_hours, assignments, participation], axis=1)
    y = (0.3 * attendance + 0.3 * prev_grade + 0.15 * study_hours + 0.15 * assignments + 0.1 * participation) / 100
    y = (y > 0.7).astype(float)

    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).unsqueeze(1)

# Train model
def train_model():
    model.train()
    X_train, y_train = generate_training_data(5000)
    epochs = 30
    batch_size = 64
    for epoch in range(epochs):
        permutation = torch.randperm(X_train.size()[0])
        for i in range(0, X_train.size()[0], batch_size):
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = X_train[indices], y_train[indices]

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
train_model()

# Routes for prediction and data
@app.route('/predict', methods=['POST'])
@login_required
def predict():
    try:
        data = request.json
        features = torch.tensor([
            [
                float(data['attendance_rate']),
                float(data['previous_grade_avg']),
                float(data['study_hours']),
                float(data['assignments_completed']),
                float(data['participation_score'])
            ]
        ], dtype=torch.float32)
        with torch.no_grad():
            pred = model(features).item()
        student_entry = StudentData(
            attendance_rate=data['attendance_rate'],
            previous_grade_avg=data['previous_grade_avg'],
            study_hours=data['study_hours'],
            assignments_completed=data['assignments_completed'],
            participation_score=data['participation_score'],
            predicted_performance=pred * 100
        )
        db.session.add(student_entry)
        db.session.commit()
        return jsonify({'prediction': round(pred * 100, 2)})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/summary')
@login_required
def summary():
    total_predictions = StudentData.query.count()
    avg_score = db.session.query(func.avg(StudentData.predicted_performance)).scalar() or 0
    at_risk_students = StudentData.query.filter(StudentData.predicted_performance < 70).count()
    model_accuracy = 0
    if total_predictions > 0:
        correct_preds = StudentData.query.filter(StudentData.predicted_performance >= 70).count()
        model_accuracy = (correct_preds / total_predictions) * 100
    return jsonify({
        'total_predictions': total_predictions,
        'average_score': round(avg_score, 2),
        'model_accuracy': round(model_accuracy, 2),
        'at_risk_students': at_risk_students
    })

@app.route('/performance_distribution')
@login_required
def performance_distribution():
    performances = [p.predicted_performance for p in StudentData.query.all()]
    bins = [0]*10
    for perf in performances:
        idx = min(int(perf // 10), 9)
        bins[idx] += 1
    return jsonify(bins)

@app.route('/recent_predictions')
@login_required
def recent_predictions():
    recent = StudentData.query.order_by(StudentData.timestamp.desc()).limit(5).all()
    results = [{
        'attendance_rate': r.attendance_rate,
        'previous_grade_avg': r.previous_grade_avg,
        'study_hours': r.study_hours,
        'assignments_completed': r.assignments_completed,
        'participation_score': r.participation_score,
        'predicted_performance': round(r.predicted_performance, 2),
        'timestamp': r.timestamp.strftime('%Y-%m-%d %H:%M:%S')
    } for r in recent]
    return jsonify(results)

# Run setup
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)