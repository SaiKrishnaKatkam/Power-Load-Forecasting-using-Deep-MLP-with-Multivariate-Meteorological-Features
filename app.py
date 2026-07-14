from flask import Flask, render_template, request, redirect, url_for, flash, session
import joblib
import torch
import numpy as np
from torch import nn

# Initialize Flask App
app = Flask(__name__)
app.secret_key = '12345y78'

# Mock user DB
users = {}

# Define MLP model architecture (same as training)
class MLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.model(x)

# Load scalers and model
scaler_x = joblib.load("scaler_x.save")
scaler_y = joblib.load("scaler_y.save")
model = MLP(input_size=17)
model.load_state_dict(torch.load("mlp_powerload_model.pt", map_location=torch.device('cpu')))
model.eval()

# ----- ROUTES -----
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/home')
def home():
    if 'user' in session:
        return render_template('home.html', user=session['user'])
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        if email in users and users[email] == password:
            session['user'] = email
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid email or password!', 'danger')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        if password != confirm_password:
            flash('Passwords do not match!', 'danger')
        elif email in users:
            flash('Email is already registered!', 'warning')
        else:
            users[email] = password
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/classification', methods=['GET', 'POST'])
def classification():
    if 'user' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        
        try:
            input_fields = [
                'T2M_toc', 'QV2M_toc', 'TQL_toc', 'W2M_toc',
                'T2M_san', 'QV2M_san', 'TQL_san', 'W2M_san',
                'T2M_dav', 'QV2M_dav', 'TQL_dav', 'W2M_dav',
                'Holiday_ID', 'holiday', 'school',
                'datetime_day', 'datetime_month', 'datetime_hour'
            ]
            input_data = [float(request.form[field]) for field in input_fields]
            print(input_data[0:17])
            features = np.array(input_data[0:17]).reshape(1, -1)
            scaled_input = scaler_x.transform(features)
            tensor_input = torch.tensor(scaled_input, dtype=torch.float32)
            with torch.no_grad():
                prediction = model(tensor_input).numpy()
            predicted_value = scaler_y.inverse_transform(prediction)[0][0]
            session['prediction'] = f"{predicted_value:.2f} MW"
            session['confidence'] = "High"
            
            return redirect(url_for('result'))
        except Exception as e:
            flash(f"Prediction Error: {str(e)}", "danger")
            return redirect(url_for('classification'))

    return render_template('classification.html', user=session['user'])

@app.route('/result')
def result():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template(
        'result.html',
        result=session.get('prediction'),
        confidence=session.get('confidence')
    )
@app.route('/charts')
def charts():
    if 'user' in session:
        return render_template('charts.html', user=session['user'])
    return redirect(url_for('login'))

@app.route('/performance')
def performance():
    if 'user' in session:
        return render_template('performance.html', user=session['user'])
    return redirect(url_for('login'))
@app.route('/logout')
def logout():
    session.pop('user', None)
    flash('You have been logged out!', 'info')
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
