from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(BASE_DIR, 'templates')

app = Flask(__name__, template_folder=template_dir)

# Safely load your model assets using absolute path rules
model_path = os.path.join(BASE_DIR, 'flood_model.pkl')
scaler_path = os.path.join(BASE_DIR, 'scaler.pkl')

with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

with open(scaler_path, 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # 1. Grab inputs from the HTML form elements
            annual_rainfall = float(request.form['annual_rainfall'])
            cloud_visibility = float(request.form['cloud_visibility'])
            monsoon_rainfall = float(request.form['monsoon_rainfall'])
            
            # 2. Build a structured DataFrame matching your exact training dimensions
            input_df = pd.DataFrame([{
                'annual_rainfall': annual_rainfall,
                'cloud_visibility': cloud_visibility,
                'monsoon_rainfall': monsoon_rainfall
            }])
            
            # 3. Transform inputs and pull the classification index
            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)[0]
            
            # 4. Pick UI warning styles based on the machine learning choice
            if int(prediction) == 1:
                result_text = "⚠️ HIGH RISK: Conditions indicate a strong probability of flooding! Initiate emergency safety protocols."
                alert_class = "danger"
            else:
                result_text = "✅ LOW RISK: Weather parameters are within safe limits. No immediate flooding predicted."
                alert_class = "success"
                
            return render_template('index.html', prediction_text=result_text, alert_class=alert_class)
            
        except Exception as e:
            # If anything breaks, show the exact error message on screen instead of a blank page
            return render_template('index.html', prediction_text=f"⚠️ System Error: {str(e)}", alert_class="warning")

if __name__ == "__main__":
    app.run(debug=True, port=9000)