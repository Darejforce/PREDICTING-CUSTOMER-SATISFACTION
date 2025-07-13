from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load model and preprocessing tools
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, 'best_xgb_model.pkl'))
scaler = joblib.load(os.path.join(BASE_DIR, 'scaler.pkl'))
label_encoder = joblib.load(os.path.join(BASE_DIR, 'label_encoder.pkl'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect data from form
        total_spend = float(request.form['total_spend'])
        items_purchased = int(request.form['items_purchased'])
        average_rating = float(request.form['average_rating'])
        days_since_last_purchase = int(request.form['days_since_last_purchase'])
        discount_applied = float(request.form['discount_applied'])
        gender = 1 if request.form['gender'] == 'Male' else 0
        membership = request.form['membership']

        # Encode membership manually (adjust to match your training encoding)
        membership_map = {'Silver': 0, 'Gold': 1, 'Platinum': 2}
        membership_encoded = membership_map.get(membership, 0)  # Default to Silver if not found

        # Create input for model
        input_features = [[
            total_spend,
            items_purchased,
            average_rating,
            days_since_last_purchase,
            discount_applied,
            gender,
            membership_encoded
        ]]

        # Scale input
        input_scaled = scaler.transform(input_features)

        # Predict
        prediction = model.predict(input_scaled)[0]

        # Decode predicted class label
        prediction_label = label_encoder.inverse_transform([prediction])[0]

        return render_template('result.html', prediction=prediction_label)

    except Exception as e:
        return f"<h3>⚠️ Error: {e}</h3>"

if __name__ == '__main__':
    app.run(debug=True)