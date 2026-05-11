from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Load the trained model
model_path = 'concrete_strength_predict.pkl'
with open(model_path, 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get form data
            cement = float(request.form['cement'])
            slag = float(request.form['slag'])
            fly_ash = float(request.form['fly_ash'])
            water = float(request.form['water'])
            superplasticizer = float(request.form['superplasticizer'])
            coarse_agg = float(request.form['coarse_agg'])
            fine_agg = float(request.form['fine_agg'])
            age = int(request.form['age'])
            
            # Create input array
            input_data = np.array([[cement, slag, fly_ash, water, superplasticizer, 
                                   coarse_agg, fine_agg, age]])
            
            # Make prediction
            prediction = model.predict(input_data)[0]
            
            # Determine strength category
            if prediction < 20:
                category = "Low Strength"
                badge_class = "badge bg-danger"
            elif prediction < 40:
                category = "Medium Strength"
                badge_class = "badge bg-warning text-dark"
            elif prediction < 60:
                category = "High Strength"
                badge_class = "badge bg-success"
            else:
                category = "Ultra High Strength"
                badge_class = "badge bg-primary"
            
            return render_template('predict.html', 
                                 prediction=round(prediction, 2),
                                 category=category,
                                 badge_class=badge_class,
                                 form_data=request.form)
        
        except Exception as e:
            return render_template('predict.html', 
                                 error=f"Error in prediction: {str(e)}")
    
    return render_template('predict.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json()
        input_data = np.array([[
            data['cement'],
            data['slag'],
            data['fly_ash'],
            data['water'],
            data['superplasticizer'],
            data['coarse_agg'],
            data['fine_agg'],
            data['age']
        ]])
        prediction = model.predict(input_data)[0]
        return jsonify({'prediction': round(prediction, 2)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)