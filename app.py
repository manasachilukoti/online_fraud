# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 14:31:59 2024

@author: VYSHNAVI
"""

import numpy as np
from flask import Flask, request, render_template
import pickle

app=Flask(__name__)


# Load the model
model = pickle.load(open('models/trained_model.sav', 'rb'))



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Parse input data
    features = []
    for key in ['step', 'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']:
        if key in ['step', 'type']:  # For integer inputs
            features.append(int(request.form[key]))
        else:
            features.append(float(request.form[key]))

    # Predict
    prediction = model.predict(np.array(features).reshape(1, -1))
    print("Raw prediction:", prediction)
    prediction_text = 'FRAUD' if prediction[0] == 1 else 'NOT FRAUD'

    # Render template with prediction result
    return render_template('index.html', prediction_text=prediction_text)


    
if __name__ =="__main__":
    app.run()
    