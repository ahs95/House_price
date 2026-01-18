# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 19:18:31 2023

@author: ahs95
"""

from flask import Flask, request, jsonify, render_template
app = Flask(__name__)
import pickle
with open('House_price_model.pkl', 'rb') as file:
    model = pickle.load(file)


@app.route('/predict', methods=['GET','POST'])
def predict():
    # Get the JSON data from the request
    data = request.get_json()

    # Extract the features from the JSON data and preprocess them
    beds = data['beds']
    bath = data['bath']

    # Perform feature engineering: Multiply the "bed" and "bath" columns
    features = [[beds * bath]]

    # Use your model to make predictions
    prediction = model.predict(features)

    # Format the prediction result as needed
    response = {'prediction': prediction[0]}  # Assuming prediction is a 1D array
    return jsonify(response)
if __name__ == '__main__':
    app.run(debug=True)


    

