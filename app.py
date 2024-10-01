from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model (ensure model.pkl is in the same directory)
with open('fraud_detection_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the form for all 28 variables and other features
    time = float(request.form['Time'])
    amount = float(request.form['amount'])

    # Retrieve all 28 variables (v1, v2, ..., v28)
    variables = []
    for i in range(1, 29):
        variables.append(float(request.form[f'v{i}']))

    # Prepare the complete data set (ensure order matches the model's feature set)
    input_features = np.array([[time, amount] + variables])

    # Make prediction using the loaded model
    prediction = model.predict(input_features)[0]

    # Convert prediction to human-readable format
    result = "Fraudulent" if prediction == 1 else "Not Fraudulent"

    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
