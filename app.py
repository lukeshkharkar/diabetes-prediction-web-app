from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

# Load the trained model and scaler
model = joblib.load('model.pkl')  # Replace with the correct path to your model
scaler = joblib.load('scaler.pkl')  # Replace with the correct path to your scaler

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    # Render the HTML form
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input data from the form
        input_data = [float(x) for x in request.form.values()]
        final_features = np.array(input_data).reshape(1, -1)

        # Standardize the input features
        standardized_features = scaler.transform(final_features)

        # Predict the outcome
        prediction = model.predict(standardized_features)
        output = 'Diabetic' if prediction[0] == 1 else 'Non-Diabetic'

        # Render the result on the same page
        return render_template('index.html', prediction_text=f'Result: The person is {output}.')
    except Exception as e:
        # Handle any errors gracefully
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)
