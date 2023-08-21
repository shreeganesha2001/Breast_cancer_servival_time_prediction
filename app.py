from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the model and threshold from the pickle file
model_and_threshold = joblib.load('model_and_threshold.pkl')
loaded_model = model_and_threshold['model']
loaded_threshold = model_and_threshold['threshold']

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None

    if request.method == 'POST':
        # Get input values from HTML form
        input_values = [float(request.form['age']), float(request.form['no_auxiliary_nodes'])]

        # Predict using the loaded model
        pred_prob = loaded_model.predict_proba([input_values])[0, 1]

        # Compare with the loaded threshold
        if pred_prob >= loaded_threshold:
            prediction = 'Positive'
        else:
            prediction = 'Negative'

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)