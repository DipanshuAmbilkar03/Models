from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open("./model/model.pkl", "rb"))
scaler = pickle.load(open("./model/scaler.pkl", "rb"))

# Feature order MUST match training
FEATURES = [
    'radius_mean',
    'texture_mean',
    'perimeter_mean',
    'area_mean',
    'compactness_mean',
    'concavity_mean',
    'concave points_mean',
    'radius_worst',
    'texture_worst',
    'perimeter_worst',
    'area_worst',
    'concavity_worst',
    'concave points_worst'
]

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        try:
            values = [float(request.form[f]) for f in FEATURES]

            # Convert to array
            X = np.array(values).reshape(1, -1)

            # Apply scaling
            X_scaled = scaler.transform(X)

            # Predict
            result = model.predict(X_scaled)[0]
            proba = model.predict_proba(X)[0]
            print("Benign:", proba[0], "Malignant:", proba[1])

            prediction = "Malignant (Cancerous)" if result == 1 else "Benign (Non-Cancerous)"

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template("index.html", features=FEATURES, prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
