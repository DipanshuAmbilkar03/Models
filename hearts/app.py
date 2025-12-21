from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# ---- Load model artifacts ----
with open("knn_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("feature_columns.pkl", "rb") as f:
    columns = pickle.load(f)

# ---- Routes ----
@app.route("/")
def index():
    return render_template("index.html")
import pandas as pd

@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_data = dict.fromkeys(columns, 0)

        # Numerical
        input_data["Age"] = float(request.form["Age"])
        input_data["RestingBP"] = float(request.form["RestingBP"])
        input_data["Cholesterol"] = float(request.form["Cholesterol"])
        input_data["FastingBS"] = int(request.form["FastingBS"])
        input_data["MaxHR"] = float(request.form["MaxHR"])
        input_data["Oldpeak"] = float(request.form["Oldpeak"])

        # Categorical → One-hot
        if request.form["Sex"] == "M":
            input_data["Sex_M"] = 1

        cp = request.form["ChestPainType"]
        if cp in ["ATA", "NAP", "TA"]:
            input_data[f"ChestPainType_{cp}"] = 1

        ecg = request.form["RestingECG"]
        if ecg in ["Normal", "ST"]:
            input_data[f"RestingECG_{ecg}"] = 1

        if request.form["ExerciseAngina"] == "Y":
            input_data["ExerciseAngina_Y"] = 1

        slope = request.form["ST_Slope"]
        if slope in ["Flat", "Up"]:
            input_data[f"ST_Slope_{slope}"] = 1

        # 🔥 Create DataFrame with feature names
        X = pd.DataFrame([input_data], columns=columns)

        X_scaled = scaler.transform(X)
        prediction = model.predict(X_scaled)[0]

        return jsonify({
            "prediction": int(prediction),
            "result": "Heart Disease Detected" if prediction == 1 else "No Heart Disease"
        })

    except Exception as e:
        print("ERROR:", e)
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
