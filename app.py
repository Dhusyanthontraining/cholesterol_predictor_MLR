from flask import Flask, request, jsonify
import joblib
from flask_cors import CORS
import pandas as pd

app = Flask(__name__)
CORS(app)

# Load model and feature list
model = joblib.load("cholesterol.pkl")
feature_cols = joblib.load("feature_cols.pkl")

@app.route("/predict", methods=["POST"])
def predict():

    # Get JSON data from user
    data = request.get_json()

    # Convert to dataframe in correct order
    df = pd.DataFrame([[data["age"],
                        data["weight_kg"],
                        data["daily_walking_km"],
                        data["exercise_hours"]]],
                      columns=feature_cols)

    # Make prediction
    prediction = model.predict(df)[0]

    # Return result
    return jsonify({
        "predicted_cholesterol": float(prediction)
    })


@app.route("/", methods=["GET"])
def home():
    return {"message": "Cholesterol Predictor API is running!"}


if __name__ == "__main__":
    app.run(debug=True)
