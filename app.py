from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# ✅ Allow only your frontend origin
CORS(app, resources={r"/predict": {"origins": "https://web-production-9ac0.up.railway.app"}})

# Load model and minimal data for scaling
model = load_model("tesla_stock_model.h5")

# Load and prepare data
try:
    data = pd.read_csv("tesla_stock_data.csv", usecols=["Open", "High", "Low", "Volume", "Close"])
    data_X = data[["Open", "High", "Low", "Volume"]].values
    data_y = data["Close"].values

    # Use a small training subset to avoid memory issues
    X_train, _, y_train, _ = train_test_split(data_X, data_y, test_size=0.9, random_state=42)

    scaler = StandardScaler()
    scaler.fit(X_train)

    y_train_mean = np.mean(y_train)
    y_train_std = np.std(y_train)
except Exception as e:
    print("Error loading CSV or fitting scaler:", e)
    scaler = None
    y_train_mean = 0
    y_train_std = 1

@app.route("/")
def index():
    return "Tesla Stock Predictor is running on Railway!"

@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        # ✅ Handle preflight requests
        response = jsonify({"message": "CORS preflight"})
        response.headers.add("Access-Control-Allow-Origin", "https://web-production-9ac0.up.railway.app")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type")
        response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
        return response

    try:
        input_data = request.get_json()
        print("Received:", input_data)

        features = [[
            float(input_data["open"]),
            float(input_data["high"]),
            float(input_data["low"]),
            float(input_data["volume"])
        ]]

        features_scaled = scaler.transform(features)
        pred_norm = model.predict(features_scaled).flatten()[0]
        predicted_close = pred_norm * y_train_std + y_train_mean

        response = jsonify({"predicted": round(predicted_close, 2)})
        response.headers.add("Access-Control-Allow-Origin", "https://web-production-9ac0.up.railway.app")
        return response

    except Exception as e:
        print("Prediction error:", e)
        response = jsonify({"error": str(e)})
        response.headers.add("Access-Control-Allow-Origin", "https://web-production-9ac0.up.railway.app")
        return response, 400

if __name__ == "__main__":
    app.run(debug=True)
