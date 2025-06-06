from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

app = Flask(__name__)
CORS(app)

# Load model and data
model = load_model("tesla_stock_model.h5")
data = pd.read_csv("tesla_stock_data.csv")

# Prepare training data
data_X = np.array(data[["Open", "High", "Low", "Volume"]])
data_y = np.array(data["Close"])
X_train, _, y_train, _ = train_test_split(data_X, data_y, test_size=0.3, random_state=42)

# Normalize input features
scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train)

# Store mean and std of target variable for inverse normalization
y_train_mean = np.mean(y_train)
y_train_std = np.std(y_train)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_data = request.get_json()
        print("Received data:", input_data)

        features = [[
            float(input_data["open"]),
            float(input_data["high"]),
            float(input_data["low"]),
            float(input_data["volume"])
        ]]

        features_scaled = scaler.transform(features)
        pred_norm = model.predict(features_scaled).flatten()[0]
        predicted_close = pred_norm * y_train_std + y_train_mean

        print(f"Predicted close: {predicted_close:.2f}")
        return jsonify({"predicted": round(predicted_close, 2)})

    except Exception as e:
        print("Prediction error:", e)
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
