import joblib
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

final_model = joblib.load("best_model.pkl")
model = final_model["model"]
label_encoders = final_model["label_encoders"]

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return jsonify({"message": "Flask API is running on Render"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        print(f"🔹 Received Data: {data}")

        print(f"🔍 LabelEncoder for sex: {label_encoders['sex'].classes_}")
        print(f"🔍 LabelEncoder for island: {label_encoders['island'].classes_}")

        island_value = data["island"]

        island_mapping = {0: "Biscoe", 1: "Dream", 2: "Torgersen"}

        if isinstance(island_value, str):
            island_name = island_value
        elif isinstance(island_value, int):
            island_name = island_mapping.get(island_value, None)
        else:
            return jsonify({"error": "Invalid island value"}), 400
        
        if island_name not in label_encoders["island"].classes_:
            return jsonify({
                "error": f"Invalid island value: {island_name}, must be one of {list(label_encoders['island'].classes_)}"
            }), 400
        
        encoded_sex = label_encoders["sex"].transform([data["sex"]])[0]
        encoded_island = label_encoders["island"].transform([island_name])[0]

        print(f"Encoded sex: {encoded_sex}")
        print(f"Encoded island: {encoded_island}")

        features = [
            float(data["bill_length_mm"]),
            float(data["bill_depth_mm"]),
            float(data["flipper_length_mm"]),
            float(data["body_mass_g"]),
            encoded_sex,
            encoded_island
        ]
        print(f"Features: {features}")

        features_array = np.array([features]).reshape(1, -1)
        prediction = model.predict(features_array)
        species_predicted = label_encoders["species"].inverse_transform([prediction[0]])[0]

        print(f"Prediction: {species_predicted}")
        return jsonify({"prediction": species_predicted})

    except Exception as e:
        print(f"Server Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
