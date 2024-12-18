import os
import numpy as np
from flask import Flask, request, render_template
import joblib

# Initialize Flask app
app = Flask(__name__)

# Path to models folder
MODELS_FOLDER = "models"

def load_accuracies():
    accuracy_file = os.path.join(MODELS_FOLDER, "accuracylist.txt")
    accuracies = {}
    try:
        with open(accuracy_file, "r") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                try:
                    accuracy = float(line.strip())
                    accuracies[f"Model {i + 1}"] = accuracy
                except ValueError:
                    print(f"Invalid accuracy value on line {i + 1}: {line.strip()}")
    except FileNotFoundError:
        print("accuracylist.txt not found.")
    return accuracies


def load_all_models_and_scalers():
    models = {}
    scalers = {}
    for i in range(1, 101):  # Loop through all the joblib file
        model_path = os.path.join(MODELS_FOLDER, f"model_{i}.joblib")
        scaler_path = os.path.join(MODELS_FOLDER, f"scaler_{i}.joblib")
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            models[i] = joblib.load(model_path)
            scalers[i] = joblib.load(scaler_path)
    return models, scalers

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/member")
def member():
    return render_template("member.html")


@app.route("/predict", methods=["POST"])
def predict():
    models, scalers = load_all_models_and_scalers()
    accuracies = load_accuracies()

    # Retrieve input features from the form
    features = [float(request.form[key]) for key in request.form.keys()]

    # Initialize weighted variables
    weighted_sum = 0
    total_weight = 0

    # Initialize confidence trackers
    confident_0_sum = 0
    confident_0_weight = 0
    confident_1_sum = 0
    confident_1_weight = 0

    # Iterate over all models and make predictions
    for model_id, model in models.items():
        scaler = scalers[model_id]
        accuracy = accuracies.get(f"Model {model_id}", 0)

        scaled_features = scaler.transform([features])

        # Predict the probability for the target class (1: heart disease)
        prob_heart_disease = model.predict_proba(scaled_features)[0][1]
        prob_no_heart_disease = model.predict_proba(scaled_features)[0][0]

        weighted_sum += prob_heart_disease * accuracy
        total_weight += accuracy

        # Confidence calculations
        if prob_heart_disease >= 0.5:  # Model predicts 1
            confident_1_sum += prob_heart_disease * accuracy
            confident_1_weight += accuracy
        else:  # Model predicts 0
            confident_0_sum += prob_no_heart_disease * accuracy
            confident_0_weight += accuracy

    # Calculate the final weighted percentage
    if total_weight > 0:
        final_percentage = (weighted_sum / total_weight) * 100
    else:
        final_percentage = 0

    if final_percentage >= 50:
        # overall_prediction = "Heart disease detected"
        confidence_percentage = (confident_1_sum / confident_1_weight) * 100 if confident_1_weight > 0 else 0
        return render_template("gagal.html", confidence_percentage=confidence_percentage)
    else:
        # overall_prediction = "No heart disease detected"
        confidence_percentage = (confident_0_sum / confident_0_weight) * 100 if confident_0_weight > 0 else 0
        return render_template("berhasil.html", confidence_percentage=confidence_percentage)


# if __name__ == "__main__":
#     MODELS_FOLDER = "models"
#     accuracies = load_accuracies()
#     for model, accuracy in accuracies.items():
#         print(f"{model}: {accuracy}")

if __name__ == "__main__":
    app.run(debug=True)
