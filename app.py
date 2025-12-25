
from flask import Flask, request, jsonify
import pandas as pd
import joblib
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


# Load all models
log_model = joblib.load("mental_health_log_model.pkl")
rf_model = joblib.load("mental_health_rf_model.pkl")
xgb_model = joblib.load("mental_health_xgb_model.pkl")
lgb_model = joblib.load("mental_health_lgbm_model.pkl")

# Encoders
encoders = {
    "Gender": {"Male": 1, "Female": 0},
    "Country": {
        "United States": 0, "United Kingdom": 1, "Canada": 2, "Australia": 3,
        "India": 4, "Germany": 5, "France": 6, "Netherlands": 7,
        "Sweden": 8, "Ireland": 9, "Brazil": 10, "New Zealand": 11,
        "South Africa": 12, "Switzerland": 13, "Israel": 14, "Italy": 15,
        "Belgium": 16, "Denmark": 17, "Singapore": 18,
        "Finland": 19, "Portugal": 20, "Nigeria": 21, "Philippines": 22
    },
    "Occupation": {"Student": 0, "Corporate": 1, "Business": 2, "Housewife": 3, "Others": 4},
    "self_employed": {"No": 0, "Yes": 1},
    "family_history": {"No": 0, "Yes": 1},
    "Mental_Health_History": {"No": 0, "Maybe": 1, "Yes": 2},
    "Days_Indoors": {
        "Go Out Every Day": 0, "1-14 days": 1, "15-30 days": 2,
        "31-60 days": 3, "More than 2 months": 4
    },
    "Growing_Stress": {"No": 0, "Maybe": 1, "Yes": 2},
    "Changes_Habits": {"No": 0, "Maybe": 1, "Yes": 2},
    "Mood_Swings": {"Low": 0, "Medium": 1, "High": 2},
    "Coping_Struggles": {"No": 0, "Maybe": 1, "Yes": 2},
    "Work_Interest": {"No": 0, "Maybe": 1, "Yes": 2},
    "Social_Weakness": {"No": 0, "Maybe": 1, "Yes": 2},
    "mental_health_interview": {"No": 0, "Yes": 1},
    "care_options": {"No": 0, "Not sure": 1, "Yes": 2}
}

FEATURE_ORDER = [
    "Gender", "Country", "Occupation", "self_employed", "family_history",
    "Days_Indoors", "Growing_Stress", "Changes_Habits",
    "Mental_Health_History", "Mood_Swings", "Coping_Struggles",
    "Work_Interest", "Social_Weakness",
    "mental_health_interview", "care_options"
]

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    encoded = {f: encoders[f][data[f]] for f in FEATURE_ORDER}
    X = pd.DataFrame([encoded])[FEATURE_ORDER]

    p1 = log_model.predict_proba(X)[0][1]
    p2 = rf_model.predict_proba(X)[0][1]
    p3 = xgb_model.predict_proba(X)[0][1]
    p4 = lgb_model.predict_proba(X)[0][1]

    avg_prob = (p1 + p2 + p3 + p4) / 4
    prediction = 1 if avg_prob >= 0.5 else 0

    return jsonify({
        "prediction": prediction,
        "probability": round(avg_prob * 100, 2)
    })

if __name__ == "__main__":
    app.run(debug=True)
