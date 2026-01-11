from flask import Flask, render_template, request
import pickle
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load("personality_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get form values
    Time_spent_Alone = int(request.form["hours_alone"])
    Stage_fear = int(request.form["stage_fear"])   # already 1 or 0
    Social_event_attendance = int(request.form["social_events"])
    Going_outside = int(request.form["days_outside"])
    Drained_after_socializing = int(request.form["drained"])         # already 1 or 0
    Friends_circle_size = int(request.form["friend_circles"])
    Post_frequency = int(request.form["social_posts"])
    
    features = np.array([[ 
        Time_spent_Alone,
        Stage_fear,
        Social_event_attendance,
        Going_outside,
        Drained_after_socializing,
        Friends_circle_size,
        Post_frequency
    ]])

    # Prediction
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    confidence = f"{max(probabilities) * 100:.2f}"

    if prediction == 1:
        result = f"You are an Extrovert ({confidence}% confidence)"
    else:
        result = f"You are an Introvert ({confidence}% confidence)"

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=False)
