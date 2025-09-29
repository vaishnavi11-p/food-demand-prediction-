import streamlit as st
import json
from datetime import datetime
import boto3
import sagemaker
from sagemaker.predictor import Predictor
from sagemaker.serializers import IdentitySerializer
from sagemaker.deserializers import JSONDeserializer

# -------------------------------
# 1) Configuration
# -------------------------------
ENDPOINT_NAME = "food-demand"  # your SageMaker endpoint

DAY_MAPPING = {"friday": 0, "monday": 1, "saturday": 2, "sunday": 3, "thursday": 4, "tuesday": 5, "wednesday": 6}
WEATHER_MAPPING = {"cloudy": 0, "rainy": 1, "sunny": 2}
DISH_MAPPING = {"biryani": 0, "burger": 1, "pasta": 2, "pizza": 3, "salad": 4, "sandwich": 5}

# -------------------------------
# 2) Streamlit UI
# -------------------------------
st.title("🍽️ Restaurant Demand Prediction with Chatbot")
st.write("Select day, weather, and dish to predict demand and ask questions.")

day = st.selectbox("Day of Week", list(DAY_MAPPING.keys()))
weather = st.selectbox("Weather", list(WEATHER_MAPPING.keys()))
dish = st.selectbox("Dish", list(DISH_MAPPING.keys()))

# -------------------------------
# 3) SageMaker Prediction
# -------------------------------
if st.button("Predict Demand"):
    day_code = DAY_MAPPING[day]
    weather_code = WEATHER_MAPPING[weather]
    dish_code = DISH_MAPPING[dish]

    payload = {
        "instances": [{
            "start": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "target": [0],
            "cat": [day_code, weather_code, dish_code]
        }],
        "configuration": {
            "num_samples": 50,
            "output_types": ["mean", "quantiles"],
            "quantiles": ["0.1", "0.9"]
        }
    }

    predictor = Predictor(
        endpoint_name=ENDPOINT_NAME,
        sagemaker_session=sagemaker.Session(),
        serializer=IdentitySerializer(content_type="application/json"),
        deserializer=JSONDeserializer()
    )

    resp = predictor.predict(json.dumps(payload))
    pred = resp["predictions"][0]

    mean_val = int(round(pred["mean"][0]))
    low = int(round(pred["quantiles"]["0.1"][0]))
    high = int(round(pred["quantiles"]["0.9"][0]))

    st.success(f"Mean Predicted Sales: {mean_val}")
    st.info(f"Low Estimate (P10): {low}, High Estimate (P90): {high}")

    # Store prediction for chatbot
    st.session_state["last_prediction"] = {
        "dish": dish,
        "day": day,
        "mean": mean_val,
        "low": low,
        "high": high
    }

# -------------------------------
# 4) Offline Rule-Based Chatbot
# -------------------------------
def local_chatbot(question, prediction):
    q = question.lower()
    d = prediction['dish']
    day = prediction['day']
    mean = prediction['mean']
    low = prediction['low']
    high = prediction['high']

    if "mean" in q or "expected" in q:
        return f"The mean predicted sales for {d} on {day} is {mean}."
    elif "low" in q:
        return f"The low estimate (P10) for {d} on {day} is {low}."
    elif "high" in q:
        return f"The high estimate (P90) for {d} on {day} is {high}."
    elif "range" in q or "estimate" in q:
        return f"The predicted range for {d} on {day} is {low}–{high} (P10–P90), mean {mean}."
    else:
        return f"The predicted mean for {d} on {day} is {mean}, range {low}-{high}."

# -------------------------------
# 5) Chatbot UI
# -------------------------------
user_question = st.text_input("Ask about the forecast:")

if st.button("Ask Chatbot") and user_question:
    if "last_prediction" not in st.session_state:
        st.warning("Please make a prediction first!")
    else:
        predicted_demand = st.session_state["last_prediction"]
        answer = local_chatbot(user_question, predicted_demand)
        st.write("🤖 Chatbot Answer:")
        st.write(answer)