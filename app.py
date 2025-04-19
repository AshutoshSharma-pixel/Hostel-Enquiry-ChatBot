from flask import Flask, render_template, request, jsonify
import pickle
import json
import numpy as np
import random

# Load model, vectorizer, and encoder
with open("chatbot_model.pkl", "rb") as f:
    model, vectorizer, encoder = pickle.load(f)

# Load intents
with open("intents.json") as file:
    intents = json.load(file)

# Get a random response from matching tag
def get_response(tag):
    for intent in intents["intents"]: 
        if intent["tag"] == tag:
            return np.random.choice(intent.get("responses", ["No response found."]))
    return "Sorry, I did not understand."

# Flask app
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def chatbot_response():
    user_input = request.form["msg"]
    print(f"\nUser input: {user_input}")

    user_vec = vectorizer.transform([user_input])
    print(f"Vector shape: {user_vec.shape}")

    pred = model.predict(user_vec)
    print(f"Predicted class index: {pred}")

    intent_tag = encoder.inverse_transform(pred)[0]
    print(f"Predicted tag: {intent_tag}")

    response = get_response(intent_tag)
    print(f"Bot response: {response}")

    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
