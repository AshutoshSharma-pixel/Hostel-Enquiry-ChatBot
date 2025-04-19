#Reads intents.json
#Vectorizes questions
#rains an ML model (like Logistic Regression)
#savesthe model to chatbot_model.pkl
import json
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

with open('intents.json') as file:
    data = json.load(file)

x = []
y = []

for intent in data['intents']:  # ← this is the only change
    for pattern in intent['patterns']:
        x.append(pattern)
        y.append(intent['tag'])

# Vectorize and encode
vectorizer = TfidfVectorizer()
x_vec = vectorizer.fit_transform(x)

encoder = LabelEncoder()
y_enc = encoder.fit_transform(y)

# Train the model
model = LogisticRegression()
model.fit(x_vec, y_enc)

# Save everything to .pkl
with open("./chatbot_model.pkl", "wb") as f:
    pickle.dump((model, vectorizer, encoder), f)

print(" Model trained and saved as chatbot_model.pkl")

with open("chatbot_model.pkl", "wb") as f:
    pickle.dump((model, vectorizer, encoder), f)

print("Model trained and saved.")
