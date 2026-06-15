from flask import Flask
from flask import render_template
from flask import request

import pickle
import re

from nltk.corpus import stopwords

app = Flask(__name__)

model = pickle.load(
    open(
        "models/healthcare_model.pkl",
        "rb"
    )
)

vectorizer = pickle.load(
    open(
        "models/vectorizer.pkl",
        "rb"
    )
)

stop_words = set(
    stopwords.words('english')
)

def clean_text(text):

    text = text.lower()

    text = re.sub(
        r'[^a-zA-Z ]',
        ' ',
        text
    )

    words = text.split()

    words = [
        word
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

@app.route('/')

def home():

    return render_template(
        "index.html"
    )

@app.route('/predict', methods=["POST"])

def predict():

    news = request.form['news']

    cleaned = clean_text(news)

    vector = vectorizer.transform([cleaned])

    prediction = model.predict(vector)[0]

    confidence = max(
        model.predict_proba(vector)[0]
    )

    return render_template(
        'index.html',
        prediction=prediction,
        confidence=round(confidence * 100, 2),
        news=news
    )

if __name__ == "__main__":

    app.run(debug=True)