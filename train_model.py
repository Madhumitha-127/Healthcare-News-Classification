import pandas as pd
import pickle
import re
import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

df = pd.read_csv(
    "dataset/healthcare_newsdataset.csv"
)

stop_words = set(stopwords.words('english'))

def clean_text(text):

    text = str(text)

    text = text.lower()

    text = re.sub(r'[^a-zA-Z ]',' ',text)

    words = text.split()

    words = [
        word
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

df["clean_text"] = (
    df["text"]
    .apply(clean_text)
)

X = df["clean_text"]

y = df["label"]

vectorizer = TfidfVectorizer(
    max_features=3000,
    ngram_range=(1,2)
)

X_vec = vectorizer.fit_transform(X)

model = LogisticRegression(
    max_iter=1000
)

model.fit(X_vec,y)

pickle.dump(
    model,
    open(
        "models/healthcare_model.pkl",
        "wb"
    )
)

pickle.dump(
    vectorizer,
    open(
        "models/vectorizer.pkl",
        "wb"
    )
)

print("Model Saved Successfully")