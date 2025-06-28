from flask import Flask, render_template, request, jsonify
import joblib
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load('spam_classifier.joblib')
vectorizer = joblib.load('tfidf_vectorizer.joblib')

# Preprocessing function
def preprocess_text(text):
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = word_tokenize(text)
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    email = request.form['email']
    processed_text = preprocess_text(email)
    features = vectorizer.transform([processed_text])
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]
    return jsonify({
        'is_spam': bool(prediction),
        'probability': float(probability),
        'text': email
    })

if __name__ == '__main__':
    app.run(debug=True)