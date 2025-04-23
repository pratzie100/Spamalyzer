
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import pandas as pd
import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Initialize Flask app with template_folder set to root directory
app = Flask(__name__, template_folder='.')  # '.' means root directory
CORS(app)  # Enable CORS for deployment

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the trained model and vectorizer
try:
    svm_model = joblib.load('svm_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    logging.info("Model and vectorizer loaded successfully")
except Exception as e:
    logging.error(f"Error loading model or vectorizer: {str(e)}")
    raise

# Initialize text preprocessing tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Text cleaning function
def clean_text(text):
    try:
        text = text.lower()
        text = re.sub(r'\d+', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        words = text.split()
        cleaned = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
        return " ".join(cleaned)
    except Exception as e:
        logging.error(f"Error in clean_text: {str(e)}")
        raise

# # Route for the homepage
# @app.route('/')
# def home():
#     return render_template('index.html')

@app.route('/')
def home():
    return 'Spamalyzer API is up and running!'


# Add new /ping endpoint
@app.route('/ping')
def ping():
    return jsonify({'status': 'alive'})


# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the message from the form
        message = request.form.get('message')
        if not message or message.strip() == '':
            logging.warning("Empty message received")
            return jsonify({'error': 'Message cannot be empty'}), 400
        
        # Clean the message
        cleaned_message = clean_text(message)
        if not cleaned_message:
            logging.warning("Cleaned message is empty")
            return jsonify({'error': 'Message contains no valid words after cleaning'}), 400
        
        # Vectorize the cleaned message
        logging.debug(f"Cleaned message: {cleaned_message}")
        message_vector = vectorizer.transform([cleaned_message])
        
        # Predict
        prediction = svm_model.predict(message_vector)[0]
        
        # Convert prediction to label
        result = 'Spam' if prediction == 1 else 'Ham'
        logging.info(f"Prediction: {result}")
        
        return jsonify({'prediction': result})
    except Exception as e:
        logging.error(f"Error in predict route: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)