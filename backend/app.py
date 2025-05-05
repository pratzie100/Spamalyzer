
# from flask import Flask, request, jsonify, render_template
# from flask_cors import CORS
# import joblib
# import pandas as pd
# import nltk
# import re
# import string
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# import logging
# import os

# # Set up logging
# logging.basicConfig(level=logging.DEBUG)

# # Initialize Flask app with template_folder set to root directory
# app = Flask(__name__, template_folder='.')  # '.' means root directory
# CORS(app)  # Enable CORS for deployment

# # Download NLTK resources
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

# # Load the trained model and vectorizer
# try:
#     svm_model = joblib.load('svm_model.pkl')
#     vectorizer = joblib.load('vectorizer.pkl')
#     logging.info("Model and vectorizer loaded successfully")
# except Exception as e:
#     logging.error(f"Error loading model or vectorizer: {str(e)}")
#     raise

# # Initialize text preprocessing tools
# lemmatizer = WordNetLemmatizer()
# stop_words = set(stopwords.words('english'))

# # Text cleaning function
# def clean_text(text):
#     try:
#         text = text.lower()
#         text = re.sub(r'\d+', '', text)
#         text = text.translate(str.maketrans('', '', string.punctuation))
#         words = text.split()
#         cleaned = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
#         return " ".join(cleaned)
#     except Exception as e:
#         logging.error(f"Error in clean_text: {str(e)}")
#         raise

# # # Route for the homepage
# # @app.route('/')
# # def home():
# #     return render_template('index.html')

# @app.route('/')
# def home():
#     return 'Spamalyzer API is up and running!'



# # Route for prediction
# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Get the message from the form
#         message = request.form.get('message')
#         if not message or message.strip() == '':
#             logging.warning("Empty message received")
#             return jsonify({'error': 'Message cannot be empty'}), 400
        
#         # Clean the message
#         cleaned_message = clean_text(message)
#         if not cleaned_message:
#             logging.warning("Cleaned message is empty")
#             return jsonify({'error': 'Message contains no valid words after cleaning'}), 400
        
#         # Vectorize the cleaned message
#         logging.debug(f"Cleaned message: {cleaned_message}")
#         message_vector = vectorizer.transform([cleaned_message])
        
#         # Predict
#         prediction = svm_model.predict(message_vector)[0]
        
#         # Convert prediction to label
#         result = 'Spam' if prediction == 1 else 'Ham'
#         logging.info(f"Prediction: {result}")
        
#         return jsonify({'prediction': result})
#     except Exception as e:
#         logging.error(f"Error in predict route: {str(e)}")
#         return jsonify({'error': str(e)}), 500

# # if __name__ == '__main__':
# #     app.run(debug=True)

# if __name__ == '__main__':
#     port = int(os.getenv('PORT', 5000))
#     app.run(host='0.0.0.0', port=port, debug=False)


from flask import Flask, request, jsonify
import pickle
import nltk
import os
import logging
import psutil
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Set NLTK data path
nltk.data.path.append('/app/nltk_data')

# Download NLTK data once during initialization
def download_nltk_data():
    try:
        nltk.download('punkt', download_dir='/app/nltk_data', quiet=True)
        nltk.download('stopwords', download_dir='/app/nltk_data', quiet=True)
        nltk.download('wordnet', download_dir='/app/nltk_data', quiet=True)
        logger.info("NLTK data loaded successfully")
    except Exception as e:
        logger.error(f"Error downloading NLTK data: {e}")

# Load model and vectorizer once during startup
try:
    with open('svm_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('vectorizer.pkl', 'rb') as vec_file:
        vectorizer = pickle.load(vec_file)
    logger.info("Model and vectorizer loaded successfully")
except Exception as e:
    logger.error(f"Error loading model/vectorizer: {e}")
    raise

# Initialize NLTK data
download_nltk_data()

# Log memory usage
def log_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    logger.info(f"Memory usage: {mem_info.rss / 1024 / 1024:.2f} MB")

# Text preprocessing function
def preprocess_text(text):
    try:
        # Tokenize
        tokens = word_tokenize(text.lower())
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        # Lemmatize
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        # Join tokens back to string
        cleaned_text = ' '.join(tokens)
        logger.debug(f"Cleaned message: {cleaned_text}")
        return cleaned_text
    except Exception as e:
        logger.error(f"Error preprocessing text: {e}")
        return text

# Health check endpoint
@app.route('/')
def home():
    log_memory_usage()
    return jsonify({'status': 'Backend is running'})

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        message = data.get('message', '')
        if not message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Preprocess the message
        cleaned_message = preprocess_text(message)
        
        # Transform and predict
        transformed_message = vectorizer.transform([cleaned_message])
        prediction = model.predict(transformed_message)
        result = 'Spam' if prediction[0] == 1 else 'Ham'
        
        logger.info(f"Prediction: {result}")
        log_memory_usage()
        
        return jsonify({'prediction': result})
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=False)