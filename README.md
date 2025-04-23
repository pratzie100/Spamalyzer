# Spamalyzer

Spamalyzer is a web application that classifies messages as **Spam** or **Ham** (non-spam) using a Support Vector Machine (SVM) model trained on SMS Spam Collection Dataset (Kaggle). Built with a **Flask** backend and a **Bootstrap**-powered frontend, it provides a user-friendly interface to detect spam messages with real-time predictions. The app is deployed using **Netlify** (frontend) and **Render** for backend. 

## Features
- Robust text preprocessing using **NLTK** (tokenization, lemmatization, stopword removal).
- Machine learning model powered by **scikit-learn** (SVM with TF-IDF vectorization).
- Real-time predictions via a RESTful API.

## 📩 Spam Example

> **Congratulations!**  
> You have been selected as the lucky winner of our international lottery!  
> To claim your **$1,000,000** prize, please reply with your full name, address, phone number, and bank account details.  
> **ACT FAST** – this offer expires in 24 hours!  
>
> *Yours sincerely,*  
> *Mr. Smith*  
> *International Lottery Coordinator*  

**🧠 Prediction:** `SPAM🚫`

---

## ✅ Non-Spam Example

> Hi friend,  
>  
> Just a quick reminder about our meeting scheduled for tomorrow at 10 AM. Let me know if you’d like to reschedule.  
> Looking forward to catching up and discussing the project updates.  
>  
> Best regards,  
> Pratyush

**🧠 Prediction:** `HAM✅`


## Deployment
- **Frontend**: Hosted on Netlify ([Live Demo](https://spamalyzer.netlify.app)).
- **Backend**: Hosted on ([Render](https://spam-classifier-backend.onrender.com/)).

## Technologies Used
- **Frontend**:
  - **HTML5**: Structure of the web page.
  - **Bootstrap 5**: Responsive styling and UI components.
  - **JavaScript (Fetch API)**: Asynchronous requests to the backend.
- **Backend**:
  - **Flask**: Lightweight Python web framework for the API.
  - **Gunicorn**: WSGI server for production deployment.
  - **Flask-CORS**: Handles cross-origin requests for Netlify/Render communication.
- **Machine Learning**:
  - **scikit-learn**: SVM model and TF-IDF vectorization.
  - **NLTK**: Text preprocessing (lemmatization, stopword removal).
  - **pandas**: Data handling during model training.
  - **joblib**: Model and vectorizer serialization.
- **Deployment**:
  - **Netlify**: Static frontend hosting.
  - **Render**: Backend hosting with Python environment.
  - **GitHub**: Version control and repository hosting.

## Setup (Local Development)
To run Spamalyzer locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/pratzie100/Spamalyzer.git
   cd Spamalyzer
   pip install -r requirements.txt
   python app.py
2. **Access the app**:
    - update url in javascript's fetch method of index.html to your localhost url.
    - Open localhost url in your browser.


