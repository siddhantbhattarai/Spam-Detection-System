from flask import Flask, request, jsonify, render_template
import joblib
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load the model and vectorizer
model_path = os.path.join('models', 'svc_model.pkl')
vectorizer_path = os.path.join('models', 'tfidf_vectorizer.pkl')
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        message = request.form['message']
        
        # Preprocess the input message
        message_vectorized = vectorizer.transform([message])
        
        # Make prediction
        prediction = model.predict(message_vectorized)[0]
        
        # Get decision function value
        decision_value = model.decision_function(message_vectorized)[0]
        
        # Convert decision value to a confidence score
        confidence = 1 / (1 + np.exp(-decision_value))

        if prediction == 'ham':
            result = {
                'classification': 'Not Spam',
                'confidence': f'{confidence:.2%}',
                'message': 'This message appears to be legitimate.',
                'advice': 'No action needed.'
            }
        else:
            result = {
                'classification': 'Spam',
                'confidence': f'{confidence:.2%}',
                'message': 'This message has been classified as spam.',
                'advice': 'Exercise caution with this message. Consider reporting or deleting it.'
            }

        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=3000)