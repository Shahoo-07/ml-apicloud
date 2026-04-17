from flask import Flask, request, jsonify
import joblib

application = Flask(__name__)


# Load model
model = joblib.load('sentiment_model.joblib')

@application.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if not data or 'text' not in data:
        return jsonify({'error': 'Invalid input. Send JSON with "text" key'}), 400

    text = data['text']

    prediction = model.predict([text])[0]

    return jsonify({
        'input_text': text,
        'sentiment_prediction': prediction,
        'model_version': '1.1' 
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)