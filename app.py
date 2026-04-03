from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow frontend requests from different domain/port

# Example route
@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json  # Receive JSON from frontend
    number = data.get('number', 0)
    result = number * 2  # Example: some computation
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)
