from flask import Flask, render_template, request
import base64
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import MobileNetV2
import json

app = Flask(__name__)

# Load the trained model
model = load_model('model.h5')

with open('class_labels.json', 'r') as f:
    class_labels = json.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    img_data = request.form['image']
    img_data = img_data.split(',')[1]  # Remove the header
    img_data = base64.b64decode(img_data)
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)

    # Make prediction
    prediction = model.predict(img)
    predicted_class_idx = np.argmax(prediction)
    predicted_class_label = class_labels[str(predicted_class_idx)]

    return predicted_class_label

if __name__ == '__main__':
    app.run(debug=True)
