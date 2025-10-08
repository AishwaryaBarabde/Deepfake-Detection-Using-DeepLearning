import os
import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import cv2
from mtcnn import MTCNN

# Suppress TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Flask setup
app = Flask(__name__, static_folder='static')
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load your custom CNN model
model_path = os.path.join(os.getcwd(), 'model', 'cnn_model.h5')
if os.path.exists(model_path):
    model = load_model(model_path)
else:
    raise FileNotFoundError(f"Model not found at {model_path}")

# Face detection and cropping using MTCNN
def detect_and_crop_face(img_path):
    img = cv2.imread(img_path)
    detector = MTCNN()
    results = detector.detect_faces(img)

    if results:
        x, y, width, height = results[0]['box']
        face = img[y:y+height, x:x+width]
        face = cv2.resize(face, (128, 128))
        return face
    else:
        return cv2.resize(img, (128, 128))  # fallback if no face detected

# Preprocessing function
def preprocess_face(face):
    img_array = image.img_to_array(face)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# Predict using the loaded model
def predict_image(img_path):
    face = detect_and_crop_face(img_path)
    processed_image = preprocess_face(face)
    prediction = model.predict(processed_image)
    confidence = float(prediction[0][0])
    if confidence < 0.5:
        return "Real", 1 - confidence
    else:
        return "Fake", confidence

@app.route('/')
def index():
    return render_template('index.html')  # Ensure this file exists in templates/

@app.route('/detect', methods=['POST'])
def detect_deepfake():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    image_path = os.path.join(UPLOAD_FOLDER, image_file.filename)
    image_file.save(image_path)

    try:
        result, confidence = predict_image(image_path)
        os.remove(image_path)
        return jsonify({"result": result, "confidence": round(confidence, 4)}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
