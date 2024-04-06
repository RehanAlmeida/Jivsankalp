# from flask import Flask, request, jsonify, render_template
# import pickle
# import numpy as np
# from keras.preprocessing import image
# from keras.applications.inception_v3 import preprocess_input
# from keras.models import model_from_json
# import base64
# import cv2
# import pyttsx3

# app = Flask(__name__)

# # Load the model architecture from the pickle file
# with open('inception_v3_architecture.pkl', 'rb') as arch_file:
#     loaded_model_architecture = pickle.load(arch_file)

# # Rebuild the model from the loaded architecture
# loaded_model = model_from_json(loaded_model_architecture)

# # Load the model weights from an HDF5 file
# loaded_model.load_weights('inception_v3_model.h5')

# # Define the number of classes in your model
# num_classes = 25

# def initialize_engine():
#     # Initialize the text-to-speech engine
#     return pyttsx3.init()

# # Initialize the text-to-speech engine
# engine = initialize_engine()

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     if request.method == 'POST':
#         img_data_url = request.json.get('imgDataUrl', None)
#         if img_data_url:
#             # Extract image data from the data URL
#             img_data = base64.b64decode(img_data_url.split(',')[1])

#             # Convert the image data to a numpy array
#             img_array = np.frombuffer(img_data, np.uint8)

#             # Decode the image using OpenCV
#             img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

#             # Convert the image to the correct target size
#             img = cv2.resize(img, (299, 299))
#             img_array = image.img_to_array(img)
#             img_array = np.expand_dims(img_array, axis=0)

#             # Preprocess the image using InceptionV3-specific preprocessing
#             img_array = preprocess_input(img_array)

#             # Make predictions using the loaded model
#             predictions = loaded_model.predict(img_array)

#             # Get the top predicted class label and confidence
#             top_class_index = np.argmax(predictions[0])
#             confidence = predictions[0][top_class_index]

#             percentage = round(confidence * 100, 2)

#             # Check if matching percentage is above 75%
#             if percentage > 75:
#                 # Speak a voice alert
#                 engine.say(f"Matching percentage is {percentage} percent. The animal may be endangered.")
#                 engine.runAndWait()

#             return jsonify({
#                 'class': str(top_class_index),
#                 'percentage': str(percentage) + '%'  # Rounding to 2 decimal places
#             })

#     return render_template('index.html')

# if __name__ == '__main__':
#     app.run(debug=True)



import time  # Import time module for sleeping
import requests
from flask import Flask, jsonify, render_template, request
import pickle
import numpy as np
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input
from keras.models import model_from_json
import base64
import cv2
import pyttsx3
import threading

app = Flask(__name__)

# Load the model architecture from the pickle file
with open('inception_v3_architecture.pkl', 'rb') as arch_file:
    loaded_model_architecture = pickle.load(arch_file)

# Rebuild the model from the loaded architecture
loaded_model = model_from_json(loaded_model_architecture)

# Load the model weights from an HDF5 file
loaded_model.load_weights('inception_v3_model.h5')

# Define the number of classes in your model
num_classes = 25

def initialize_engine():
    # Initialize the text-to-speech engine
    return pyttsx3.init()

# Initialize the text-to-speech engine
engine = initialize_engine()

def speak_alert(percentage):
    engine.say(f"Matching percentage is {percentage} percent. The animal may be endangered.")
    engine.runAndWait()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        img_data_url = request.json.get('imgDataUrl', None)
        if img_data_url:
            # Extract image data from the data URL
            img_data = base64.b64decode(img_data_url.split(',')[1])

            # Convert the image data to a numpy array
            img_array = np.frombuffer(img_data, np.uint8)

            # Decode the image using OpenCV
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            # Convert the image to the correct target size
            img = cv2.resize(img, (299, 299))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)

            # Preprocess the image using InceptionV3-specific preprocessing
            img_array = preprocess_input(img_array)

            # Make predictions using the loaded model
            predictions = loaded_model.predict(img_array)

            # Get the top predicted class label and confidence
            top_class_index = np.argmax(predictions[0])
            confidence = predictions[0][top_class_index]

            percentage = round(confidence * 100, 2)

            latitude, longitude = None, None

            # Check if matching percentage is above 75%
            if percentage > 75:
                latitude, longitude = get_current_location()
                # Speak a voice alert in a separate thread
                threading.Thread(target=speak_alert, args=(percentage,), daemon=True).start()

            return jsonify({
                'class': str(top_class_index),
                'percentage': str(percentage) + '%',  # Rounding to 2 decimal places
                'latitude': latitude,
                'longitude': longitude
            })

    return render_template('image.html')

def get_current_location():
    try:
        # Fetching location based on IP address using ipinfo.io
        response = requests.get("https://ipinfo.io/json")
        if response.status_code == 200:
            location_data = response.json()
            latitude, longitude = location_data['loc'].split(',')
            return float(latitude), float(longitude)
        else:
            print("Error fetching location data.")
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None

if __name__ == '__main__':
    app.run(debug=True)
