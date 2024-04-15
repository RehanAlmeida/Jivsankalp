# app.py
from flask import Flask, render_template, request, jsonify
import os
import pyaudio
import wave
import librosa
import numpy as np
from tensorflow.keras.models import model_from_json
import pickle
import tensorflow_hub as hub

app = Flask(__name__)

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Manually download YAMNet model and provide the local path
yamnet_model_path = "C:/Users/RUDALPH/Documents/TE PBL/Disuploadcopy/archive"
yamnet_model = hub.load(yamnet_model_path)

# Define the function to extract YAMNet embeddings
def extract_yamnet_features(audio):
    # Use YAMNet to extract embeddings
    scores, embeddings, spectrogram = yamnet_model(audio)

    # Get the number of frames in the embeddings
    num_frames, embedding_dim = embeddings.shape

    # Define the desired number of frames (e.g., 20 frames)
    desired_frames = 20

    # Pad or truncate the embeddings to match the desired number of frames
    if num_frames < desired_frames:
        # If there are fewer frames, pad with zeros
        embeddings = np.pad(embeddings, ((0, desired_frames - num_frames), (0, 0)), mode='constant')
    elif num_frames > desired_frames:
        # If there are more frames, truncate to the desired number
        embeddings = embeddings[:desired_frames, :]

    # Convert embeddings to a NumPy array
    return embeddings

# Load the model architecture from the pickle file
with open('audio_model_architecture(2).pkl', 'rb') as arch_file:
    loaded_model_architecture = pickle.load(arch_file)

# Rebuild the model from the loaded architecture
loaded_model = model_from_json(loaded_model_architecture)

# Load the model weights
loaded_model.load_weights('audio_model_weights(2).h5')

# Define the class names (change this to match your class labels)
class_names = ["Dog","Elephant","Tiger"]  # Update with your class labels

# Define the maximum audio length (in seconds)
max_audio_length = 10

# Define the input shape that matches YAMNet embeddings
input_shape = (20, 1024)  # Assuming YAMNet embeddings are now padded/truncated to (20, 1024)

# Function to record audio from microphone and save as WAV file
def record_and_process_audio():
    CHUNK = 1024
    RECORD_SECONDS = max_audio_length
    sample_rate = 44100
    channels = 2
    format = pyaudio.paInt16

    audio = pyaudio.PyAudio()

    stream = audio.open(format=format,
                        channels=channels,
                        rate=sample_rate,
                        input=True,
                        frames_per_buffer=CHUNK)

    print("Recording...")
    frames = []

    for i in range(0, int(sample_rate / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Recording finished")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save recorded audio
    file_name = os.path.join('recordings', "audio_recorded.wav")
    wf = wave.open(file_name, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(audio.get_sample_size(format))
    wf.setframerate(sample_rate)
    wf.writeframes(b''.join(frames))
    wf.close()
    print(f"Audio saved as '{file_name}'")

    # Process the recorded audio
    audio, _ = librosa.load(file_name, sr=None, mono=True, duration=max_audio_length)
    audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)  # Resample to YAMNet's required sample rate

    # Extract YAMNet embeddings
    embeddings = extract_yamnet_features(audio)

    # Make predictions using the loaded model
    # Ensure that the YAMNet embeddings have the correct shape
    predictions = loaded_model.predict(np.expand_dims(embeddings, axis=0))
    predicted_class_index = np.argmax(predictions[0])

    # Get the predicted class label
    predicted_class = class_names[predicted_class_index]

    return predicted_class

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Record and process audio when the button is clicked
        predicted_class = record_and_process_audio()
        return jsonify({'predicted_class': predicted_class})
    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run(port=5001)
