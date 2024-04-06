'''
A script that takes a 10 second audio clip of a machine and classifies it as normal or abnormal.

Usage: python3 predictor.py Dataset/fan/id_00/normal/00000000.wav 
'''

import time
start_program = time.time()

import silence_tensorflow.auto # removes unnecessary tensorflow logging
import os
import pickle
import warnings
import argparse
import numpy as np
import librosa
import tensorflow as tf

warnings.filterwarnings("ignore", message="Trying to unpickle estimator") # ignore warnings from sklearn

# ask for the path to the wav file as an argument
parser = argparse.ArgumentParser(description='A script that takes a 10 second audio clip of a machine and classifies it as normal or abnormal.')
parser.add_argument('wavfile', type=str, help='the path to the wav file which wants to be classified')
parser.add_argument('--show_time', action='store_true', help='show the time taken for the program to run')
args = parser.parse_args()
data = args.wavfile
show_time = args.show_time

# check if the required files exists
required_files = ['encoder.tflite', 'classifier.model', '_min.npy', '_max.npy']
if not all([os.path.exists(file) for file in required_files]):
    print("Please make sure the following files are in the same directory: encoder.tflite, classifier.model, _min.npy, _max.npy")
    exit()

# check if wav file exists and is a wav file
if not os.path.exists(data) or not data.endswith('.wav'):
    print("Please make sure the audio clip exists")
    exit()

start_processing_and_loading = time.time()

# load persistent data
_min = np.load(os.path.join('_min.npy'))
_max = np.load(os.path.join('_max.npy'))

# load models
encoder = tf.lite.Interpreter(model_path='encoder.tflite')
with open('classifier.model', 'rb') as f:
    detector = pickle.load(f)

start_processing = time.time()

# process data
signal, sampling_rate = librosa.load(data, sr=None)
processed_data = librosa.feature.melspectrogram(y=signal, sr=sampling_rate, n_fft=1024, hop_length=512)
processed_data = librosa.power_to_db(processed_data, ref=np.max)
processed_data = processed_data.reshape(1, *processed_data.shape, 1).astype("float32")
processed_data = np.pad(processed_data, ((0, 0), (0, 0), (0, 7), (0, 0)), mode="constant")
processed_data = (processed_data - _min) / (_max - _min)

# get latent space embeddings from encoder
encoder.allocate_tensors()
input_details = encoder.get_input_details()
output_details = encoder.get_output_details()
encoder.set_tensor(input_details[0]['index'], processed_data)
encoder.invoke()
embeddings = encoder.get_tensor(output_details[0]['index'])
embeddings = embeddings.reshape(embeddings.shape[0], -1)

# use anomaly detection model
y = detector.predict(embeddings)
prediction = "normal" if y[0] == 1 else "abnormal"

# print results
print(f"prediction: {prediction}")

if show_time:
    # calculate time taken for the program to run
    end = time.time()
    duration_program = end - start_program
    duration_processing_and_loading = end - start_processing_and_loading
    duration_processing = end - start_processing

    print(f"durations: ")
    print(f"\tfull program: {duration_program:.2f}s, ")
    print(f"\tprocessing and loading: {duration_processing_and_loading:.2f}s, ")
    print(f"\tprocessing: {duration_processing:.2f}s")
