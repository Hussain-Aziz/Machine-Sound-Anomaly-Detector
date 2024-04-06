import pandas as pd
import os
import numpy as np
from typing import Literal
from sklearn.model_selection import train_test_split
from .models import BaseAutoEncoder
from .anomaly_detection import LocalOutlierFactorDetector
import pickle

machine_types = Literal["fan", "pump", "slider", "valve"]

def load_full_clips(machine_type: machine_types,
                    directory: str = "./Preprocessed Dataset/",
                    match_test_sizes: bool = True, 
                    padding: int = 7):
    """
    Load full 10 seconds clips for a machine type.
    
    args:
    machine_type: the machine type to load (fan, pump, slider, valve)
    directory: the directory where the preprocessed data is stored
    match_test_sizes: if true set the X_normal_test size to be equal to the X_abnormal size. if false, X_normal_test size will be 0.2 of X_normal
    padding: the number of zeros to pad the 3rd dimention with to make it more divisible (depends on the nftt and hop rate chosen during preprocessing)
    """
    
    if os.path.isdir(directory) == False:
        raise ValueError(f"Invalid directory path. {directory} is not a valid directory.")
    
    if os.path.exists("./Dataset/metadata.csv") == False:
        raise ValueError(f"metadata.csv file not found in ./Dataset/ directory.")
    
    
    metadata = pd.read_csv("./Dataset/metadata.csv")
    samples = (
        (metadata["machine_type"] == machine_type)
    )

    normal_samples = (samples & (metadata["condition"] == "normal"))
    abnormal_samples = (samples & (metadata["condition"] == "abnormal"))
    X_normal = np.array(
    [
        np.load(
            os.path.join(
                directory, os.path.splitext(file_name)[0] + ".npy"
            )
        )
        for file_name in metadata[normal_samples]["file_name"]
    ]
    )

    X_normal = X_normal.reshape(*X_normal.shape, 1).astype("float32")

    # padding to make the shape divisible by 2
    X_normal = np.pad(X_normal, ((0, 0), (0, 0), (0, padding), (0, 0)), mode="constant")

    X_abnormal = np.array(
        [
            np.load(
                os.path.join(
                    directory, os.path.splitext(file_name)[0] + ".npy"
                )
            )
            for file_name in metadata[abnormal_samples]["file_name"]
        ]
    )

    X_abnormal = X_abnormal.reshape(*X_abnormal.shape, 1).astype("float32")

    # padding to make the shape divisible by 2
    X_abnormal = np.pad(X_abnormal, ((0, 0), (0, 0), (0, padding), (0, 0)), mode="constant")

    test_size = len(X_abnormal)/len(X_normal) if match_test_sizes else 0.2

    X_normal_train, X_normal_test = train_test_split(X_normal, test_size=test_size)

    X_test = np.concatenate([X_normal_test, X_abnormal])
    y_test = np.concatenate([np.zeros(X_normal_test.shape[0]), np.ones(X_abnormal.shape[0])])

    _min, _max = np.min(X_normal_train), np.max(X_normal_train)

    X_normal_train = (X_normal_train - _min) / (_max - _min)
    X_test = (X_test - _min) / (_max - _min)
    X_normal = (X_normal - _min)/(_max-_min)
    X_normal_test = (X_normal_test - _min) / (_max - _min)
    X_abnormal = (X_abnormal - _min) / (_max - _min)
    
    X = np.concatenate([X_normal, X_abnormal])
    y = np.concatenate([np.zeros((len(X_normal),)), np.ones(len(X_abnormal))])

    print(f"X_train shape: {X_normal_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    
    return X_normal, X_abnormal, X_normal_train, X_normal_test, X_test, y_test, X, y, _min, _max


def save_run(machine_type: machine_types, 
             model_type: str,
             encoder,
             detector,
             X_normal_train,
             X_normal_test,
             X_abnormal_validation,
             X_abnormal_test,
             _min,
             _max,
             f1_score: float,
             auc_score: float,
             ):
    """
    Saves the encoder, detector, x, and y to the session directory
    Saves the tflite encoder, detector, min and max to the tflite directory

    Usage: save_run(machine_type, model_type, encoder, detector, X_normal_train, X_normal_test, X_abnormal_validation, X_abnormal_test, _min, _max)
    """

    session_save_dir = f'Saved_Models/{machine_type}_{model_type}/session'
    tflite_save_dir = f'Saved_Models/{machine_type}_{model_type}/tflite'
    
    os.makedirs(session_save_dir, exist_ok=True)
    os.makedirs(tflite_save_dir, exist_ok=True)

    BaseAutoEncoder.save_session(encoder, X_normal_train, X_normal_test, X_abnormal_validation, X_abnormal_test, session_save_dir)
    detector.save_model(session_save_dir)

    detector.save_model(tflite_save_dir)
    BaseAutoEncoder.save_lite_model(encoder, tflite_save_dir)
    np.save(os.path.join(tflite_save_dir, '_min.npy'), _min)
    np.save(os.path.join(tflite_save_dir, '_max.npy'), _max)

    with open(os.path.join(session_save_dir, 'info.txt'), 'w') as f:
        f.write(f'f1_score: {f1_score}\nauc_score: {auc_score}')


def load_run(machine_type: machine_types, model_type: str):
    """
    Loads the encoder, detector, x, and y from the session directory. Note the detector loaded is the sklearn classifier and not a BaseAnomalyDetector object
    """

    session_save_dir = f'Saved_Models/{machine_type}_{model_type}/session'

    if not os.path.exists(session_save_dir):
        return False

    encoder, X_normal_train, X_normal_test, X_abnormal_validation, X_abnormal_test, X_test, y_test = BaseAutoEncoder.load_session(session_save_dir)
    with open(os.path.join(session_save_dir, 'classifier.model'), 'rb') as f:
        detector_clf = pickle.load(f)

    detector = LocalOutlierFactorDetector(None, None, None, loaded_model=detector_clf)

    return encoder, detector, X_normal_train, X_normal_test, X_abnormal_validation, X_abnormal_test, X_test, y_test