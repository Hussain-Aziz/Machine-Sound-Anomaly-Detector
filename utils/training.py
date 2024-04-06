from keras.models import Model
from keras.callbacks import Callback
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from tqdm import tqdm
from typing import Callable, Any
from typing import Literal
import numpy as np
from .anomaly_detection import BaseAnomalyDetector

class SaveBestModel(Callback):
    def __init__(self, metric, mode=Literal['min', 'max']):
        super().__init__()
        self.metric = metric
        self.mode = mode
        if self.mode == 'max':
            self.best = float('-inf')
        else:
            self.best = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        metric_value = logs[self.metric]
        if self.mode == 'max':
            if metric_value > self.best:
                self.best = metric_value
                self.best_model = self.model
        else:
            if metric_value < self.best:
                self.best = metric_value
                self.best_model = self.model
    
    def get_best_model(self):
        return self.best_model

def cross_validation(create_model: Callable[[float, Any], Model], 
                    X, 
                    y, 
                    k=5, 
                    batch_size=16, 
                    epochs=50, 
                    lr=0.001, 
                    include_accuracy:bool=False,
                    verbose=1
                    ):
    results = dict()
    best_model = dict()
    metrics = ['accuracy'] if include_accuracy else None
    for train_i, test_i in tqdm(KFold(n_splits=k).split(X, y), total=k):
        
        model_copy = create_model(lr, metrics)
        
        hist = model_copy.fit(
            X[train_i], y[train_i], batch_size=batch_size, epochs=epochs, verbose=verbose, validation_data=(X[test_i], y[test_i])
        )
        scores = model_copy.evaluate(X[test_i], y[test_i], verbose=0)

        if not isinstance(scores, list):
            scores = [scores]

        for metric, value in zip(model_copy.metrics_names, scores):
                results.setdefault(metric, []).append(value)

        if results["loss"][-1] == min(results["loss"]):
            best_model["model"] = model_copy
            best_model["train_i"] = train_i
            best_model["test_i"] = test_i
            best_model["history"] = hist
            for metric, value in zip(model_copy.metrics_names, scores):
                best_model[metric] = results[metric][-1]

    return results, best_model 


def cnn_and_detector_cross_validation(
        create_model: Callable[[float, Any], Model], 
        get_encoder: Callable[[Model], Model],
        detector_class: BaseAnomalyDetector,
        X_normal_train: np.ndarray,
        X_abnormal_validation: np.ndarray
        ) -> tuple[tuple[Model, BaseAnomalyDetector], list[float], list[tuple[np.ndarray, np.ndarray]]]:
    
    f1_scores: float = []
    y_preds: list[tuple[np.ndarray, np.ndarray]] = []
    encoders: list[Model] = []
    detectors: list[BaseAnomalyDetector] = []

    y_normal = np.zeros(X_normal_train.shape[0])
    y_abnormal = np.ones(X_abnormal_validation.shape[0])
    for train_i, test_i in tqdm(KFold(n_splits=5).split(X_normal_train, y_normal), total=5):
        
        X_test = np.concatenate((X_normal_train[test_i], X_abnormal_validation))
        y_test = np.concatenate((y_normal[test_i], y_abnormal))
        model_copy = create_model(0.001, None)
        
        model_copy.fit(X_normal_train[train_i], X_normal_train[train_i], batch_size=32, epochs=20, verbose=0, validation_split=0.1)
        
        encoder = get_encoder(model_copy)

        train_embeddings = encoder.predict(X_normal_train[train_i], verbose=0)
        train_embeddings = train_embeddings.reshape(train_embeddings.shape[0], -1)
        test_embeddings = encoder.predict(X_test, verbose=0)
        test_embeddings = test_embeddings.reshape(test_embeddings.shape[0], -1)

        detector = detector_class(train_embeddings, test_embeddings, y_test)

        y_pred = detector.predict()

        f1_scores.append(f1_score(y_test, y_pred, average='weighted'))
        y_preds.append((y_test, y_pred))
        encoders.append(encoder)
        detectors.append(detector)

    best_score_index = np.argmax(f1_scores)
    best_model = (encoders[best_score_index], detectors[best_score_index])
    return (best_model, f1_scores, y_preds)
        

        

    