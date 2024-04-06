from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from IPython.display import display
from .testing import plot_roc, plot_precision_recall, plot_3_vars
from sklearn.base import BaseEstimator
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
import pickle
import os
from typing import List

class BaseAnomalyDetector():
    """Base class for anomaly detection models."""
    def __init__(self, 
                classifier: BaseEstimator,
                train_embeddings: np.ndarray, 
                test_embeddings: np.ndarray, 
                y_test: np.ndarray,
                **kwargs):
        """
        Args:
            classifier (BaseEstimator): A scikit-learn compatible classifier.
            train_embeddings (np.ndarray): The embeddings of the training set.
            test_embeddings (np.ndarray): The embeddings of the test set.
            y_test (np.ndarray): The ground truth labels of the test set.
            **kwargs: arguments to pass to the classifier.
        """
        self.train_embeddings = train_embeddings
        self.test_embeddings = test_embeddings
        self.y_test = y_test
        self.clf = classifier(**kwargs)


    def predict(self):
        """
        Trains the classifier on the training embeddings and predicts the labels of the test embeddings.
        """
        self.clf.fit(self.train_embeddings)
        self.y_pred = self.clf.predict(self.test_embeddings)
        self.transform_y_pred(self.y_pred)
        
        return self.y_pred
    
    def test_predict(self, test_embeddings, y_test):
        """
        Predicts the labels of the test embeddings.
        """
        y_pred = self.clf.predict(test_embeddings)
        self.transform_y_pred(y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        display(pd.DataFrame(conf_matrix, index=['normal', 'abnormal'], columns=['normal', 'abnormal']))
        print(classification_report(y_test, y_pred, target_names=["normal", "abnormal"]))
        plot_roc(y_test.reshape(-1,1), y_pred.reshape(-1,1), ["N"])
        plot_precision_recall(y_test.reshape(-1,1), y_pred.reshape(-1,1), ['N'])

        return y_pred
        
    
    @property
    def model(self):
        """Returns the sklearn classifier. Will be trained if called after predict()."""
        return self.clf
    

    def save_model(self, path):
        """Saves the model to a file using pickle."""
        with open(os.path.join(path, 'classifier.model'), 'wb') as f:
            pickle.dump(self.model, f)


    @staticmethod
    def transform_y_pred(y_pred):
        '''Transforms y_pred from anomaly = -1, normal = 1 to anomaly = 1, normal = 0.'''
        y_pred[y_pred == 1] = 0 # make normal 0
        y_pred[y_pred == -1] = 1 # make anomaly 1
    

    def plot_metrics(self):
        '''Plots confusion matrix, classification report, ROC curve, and Precision-Recall curve. Needs to be called after predict() so that y_pred is available.'''
        conf_matrix = confusion_matrix(self.y_test, self.y_pred)
        display(pd.DataFrame(conf_matrix, index=['normal', 'abnormal'], columns=['normal', 'abnormal']))
        print(classification_report(self.y_test, self.y_pred, target_names=["normal", "abnormal"]))
        plot_roc(self.y_test.reshape(-1,1), self.y_pred.reshape(-1,1), ["N"])
        plot_precision_recall(self.y_test.reshape(-1,1), self.y_pred.reshape(-1,1), ['N'])


    def tune(self, 
             clf_name: str, 
             var1_name: str, 
             var2_name: str, 
             var1_range: List, 
             var2_range: List,
             ) -> pd.DataFrame:
        """Loops through all possible values of var1 and var2 and saves the results to a csv file. Used to find the best model hyperparameters
        Args:
            clf_name: The name of the classifier which is used in csv file name.
            var1_name: The name of the first hyperparameter which is used in plotting and for header in csv.
            var2_name: The name of the second hyperparameter which is used in plotting and for header in csv.
            var1_range: A list of possible values for the first hyperparameter.
            var2_range: A list of possible values for the second hyperparameter.
        
        """
        var1_data = []
        var2_data = []
        accuracy_data = []
        f1_weighted_data = []

        combined = [(var1, var2) for var1 in var1_range for var2 in var2_range]

        for var1, var2 in tqdm(combined):
            self.clf.fit(self.train_embeddings)
            y_pred = self.clf.predict(self.test_embeddings)
            self.transform_y_pred(y_pred)
            
            var1_data.append(var1)
            var2_data.append(var2)
            accuracy_data.append(np.mean(y_pred == self.y_test))
            f1_weighted_data.append(f1_score(self.y_test, y_pred, average='weighted'))
                

        df = pd.DataFrame({var1_name: var1_data, var2_name: var2_data, 'accuracy': accuracy_data, 'f1_weighted': f1_weighted_data})

        os.makedirs('tmp', exist_ok=True)
        df.to_csv(f"tmp/{clf_name}_hyperparams.csv", index=False)

        plot_3_vars(df, var1_name, var2_name, 'accuracy')
        plot_3_vars(df, var1_name, var2_name, 'f1_weighted')

        

class LocalOutlierFactorDetector(BaseAnomalyDetector):
    """Anomaly detector using Local Outlier Factor for novelty detection"""
    def __init__(self, 
                 train_embeddings: np.ndarray, 
                 test_embeddings: np.ndarray, 
                 y_test: np.ndarray,
                 contamination: float = 0.1, 
                 n_neighbors: int = 5,
                 loaded_model = None
                 ):
        """
        Creates an instance of the LocalOutlierFactorDetector class.
        if loaded_model is provided, the other args are ignored and also the predict function will not work.
        """
        if loaded_model:
            self.clf = loaded_model
        else:
            super().__init__(classifier=LocalOutlierFactor, 
                            train_embeddings=train_embeddings, 
                            test_embeddings=test_embeddings, 
                            y_test=y_test, 
                            contamination=contamination, 
                            n_neighbors=n_neighbors, 
                            novelty=True,
                            n_jobs=-1)

    def tune(self):
        n_neighbors_range = [i for i in range(1, 100)]
        contamination_range = [0.00000000001, 0.0000000001, 0.000000001, 0.00000001, 0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5]
        super().tune(clf_name='LOF',
                     var1_name='n_neighbors',
                     var2_name='contamination',
                     var1_range=n_neighbors_range,
                     var2_range=contamination_range)


class OneClassSVMDetector(BaseAnomalyDetector):
    """Anomaly detector using One-Class SVM"""
    def __init__(self, 
                 train_embeddings: np.ndarray, 
                 test_embeddings: np.ndarray, 
                 y_test: np.ndarray,
                 nu: float = 0.1, 
                 gamma: float = 0.01):
        
        super().__init__(classifier=OneClassSVM,
                         train_embeddings=train_embeddings,
                         test_embeddings=test_embeddings,
                         y_test=y_test,
                         nu=nu,
                         gamma=gamma,
                         kernel='rbf')
        

        
    def tune(self) -> pd.DataFrame:
        nu_range = [0.00000000001, 0.0000000001, 0.000000001, 0.00000001, 0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5, 0.9]
        gamma_range = [0.00000000001, 0.0000000001, 0.000000001, 0.00000001, 0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5, 1, 10, 50, 100]
        super().tune(clf_name='OneClassSVM',
                        var1_name='nu',
                        var2_name='gamma',
                        var1_range=nu_range,
                        var2_range=gamma_range)


class NearestNeighborsDetector(BaseAnomalyDetector):
    """Anomaly detector using Nearest Neighbors."""
    def __init__(self, 
                 train_embeddings: np.ndarray, 
                 test_embeddings: np.ndarray,
                 y_test: np.ndarray,
                 n_neighbors: int = 5,
                 threshold: int = 85,
                 use_sklearn: bool = False
                 ):
        
        self.train_embeddings = train_embeddings
        self.test_embeddings = test_embeddings
        self.y_test = y_test
        self.n_neighbors = n_neighbors
        self.threshold = threshold
        self.use_sklearn = use_sklearn
    
    def predict(self):

        # this is cursed but it allows our code can run in envs without cuml
        if self.use_sklearn:
            from sklearn.neighbors import NearestNeighbors
            knn = NearestNeighbors(n_neighbors=self.n_neighbors, metric='euclidean')
        else:
            import cuml
            knn = cuml.NearestNeighbors(n_neighbors=self.n_neighbors, metric='euclidean')
        
        knn.fit(X=self.train_embeddings)
        train_distances, _ = knn.kneighbors(self.train_embeddings)
        threshold = np.percentile(
            train_distances, threshold
        )
        
        self.distances, _ = knn.kneighbors(self.test_embeddings)
        anomaly_score = np.mean(self.distances, axis=1)

        print("\nanomaly_score:")
        print(f"min, max = {np.min(anomaly_score)}, {np.max(anomaly_score)}")
        print(f"mean = {np.mean(anomaly_score)}, median = {np.median(anomaly_score)}")
        self.y_pred = np.where(anomaly_score > threshold, 1, 0)
        
        return self.y_pred


    def plot_metrics(self):
        '''Needs to be called after predict() so that y_pred is available'''
        conf_matrix = confusion_matrix(self.y_test, self.y_pred)
        display(pd.DataFrame(conf_matrix, index=['normal', 'abnormal'], columns=['normal', 'abnormal']))
        print(classification_report(self.y_test, self.y_pred, target_names=["normal", "abnormal"]))
        plot_roc(self.y_test.reshape(-1,1), self.distances, ["N"])
        plot_precision_recall(self.y_test.reshape(-1,1), self.distances, ['N'])
        

    def tune(self) -> pd.DataFrame:
        '''
        Runs the Nearest Neighbors algorithm for multiple values of n_neighbors and thresholds and returns the results in a DataFrame.
        '''
        import cuml
        n_neighbors_range = [i for i in range(1, 700)]
        thresholds_range = [i for i in range(0, 101)]

        n_neighbors_data = []
        thresholds_data = []
        accuracy_data = []
        f1_abnormal_data = []
        f1_normal_data = []
        f1_weighted_data = []

        for n_neighbors in tqdm(n_neighbors_range):
            knn = cuml.NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
            knn.fit(self.train_embeddings)
            train_distances, _ = knn.kneighbors(self.train_embeddings)
            distances, _ = knn.kneighbors(self.test_embeddings)
            anomaly_score = np.mean(distances, axis=1)
            
            for thresh in thresholds_range:
                threshold = np.percentile(train_distances, thresh)
                y_pred = np.where(anomaly_score > threshold, 1, 0)
                
                n_neighbors_data.append(n_neighbors)
                thresholds_data.append(thresh)
                accuracy_data.append(np.mean(y_pred == self.y_test))
                f1_abnormal_data.append(f1_score(self.y_test, y_pred))
                f1_normal_data.append(f1_score(self.y_test, y_pred, pos_label=0))
                f1_weighted_data.append(f1_score(self.y_test, y_pred, average='weighted'))
                

        df = pd.DataFrame({'n_neighbors': n_neighbors_data, 'thresholds': thresholds_data, 'accuracy': accuracy_data, 'f1_abnormal': f1_abnormal_data, 'f1_normal': f1_normal_data, 'f1_weighted': f1_weighted_data})

        df.to_csv(f"tmp/NN_hyperparams.csv", index=False)
        plot_3_vars(df, 'n_neighbors', 'thresholds', 'accuracy', 'Accuracy')
        plot_3_vars(df, 'n_neighbors', 'thresholds', 'f1_weighted', 'F1 (Weighted)')
        return df