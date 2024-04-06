# Machine Sound Anomaly Detector

![Python](https://img.shields.io/badge/python-%2314354C.svg?style=for-the-badge&logo=python&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

## Introduction

Anomaly detection is the process of identifying patterns, events, or observations that deviate significantly from the expected or normal behavior. However, anomaly detection is difficult since it is almost always impossible to define the meaning of abnormality, hence anomaly detection systems should be trained on normal samples only, making it an unsupervised problem. The dataset used is the [MIMII dataset](https://arxiv.org/pdf/1909.09347v1.pdf) which provides sounds for 4 different types of machines: industrial fans, water pumps, linear slide rails, and solenoid valves. It also provides us with four
different models of the machine for each machine type (denoted by ids 00, 02, 04, 06). In this project we will use Convolutional Autoencoders and use novelty detection on the latent space to detect anomalies in the sound samples.

## Previous Work

Previous work in this research used autoencoder models with the reconstruction loss as an anomaly score. There were also other successful methods such as GANs, SNN, and various types of CNNs such as mobile net-inspired architectures and ResNet. However we noticed that the previous approaches with the best results have augmented their architecture by using the specific machine ID to help in the anomaly score, which in our opinion invalidates their results. This is because it results in the model’s overfitting for the specific machines models given in the training set and wouldn’t generalize for all the machines of the same type that are not part of the dataset. In this project we will not use the machine ids in the training which would be more useful in real world scenarios as our trained model will work for any model of the machine type.

## Preprocessing

For neural networks to process audio data such as ours, we need to convert them to a spectrogram.
We can either use STFT and stop there or further process it to the mel scale. We plotted both and found that the mel spectrogram shows more information. For demonstration, below are 4 images of normal fan samples. On the left side STFT spectrograms are shown and on the right mel spectrograms of the same samples are shown.
![stft vs mel](https://raw.githubusercontent.com/Hussain-Aziz/Machine-Sound-Anomaly-Detector/master/imgs/stft_vs_mel.png)

Below is the flow chart of the preprocessing steps:
![preprocessing flowchart](https://raw.githubusercontent.com/Hussain-Aziz/Machine-Sound-Anomaly-Detector/master/imgs/preprocessing_flowchart.png)

## Models

We used convolutional autoencoders and used the reconstruction loss to train the model. We then detacted the latent space and used LocalOutlierFactor to detect the anomalies. We also experimented with variational autoencoders to get the latent space and one class SVM to detect the anomalies.

## Validation Methodology

For each machine (fan, pump, slider, valve), we took all the normal samples from all ids, shuffled and split them into a train and test set with an 80-20 split. We then took all the abnormal samples and split them into test and validation set with 80-20 split. We used the 80% normal train data in 5 fold cross validation and used the 20% abnormal validation data to help select the best model. We then used the 20% normal test data and 80% abnormal test data to evaluate the model to get the results

![validation_flow](https://raw.githubusercontent.com/Hussain-Aziz/Machine-Sound-Anomaly-Detector/master/imgs/validation_flow.png)

## Results

### Fan

Accuracy: 0.92
F1 Score: 0.919
AUC: 0.916
![fan_roc](https://raw.githubusercontent.com/Hussain-Aziz/Machine-Sound-Anomaly-Detector/master/imgs/fan_roc.png)
![fan_pr](https://raw.githubusercontent.com/Hussain-Aziz/Machine-Sound-Anomaly-Detector/master/imgs/fan_pr.png)
