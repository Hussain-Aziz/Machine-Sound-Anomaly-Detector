from keras.models import Model, load_model
from keras import backend as K
from keras.optimizers import Adam
from keras.layers import Input, Conv2D, UpSampling2D, Conv2DTranspose, MaxPooling2D, Flatten, Dense, Reshape, Lambda, Dense, Flatten, BatchNormalization, Dropout
import numpy as np
from typing import Literal
import os

machine_types = Literal["fan", "pump", "slider", "valve"]

class BaseAutoEncoder():
    def __init__(self, machine_type: machine_types, input_shape) -> None:
        self.machine_type = machine_type
        self.input_shape = input_shape

    @staticmethod
    def get_model_type():
        pass
    

    @staticmethod
    def get_encoder(model):
        pass

    @staticmethod
    def save_lite_model(model, save_path):
        from tensorflow import lite
        converter = lite.TFLiteConverter.from_keras_model(model)

        tflite_model = converter.convert()

        # Save the model.
        with open(os.path.join(save_path, 'encoder.tflite'), 'wb') as f:
            f.write(tflite_model)
    
    @staticmethod
    def save_session(encoder: Model,
                     X_normal_train,
                     X_normal_test,
                     X_abnormal_validation,
                     X_abnormal_test,
                     save_path: str,
                     ):
        '''
        Saves all the necessary data to recreate the model later.
        usage: save_session(encoder, X_normal_train, X_normal_test, X_abnormal_validation, X_abnormal_test, save_path)
        '''

        os.makedirs(save_path, exist_ok=True)
        encoder.save(f"{save_path}/encoder.keras")
        np.save(f"{save_path}/X_normal_train.npy", X_normal_train)
        np.save(f"{save_path}/X_normal_test.npy", X_normal_test)
        np.save(f"{save_path}/X_abnormal_validation.npy", X_abnormal_validation)
        np.save(f"{save_path}/X_abnormal_test.npy", X_abnormal_test)

    @staticmethod    
    def load_session(save_path: str):
        
        if not os.path.exists(f"{save_path}/encoder.keras"):
            return False
        
        encoder = load_model(f"{save_path}/encoder.keras", safe_mode=False)
        X_normal_train = np.load(f"{save_path}/X_normal_train.npy")
        X_normal_test = np.load(f"{save_path}/X_normal_test.npy")
        X_abnormal_validation = np.load(f"{save_path}/X_abnormal_validation.npy")
        X_abnormal_test = np.load(f"{save_path}/X_abnormal_test.npy")

        X_test = np.concatenate([X_normal_test, X_abnormal_test])
        y_test = np.concatenate([np.zeros(X_normal_test.shape[0]), np.ones(X_abnormal_test.shape[0])])
        
        return encoder, X_normal_train, X_normal_test, X_abnormal_validation, X_abnormal_test, X_test, y_test
        

class CNNAutoEncoders(BaseAutoEncoder):
    
    @staticmethod
    def get_encoder(model):
        return Model(model.inputs, model.get_layer(name="latent_space").output)
    

    @staticmethod
    def get_model_type():
        return "non_vae"
    
    
    def create_conv2d_model(self, lr=0.001, metrics=['accuracy'], loss = 'mse') -> Model:
        _input = Input(shape=self.input_shape, name="input")

        h = Conv2D(16, kernel_size=(8, 8), activation="relu", padding="same")(_input)
        h = MaxPooling2D(pool_size=(2, 2), padding="same")(h)

        h = Conv2D(32, kernel_size=(6, 6), activation="relu", padding="same")(h)
        h = MaxPooling2D(pool_size=(2, 2), padding="same")(h)

        h = Conv2D(64, kernel_size=(4, 4), activation="relu", padding="same")(h)
        h = MaxPooling2D(pool_size=(2, 2), padding="same")(h)


        h = Conv2D(64, kernel_size=(2, 2), activation="relu", padding='same')(h)
        latent_space = MaxPooling2D(pool_size=(2, 2), padding="same", name="latent_space")(h)

        h = Conv2DTranspose(64, strides=2, kernel_size=(4, 4), activation="relu", padding="same")(latent_space)

        h = Conv2DTranspose(32, strides=2, kernel_size=(6, 6), activation="relu", padding="same")(h)

        h = Conv2DTranspose(16, strides=2, kernel_size=(8, 8), activation="sigmoid", padding="same")(h)

        output = Conv2DTranspose(1, strides=2, kernel_size=(1, 1), activation="sigmoid", padding="same")(h)

        model = Model(_input, output, name="model")
        
        model.compile(loss=loss, optimizer=Adam(learning_rate=lr), metrics=metrics)
        
        return model
    
    
    def create_upsampling_model(self, lr=0.001, metrics=['accuracy'], loss = 'mse') -> Model:
        _input = Input(shape=self.input_shape, name="input")

        h = Conv2D(16, kernel_size=(8, 8), activation="relu", padding="same")(_input)
        h = MaxPooling2D(pool_size=(2, 2), padding="same")(h)

        h = Conv2D(32, kernel_size=(6, 6), activation="relu", padding="same")(h)
        h = MaxPooling2D(pool_size=(2, 2), padding="same")(h)

        h = Conv2D(64, kernel_size=(4, 4), activation="relu", padding="same")(h)
        h = MaxPooling2D(pool_size=(2, 2), padding="same")(h)


        h = Conv2D(64, kernel_size=(2, 2), activation="relu", padding='same')(h)
        latent_space = MaxPooling2D(pool_size=(2, 2), padding="same", name="latent_space")(h)
        
        h = Conv2D(64, kernel_size=(4, 4), activation="relu", padding="same")(latent_space)
        h = UpSampling2D((2, 2))(h)
        
        h = Conv2D(32, kernel_size=(6, 6), activation="relu", padding="same")(h)
        h = UpSampling2D((2, 2))(h)
        
        h = Conv2D(16, kernel_size=(8, 8), activation="sigmoid", padding="same")(h)
        h = UpSampling2D((2, 2))(h)
        
        h = Conv2D(1, kernel_size=(1, 1), activation="sigmoid", padding="same")(h)
        output = UpSampling2D((2, 2), name="output")(h)

        model = Model(_input, output, name="model")
        
        model.compile(loss=loss, optimizer=Adam(learning_rate=lr), metrics=metrics)
        
        return model


class VariationalAutoEncoder(BaseAutoEncoder):
    
    @staticmethod
    def get_encoder(model):
        return Model(model.inputs, model.get_layer(name="encoder").output[2])
    

    @staticmethod
    def get_model_type():
        return "vae"
    
    
    def create_vae_model(self, lr=0.001, metrics=['accuracy'], loss = 'mse') -> Model:
        _input = Input(shape=self.input_shape, name="input")

        h = Conv2D(16, kernel_size=(8, 8), activation="relu", padding="same")(_input)
        h = MaxPooling2D(pool_size=(2, 2), padding="same")(h)

        h = Conv2D(32, kernel_size=(6, 6), activation="relu", padding="same")(h)
        h = MaxPooling2D(pool_size=(2, 2), padding="same")(h)

        h = Conv2D(64, kernel_size=(4, 4), activation="relu", padding="same")(h)
        h = MaxPooling2D(pool_size=(2, 2), padding="same")(h)

        h = Conv2D(64, kernel_size=(2, 2), activation="relu", padding='same')(h)
        h = MaxPooling2D(pool_size=(2, 2), padding="same")(h)

        h = Flatten()(h)
        latent_dim = 64
        z_mean = Dense(latent_dim)(h)
        z_log_sigma = Dense(latent_dim)(h)

        def sampling(args):
            z_mean, z_log_sigma = args
            epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=0.05)
            return z_mean + K.exp(z_log_sigma) * epsilon

        z = Lambda(sampling)([z_mean, z_log_sigma])
        encoder = Model(_input, [z_mean, z_log_sigma, z], name='encoder')

        latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
        h = Dense(units=8*20*1, activation='relu')(latent_inputs)
        h = Reshape(target_shape=(8,20,1))(h)

        h = Conv2D(64, kernel_size=(4, 4), activation="relu", padding="same")(h)
        h = UpSampling2D((2, 2))(h)

        h = Conv2D(32, kernel_size=(6, 6), activation="relu", padding="same")(h)
        h = UpSampling2D((2, 2))(h)

        h = Conv2D(16, kernel_size=(8, 8), activation="sigmoid", padding="same")(h)
        h = UpSampling2D((2, 2))(h)

        h = Conv2D(1, kernel_size=(1, 1), activation="sigmoid", padding="same")(h)
        h = UpSampling2D((2, 2))(h)

        decoder = Model(latent_inputs, h, name='decoder')

        outputs = decoder(encoder(_input)[2])
        model = Model(_input, outputs, name='vae')

        model.compile(loss=loss, optimizer=Adam(learning_rate=lr), metrics=metrics)
        return model
    

class DenseClassifier():
    def __init__(self, encoder: Model) -> None:
        self.encoder = encoder
        
    def create_dense_classifier(self, lr, metrics):
        x_top = self.encoder.output
        h = Flatten()(x_top)
        h = Dense(512, activation = 'relu')(h)
        h = Dropout(0.2)(h)
        h = Dense(256, activation='relu')(h)
        h = BatchNormalization()(h)
        output = Dense(2, activation = 'softmax')(h)

        classifier = Model(self.encoder.inputs, output)

        for layer in self.encoder.layers:
            layer.trainable = False

        classifier.compile(Adam(learning_rate=lr), loss = 'binary_crossentropy', metrics=metrics)
        return classifier
    

def limit_memory():
    import tensorflow as tf

    gpu = tf.config.experimental.list_physical_devices('GPU')[0] # we only have access to 1 GPU via the env variable in ~/.bashrc

    tf.config.experimental.set_virtual_device_configuration(gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=81920 * 0.75)])