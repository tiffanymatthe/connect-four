#!/usr/bin/env python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras.models import Model
from keras.layers import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class Network(object):
    def __init__(self, model_name=None) -> None:
        # TODO: initialize model with uniform policy and value 0.5
        self.model = self.__load_model(model_name) if model_name else self.__get_model()
        self.width = 7
        self.height = 6
        self.losses = {
            "value_output": "mse", 
            "probability_output":"categorical_crossentropy"
        }

    def inference(self, image):
        """
        image is a numpy array of size 6 x 7 x 2

        Returns a tuple with value and policy: (scalar, array of length 7)
        https://github.com/tensorflow/tensorflow/issues/40261#issuecomment-647191650
        """
        tensor = tf.convert_to_tensor(image[None], dtype=tf.int32)
        value, policy = self.model(tensor, training=False)
        return value[0,0], policy[0]

    def get_weights(self):
        # Returns the weights of this network as a list.
        # https://github.com/google/prettytensor/issues/6
        return self.model.trainable_weights

    def clone_network(self):
        """Clones the network with same weights. Only for prediction, so not compiled."""
        new = Network()
        new.model = tf.keras.models.clone_model(self.model)
        new.model.set_weights(self.model.get_weights())

        return new

    @staticmethod
    def get_common_layers(inputs):
        # Returns a tensor representing the common layers
        x = Conv2D(32, (3, 3), padding="same")(inputs)
        x = BatchNormalization(axis=-1)(x)
        x = Activation("relu")(x)
        x = Flatten()(x)
        x = Conv2D(32, (3, 3), padding="same")(inputs)
        x = BatchNormalization(axis=-1)(x)
        x = Activation("relu")(x)
        x = Flatten()(x)
        x = Conv2D(32, (3, 3), padding="same")(inputs)
        x = BatchNormalization(axis=-1)(x)
        x = Activation("relu")(x)
        x = Flatten()(x)

        return x

    @staticmethod
    def get_policy_branch(inputs):
        """
        This is assuming that inputs are the image arrays with 6x7 dimension
        and 3 feature planes. We know that the size of the output should be 7
        from before hand therefore, that is our output size.
        """
        x = Network.get_common_layers(inputs)
        x = Conv2D(2, (1, 1), padding="same")(inputs)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Flatten()(x)

        x = Dense(7)(x)
        x = Activation("softmax", name="probability_output")(x)

        return x

    @staticmethod
    def get_value_branch(inputs):
        """
        In this function we know that the result of this branch should be a 
        0 or a 1 representing whether the current player is going to win or not. 
        Hence, the size of the output is just 1. 

        Not sure about how to call the binary step function here. 
        """
        x = Network.get_common_layers(inputs)
        x = Conv2D(1, (1, 1), padding="same")(inputs)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Dense(32)(x)
        x = Activation("relu")(x)
        x = Flatten()(x)

        x = Dense(1)(x)  # size of the output
        x = Activation("tanh", name="value_output")(x)

        return x

    @staticmethod
    def compile_model(width, height):
        # 2 because we will have 2 channels (player_to_move, opponent)
        inputs = Input(shape=(height, width, 2))
        value_branch = Network.get_value_branch(inputs)
        policy_branch = Network.get_policy_branch(inputs)

        model = Model(inputs=inputs, outputs=[value_branch, policy_branch])
        losses = {
            "value_output": "mse", 
            "probability_output":"categorical_crossentropy"
        }
        model.compile(optimizer='sgd', loss=losses) # not even sure if this is needed
        return model

    def __get_model(self):
        # Returns TF model with two output branches.
        width = 7
        height = 6
        return Network.compile_model(width, height)

    def __load_model(self, model_name):
        model = tf.keras.models.load_model(f'models/{model_name}')
        losses = {
            "value_output": "mse", 
            "probability_output":"categorical_crossentropy"
        }
        model.compile(optimizer='sgd', loss=losses)
        print("Loaded model.")
        return model

    def print_model_summary(self):
        print(self.model.summary())
