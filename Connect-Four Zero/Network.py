#!/usr/bin/env python

from keras.models import Model
from keras.layers import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Lambda
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
import tensorflow as tf


class Network(object):
    def __init__(self) -> None:
        self.model = self.__get_model()

    def inference(self, image):
        """Returns a tuple with value and policy: (-1, {})"""
        return self.model.predict(image)

    def get_weights(self):
        # Returns the weights of this network as a list.
        # https://github.com/google/prettytensor/issues/6
        return [v for v in tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES) if v.name.endswith('weights:0')]

    @staticmethod
    def get_common_layers(inputs):
        # Returns a tensor representing the common layers
        x = Conv2D(32, (3, 3), padding="same")(inputs)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=-1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)

        return x

    @staticmethod
    def get_policy_branch(inputs):
        """
        This is assuming that inputs are the image arrays with 6x7 dimension
        and 3 feature planes. We know that the size of the output should be 7
        from before hand therefore, that is our output size.
        """
        x = Network.get_common_layers(inputs)
        x = Flatten()(x)
        x = Dense(128)(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)

        x = Dense(7)(x)  # size of the output
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
        x = Flatten()(x)
        x = Dense(128)(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)

        x = Dense(1)(x)  # size of the output
        x = Activation("tanh", name="value_output")(x)

        return x

    @staticmethod
    def compile_model(width, height):
        # 3 because we will 3 channels (red tokens, yellow tokens, board state)
        inputs = Input(shape=(height, width, 3))
        value_branch = Network.get_value_branch(inputs)
        policy_branch = Network.get_policy_branch(inputs)

        model = Model(inputs=inputs, outputs=[value_branch, policy_branch])

        return model

    def __get_model(self):
        # Returns TF model with two output branches.
        width = 7
        height = 6
        return Network.compile_model(width, height)

    def print_model_summary(self):
        print(self.model.summary())
