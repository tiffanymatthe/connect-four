#!/usr/bin/env python

from keras.models import Model
from keras.layers.normalization import BatchNormalization
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

    def inference(self, image):
        return (-1, {})  # Value, Policy

    def get_weights(self):
        # Returns the weights of this network.
        return []

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
        # Returns a tensor to put in the model output
        # TODO: implement
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
        # Returns a tensor to put in the model output
        # TODO: implement
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
        inputs = Inputs(shape=(height, width, 3))
        value_branch = Network.get_value_branch(inputs)
        policy_branch = Network.get_policy_branch(inputs)

        model = Model(input=inputs, output=[value_branch, policy_branch])

        return model

    @staticmethod
    def get_model():
        # Returns TF model with two output branches. should already be compiled.
        # TODO: implement
        width = 7
        height = 6
        model = Network.compile_model(width, height)

        return model

    def print_model_summary(self):
        model = self.get_model()
        print(model.summary())


if __name__ == "__main__":
    network = Network()
    network.print_model_summary()
