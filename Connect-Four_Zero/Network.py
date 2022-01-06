#!/usr/bin/env python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from Residual_CNN import Residual_CNN
from C4Config import C4Config
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class Network(object):
    def __init__(self, config: C4Config, model_name=None, model=None) -> None:
        self.cnn = None
        if model:
            self.cnn = Residual_CNN(config, model=model)
        elif model_name:
            print("starting residual cnn")
            self.cnn = Residual_CNN(config, model_name=model_name)
        else:
            self.cnn = Residual_CNN(config)
        self.width = 7
        self.height = 6

    def inference(self, image):
        """
        image is a numpy array of size 6 x 7 x 2

        Returns a tuple with value and policy: (scalar, array of length 7)
        https://github.com/tensorflow/tensorflow/issues/40261#issuecomment-647191650
        """
        tensor = tf.convert_to_tensor(image[None], dtype=tf.int32)
        value, policy = self.cnn.model(tensor, training=False)
        print("done prediction")
        return value[0, 0], policy[0]

    def get_weights(self):
        # Returns the weights of this network as a list.
        # https://github.com/google/prettytensor/issues/6
        return self.cnn.model.trainable_weights

    def clone_network(self, config):
        """Clones the network with same weights. Good for training."""
        new = Network(config)
        new.cnn.model.set_weights(self.cnn.model.get_weights())
        return new

    def print_model_summary(self):
        print(self.cnn.model.summary())
