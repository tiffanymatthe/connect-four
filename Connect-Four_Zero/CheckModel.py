#!/usr/bin/env python
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(dir_path, os.pardir)))
import numpy as np
from DataGenerator import DataGenerator
from NetworkTraining import NetworkTraining
from C4Config import C4Config
from Node import Node
import tensorflow as tf
from Network import Network
from SelfPlay import SelfPlay
from C4Game import C4Game
from Losses import Losses
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

if __name__ == "__main__":
    version = '32'
    losses = Losses()
    losses.get_losses(version)
    losses.plot_losses()