#!/usr/bin/env python

class Network(object):

  def inference(self, image):
    return (-1, {})  # Value, Policy

  def get_weights(self):
    # Returns the weights of this network.
    return []