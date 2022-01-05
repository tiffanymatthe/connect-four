#!/usr/bin/env python
import pickle

class Losses(object):
    def __init__(self) -> None:
        self.losses = {
            'loss': [],
            'value_head_loss': [],
            'policy_head_loss': []
        }

    def add_loss(self, overall, value, policy):
        self.losses['loss'].append(overall)
        self.losses['value_head_loss'].append(value)
        self.losses['policy_head_loss'].append(policy)

    def save(self, file: str):
        """file: name of pickle file (not including extension) to save to. Stored under losses folder."""
        with open(f"losses/{file}.pickle", "wb") as file_to_write:
            pickle.dump(self.losses, file_to_write)
            print("successful dump")

    def get_losses(self, file: str):
        """
        file: name of pickle file (not including extension) to read from. Stored under losses folder.
        CAUTION: will overwrite any losses of current object.
        """
        with open(f"losses/{file}.pickle", "rb") as file_to_read:
            self.losses = pickle.load(file_to_read)
            print("successful read")