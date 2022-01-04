#!/usr/bin/env python
import pickle


class Losses(object):
    def __init__(self, file=None) -> None:
        """Initializes a Losses object. If file is included, will extract losses from losses/{file}.pickle."""
        self.losses = []
        if file:
            self.get_losses(file)

    def add_loss(self, loss: float):
        self.losses.append(loss)

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

    def plot_losses(self):
        cnt = 0
        for item in self.losses:
            print('The data ', cnt, ' is : ', item)
            print('item 1 ', item.numpy())
            cnt += 1