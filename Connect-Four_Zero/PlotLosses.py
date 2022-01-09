#!/usr/bin/env python
from Losses import Losses

if __name__ == "__main__":
    version = '33'
    losses = Losses()
    losses.get_losses(version)
    losses.plot_losses()