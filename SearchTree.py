#!/usr/bin/env python
import numpy as np
import random
import SearchNode

class SearchTree:
    def __init__(self, root: SearchNode) -> None:
        self.root = root