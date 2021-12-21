import numpy as np

class board:
    def __init__(self) -> None:
        self.num_col = 7
        self.col_height = 6
        # 1 is white, -1 is black, 0 is empty
        self.current_state = np.zeros((self.col_height, self.num_col))

    def see_board():
        """
        Display board. 'o' represents white, 'x' represents black, '-' represents an empty space.
        """