#!/usr/bin/env python

class C4Game():
    def __init__(self, history=None) -> None:
        self.history = history or [] # check if this works
        self.child_visits = []
        self.num_actions = 7

    def terminal(self):
        pass

    def terminal_value(self, to_play):
        pass

    def legal_actions(self):
        return []

    def clone(self):
        return C4Game(list(self.history))

    def apply(self, action):
        self.history.append(action)

    def store_search_statistics(self, root):
        # https://ai.stackexchange.com/questions/25451/how-does-alphazeros-mcts-work-when-starting-from-the-root-node
        sum_visits = sum(child.visit_count for child in root.children.itervalues())
        self.child_visits.append([
            root.children[a].visit_count / sum_visits if a in root.children else 0
            for a in range(self.num_actions)
        ])

    def make_image(self, state_index: int):
        return []

    def make_target(self, state_index: int):
        return (self.terminal_value(state_index % 2),
                self.child_visits[state_index])

    def to_play(self):
        return len(self.history) % 2