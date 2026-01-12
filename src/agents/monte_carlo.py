from src.environment import Action, Reward, State

import numpy as np
from typing import List, Tuple


class MonteCarloAgent:
    def __init__(self) -> None:
        self.k = 100  # epsilon-greedy constant
        self.gamma = 1  # reward discount factor

        self.N = np.zeros((11, 22, 2))
        self.Q = np.zeros((11, 22, 2))

    def epsilon(self, state: State) -> float:
        return self.k / (self.k + self.N[state.dealer_sum, state.player_sum].sum())

    def get_action(self, state: State) -> Action:
        # exploratory action with probability epsilon
        if np.random.random() < self.epsilon(state):
            return np.random.choice(list(Action))

        # greedy action otherwise
        else:
            return Action(np.argmax(self.Q[state.dealer_sum, state.player_sum]))

    def step_size(self, state: State, action: Action) -> float:
        return 1 / self.N[state.dealer_sum, state.player_sum, action]

    def update(self, episode: List[Tuple[State, Action, Reward]]) -> None:
        # keep the index of the first occurence of
        # (state, action) pairs for first-visit Monte Carlo
        first_occurences = {}
        for i, (state, action, _) in enumerate(episode):
            sa_pair = (state, action)
            if sa_pair not in first_occurences:
                first_occurences[sa_pair] = i

        G = 0
        for i in range(len(episode) - 1, -1, -1):
            s, a, r = episode[i]

            G = self.gamma * G + r

            if first_occurences[(s, a)] == i:
                d = s.dealer_sum
                p = s.player_sum

                self.N[d, p, a] += 1
                self.Q[d, p, a] += self.step_size(s, a) * (G - self.Q[d, p, a])
