from src.environment import Action, Reward, State

import numpy as np


class SarsaAgent:
    def __init__(self, lambda_val: float = 0.0) -> None:
        self.k = 100  # epsilon-greedy constant
        self.gamma = 1  # reward discount factor
        self.lambda_val = lambda_val

        self.N = np.zeros((11, 22, 2))
        self.Q = np.zeros((11, 22, 2))
        self.E = np.zeros((11, 22, 2))

    def epsilon(self, state: State) -> float:
        return self.k / (self.k + self.N[state.dealer_sum, state.player_sum].sum())

    def get_action(self, state: State) -> Action:
        # exploratory action with probability epsilon
        if np.random.random() < self.epsilon(state):
            return np.random.choice(list(Action))

        # greedy action otherwise
        else:
            return Action(np.argmax(self.Q[state.dealer_sum, state.player_sum]))

    def update(
        self, s: State, a: Action, r: Reward, ns: State, na: Action | None, done: bool
    ) -> None:
        d = s.dealer_sum
        p = s.player_sum
        nds = ns.dealer_sum
        nps = ns.player_sum

        # calculate the TD target
        q = self.Q[d, p, a]
        nq = self.Q[nds, nps, na] if not done else 0.0

        td_target = r + self.gamma * nq
        td_error = td_target - q

        # increment eligibility trace and visit count
        self.E[d, p, a] += 1
        self.N[d, p, a] += 1

        # calculate step_size
        ss = np.zeros_like(self.N)
        visited_mask = self.N > 0
        ss[visited_mask] = 1 / self.N[visited_mask]

        # update action value function and eligibility trace
        self.Q += ss * td_error * self.E
        self.E *= self.gamma * self.lambda_val

    def reset_traces(self) -> None:
        self.E.fill(0)
