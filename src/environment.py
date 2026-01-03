from enum import IntEnum
import numpy as np
from typing import NamedTuple, Optional, Tuple, TypeAlias

class State(NamedTuple):
    dealer_sum: int
    player_sum: int

class Action(IntEnum):
    HIT = 0
    STICK = 1

Reward: TypeAlias = float
Done: TypeAlias = bool

class Easy21Env():

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

        self.lowest_value: int = 1
        self.highest_value: int = 21

        self.lowest_card: int = 1
        self.highest_card: int = 10
        self.red_prob: float = 1 / 3

        self.dealer_stick_value: int = 17

        self.lose_reward: int = -1
        self.draw_reward: int = 0
        self.win_reward: int = 1

        self.state: Optional[State] = None

    def drawCard(self) -> int:
        return np.random.randint(self.lowest_card, self.highest_card + 1)

    def sampleCardValue(self) -> int:
        return (-1 if np.random.random() < self.red_prob else 1) * self.drawCard()

    def bust(self, s: int) -> bool:
        return s < self.lowest_value or s > self.highest_value

    def reset(self) -> State:
        self.state = State(self.drawCard(), self.drawCard())
        return self.state

    def step(self, action: Action) -> Tuple[State, Reward, Done]:
        if self.state is None:
            raise RuntimeError("Call env.reset() before step()")

        match action:
            case Action.HIT:
                value = self.sampleCardValue()
                self.state = self.state._replace(player_sum = self.state.player_sum + value)

                if self.verbose:
                    print(f"[Environment] Player Hit Value: {value}")

                if self.bust(self.state.player_sum):
                    return (self.state, self.lose_reward, True)
                return (self.state, 0, False)

            case Action.STICK:
                while self.state.dealer_sum < self.dealer_stick_value:
                    value = self.sampleCardValue()
                    self.state = self.state._replace(dealer_sum = self.state.dealer_sum + value)

                    if self.verbose:
                        print(f"[Environment] Dealer Hit Value: {value}")

                    if self.bust(self.state.dealer_sum):
                        return (self.state, self.win_reward, True)

                if self.state.player_sum > self.state.dealer_sum:
                    reward = self.win_reward
                elif self.state.player_sum < self.state.dealer_sum:
                    reward = self.lose_reward
                else:
                    reward = self.draw_reward

                return (self.state, reward, True)

