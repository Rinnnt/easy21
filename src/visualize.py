import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def plot_value_function(agent, title: str = "Value Function") -> None:
    V = np.max(agent.Q, axis=2)
    V = V[1:, 1:]

    dealer_range = np.arange(1, 11)
    player_range = np.arange(1, 22)
    X, Y = np.meshgrid(dealer_range, player_range)

    Z = V.T

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, edgecolor="none")

    ax.set_xlabel("Dealer Showing")
    ax.set_ylabel("Player Sum")
    ax.set_zlabel("Value V(s)")
    ax.set_title(title)

    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    plt.show()
