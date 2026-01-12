from src.agents.monte_carlo import MonteCarloAgent
from src.agents.sarsa_agent import SarsaAgent
from src.environment import Action, Easy21Env
from src.visualize import plot_value_function

from tqdm import tqdm


def play_manual_game() -> None:
    print("--- Starting Easy21Env Test---")

    env = Easy21Env(verbose=True)
    state = env.reset()
    done = False

    print(f"Start State: {state}")
    while not done:
        user_input = input("\nChoose an action [h=hit, s=stick]: ").lower().strip()

        if user_input == "h":
            action = Action.HIT
        elif user_input == "s":
            action = Action.STICK
        else:
            print("Invalid input!")
            continue

        print(f"Action: {action.name}")
        next_state, reward, done = env.step(action)

        if not done:
            print(f"New State: {next_state} | Reward: {reward}")
        else:
            print("---Game Over---")
            print(f"Final State: {next_state}")
            print(f"Final Reward: {reward}")


def train_monte_carlo(episodes: int) -> None:
    agent = MonteCarloAgent()
    env = Easy21Env()

    print(f"Training {type(agent).__name__} for {episodes} episodes...")

    for _ in tqdm(range(episodes), desc="Training Progress"):
        episode = []
        state = env.reset()
        done = False

        while not done:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            episode.append((state, action, reward))
            state = next_state

        agent.update(episode)

    print("Plotting value function...")
    plot_value_function(agent, title=f"Monte Carlo Control ({episodes} episodes)")


def train_sarsa(lambda_val: float, episodes: int) -> None:
    agent = SarsaAgent(lambda_val)
    env = Easy21Env()

    print(f"Training {type(agent).__name__} for {episodes} episodes...")

    for _ in tqdm(range(episodes), desc="Training Progress"):
        agent.reset_traces()

        state = env.reset()
        action = agent.get_action(state)

        while True:
            next_state, reward, done = env.step(action)

            if done:
                agent.update(state, action, reward, next_state, None, done=True)
                break

            next_action = agent.get_action(next_state) if not done else None

            agent.update(state, action, reward, next_state, next_action, done)

            state = next_state
            action = next_action

    print("Plotting value function...")
    plot_value_function(agent, title=f"Sarsa Control ({episodes} episodes)")


if __name__ == "__main__":
    train_sarsa(0.5, 1_000_000)
