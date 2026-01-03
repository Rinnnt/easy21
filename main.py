from src.environment import Action, Easy21Env, State

def play_manual_game():
    env = Easy21Env(verbose=True)
    print("--- Starting Easy21Env Test---")

    state = env.reset()
    done = False

    print(f"Start State: {state}")

    while not done:
        user_input = input("\nChoose an action [h=hit, s=stick]: ").lower().strip()

        if user_input == 'h':
            action = Action.HIT
        elif user_input == 's':
            action = Action.STICK
        else:
            print("Invalid input!")
            continue

        print(f"Action: {action.name}")
        next_state, reward, done = env.step(action)

        if not done:
            print(f"New State: {next_state} | Reward: {reward}")
        else:
            print(f"---Game Over---")
            print(f"Final State: {next_state}")
            print(f"Final Reward: {reward}")

            if reward == 1:
                print("You Won!")
            elif reward == -1:
                print("You Lose!")
            else:
                print("Result: Draw.")

if __name__ == "__main__":
    play_manual_game()
