import gym

# Create the FrozenLake environment with human-rendered mode
env = gym.make("FrozenLake-v1", render_mode="human")
env.reset()
env.render()

# Observation space
print("Observation space:", env.observation_space)

# Action space
print("Action space:", env.action_space)

# Generate a random action
randomAction = env.action_space.sample()
print("Random action:", randomAction)

# Step the environment with the random action
returnValue = env.step(randomAction)
print("Return value:", returnValue)

# Format of returnValue is (observation, reward, terminated, truncate, info)

# Accessing the transition probability for state 1 and action 2
transition_prob = env.P[1][2]
print("Transition probability for state 1 and action 2:", transition_prob)
