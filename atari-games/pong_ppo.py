import gymnasium as gym

env = gym.make("ALE/Pong-v5", render_mode = "human")


observation = env.reset()
num_steps = 1000
total_reward = 0

for _ in range(num_steps):
    env.render()
    action = env.action_space.sample()
    observation, reward, done, _, _ = env.step(action)
    total_reward += reward
    if done:
        print("Episode finished")
        break
env.close()
print("Total reward:", total_reward)