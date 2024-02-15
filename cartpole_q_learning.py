import gymnasium as gym
import numpy as np


class QLearningCartPoleAgent: # Quality Learning
    def __init__(self):
        pass

    def choose_action(self, observation):
        pass

if __name__=="__main__":
    env = gym.make("CartPole-v1", render_mode = "human")

    agent = QLearningCartPoleAgent()
    score = 0
    num_episodes = 100
    for e in range(num_episodes):
        observation, info = env.reset()
        done = False
        while not done:
            env.render()
            # action = agent.choose_action(observation)
            action = env.action_space.sample() 
            observation, reward, done, truncated, info = env.step(action)
            score += reward
        print(f"Episode {e}, score {score}")

    env.close()


