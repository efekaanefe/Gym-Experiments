import gymnasium as gym
import numpy as np
import torch

if __name__=="__main__":
    model_name = "CartPole-v1_1276"
    model = torch.load(f"models\\{model_name}")
    env = gym.make("CartPole-v1", render_mode = "human")
    num_episodes = 2
    
    for e in range(num_episodes):
        state, _ = env.reset()
        done = False; score = 0
        
        while not done:
            action = model(obs)
            new_state, reward, done, _, _ = env.step(action)
            new_discrete_state = discretizer(new_state)
            state = new_state
            score += reward
            env.render()
            if score % 100 == 0:
                print(f"Episode {e}, score {score}")

    env.close()