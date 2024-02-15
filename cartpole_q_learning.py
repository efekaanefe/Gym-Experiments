import gymnasium as gym
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer


class QLearningAgent: # Quality Learning
    def __init__(self, env, lr = 0.1, n_bins=(6,12)):
        self.Q_table = np.zeros(n_bins + (env.action_space.n,))
        print(f"Q table shape: {self.Q_table.shape}")
        self.lr = lr

    def new_Q_value( reward : float ,  new_state : tuple , discount_factor=1 ) -> float:
        """Temperal diffrence for updating Q-value of state-action pair"""
        future_optimal_value = np.max(self.Q_table[new_state])
        learned_value = reward + discount_factor * future_optimal_value
        return learned_value

    def exploration_rate(n : int, min_rate= 0.1 ):
        """Decaying exploration rate"""
        return max(min_rate, min(1, 1.0 - np.log10((n  + 1) / 25)))


    def choose_action(self, state):
        return np.argmax(self.Q_table[state])


if __name__=="__main__":
    import warnings
    warnings.filterwarnings("ignore")

    env = gym.make("CartPole-v1", render_mode = "human")

    lower_bounds = [ env.observation_space.low[2], -np.radians(50) ]
    upper_bounds = [ env.observation_space.high[2], np.radians(50) ]

    def discretizer( _ , __ , angle, pole_velocity ) :
        """Convert continues state intro a discrete state"""
        est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
        est.fit([lower_bounds, upper_bounds ])
        return tuple(map(int,est.transform([[angle, pole_velocity]])[0]))
    
    n_bins=(6,12)

    agent = QLearningAgent(env)
    num_episodes = 1
    for e in range(num_episodes):
        observation, info = env.reset()
        state = discretizer(*observation) 
        done = False; score = 0
        while not done:
            env.render()
            # action = agent.choose_action(observation)
            action = env.action_space.sample() 
            observation, reward, done, truncated, info = env.step(action)
            state = discretizer(*observation)
            score += reward
            print(state)
            break
        print(f"Episode {e}, score {score}")

    env.close()


