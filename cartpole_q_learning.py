import gymnasium as gym
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer


class QLearningAgent: # Quality Learning
    def __init__(self, env, lr = 0.1, exploration_rate = 0.1, n_bins = (6,12)):
        self.Q_table = np.zeros(n_bins + (env.action_space.n,))
        print(f"Q table shape: {self.Q_table.shape}")
        self.lr = lr
        self.exploration_rate = exploration_rate

    def new_Q_value(self, reward : float ,  new_state : tuple , discount_factor=1 ) -> float:
        """Temperal diffrence for updating Q-value of state-action pair"""
        future_optimal_value = np.max(self.Q_table[new_state])
        learned_value = reward + discount_factor * future_optimal_value
        return learned_value

    def choose_action(self, state):
        return np.argmax(self.Q_table[state])

if __name__=="__main__":
    # https://www.youtube.com/watch?v=JNKvJEzuNsc
    # https://github.com/udacity/reinforcement-learning/blob/master/notebooks/Discretization.ipynb
    env = gym.make("CartPole-v1", render_mode = "human")

    lower_bounds = [ env.observation_space.low[2], -np.radians(50) ]
    upper_bounds = [ env.observation_space.high[2], np.radians(50) ]

    def discretizer( _ , __ , angle, pole_velocity ) :
        """Convert continues state intro a discrete state"""
        est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform', subsample=200_000)
        est.fit([lower_bounds, upper_bounds ])
        return np.array(tuple(map(int,est.transform([[angle, pole_velocity]])[0])))
    
    n_bins = (6,2) # shape: n_features

    agent = QLearningAgent(env, n_bins = n_bins)
    num_episodes = 1000
    for e in range(num_episodes):
        observation, info = env.reset()
        current_state = discretizer(*observation) 
        done = False; score = 0
        while not done:
            action = agent.choose_action(current_state)
            # if agent.exploration_rate > np.random.random():
            #     action = env.action_space.sample() 

            observation, reward, done, _, _ = env.step(action)
            new_state = discretizer(*observation)

            # update
            learnt_value = agent.new_Q_value(reward , new_state )
            old_value = agent.Q_table[current_state][action]
            agent.Q_table[current_state][action] = (1-agent.lr)*old_value + agent.lr*learnt_value


            current_state = new_state
            score += reward
            env.render()

            # print(current_state, observation)
        print(f"Episode {e}, score {score}")

    env.close()


