import gymnasium as gym
import numpy as np


if __name__=="__main__":
    # https://www.youtube.com/watch?v=JNKvJEzuNsc
    # https://github.com/udacity/reinforcement-learning/blob/master/notebooks/Discretization.ipynb
    # https://www.youtube.com/watch?v=yMk_XtIEzH8&list=PLQVvvaa0QuDezJFIOU5wDdfy4e9vdnx-7

    env = gym.make("CartPole-v1", render_mode = "human")

    # only working with obs[0] and obs[2], since others can have value of +-inf, not good for q-learning
    max_obs_values = env.observation_space.high
    min_obs_values = env.observation_space.low
    max_obs_values = np.array([max_obs_values[0], np.rad2deg(max_obs_values[2])])
    min_obs_values = np.array([min_obs_values[0], np.rad2deg(min_obs_values[2])])

    ## DISCRETIZING
    DISCRETE_OBS_SPACE_SIZE = [20]* len(max_obs_values) # these are the what the max values will corresponds to
    discrete_obs_space_step_size = (max_obs_values - min_obs_values) / DISCRETE_OBS_SPACE_SIZE

    def discretizer(obs):
        obs = np.array([obs[0], obs[2]])
        discrete_obs = (obs - min_obs_values)/discrete_obs_space_step_size
        return discrete_obs.astype(np.int16)
    
    ## CONSTANTS, these may be adjusted based on episode number
    LEARNING_RATE = 0.1
    DISCOUNT = 0.95
    EXPLORATION_RATE = 0.2

    ## Q-table
    q_table = np.random.uniform(min=-2,max=2, size=(DISCRETE_OBS_SPACE_SIZE + [env.action_space.n]))

    num_episodes = 1
    for e in range(num_episodes):
        observation, info = env.reset()
        current_obs = discretizer(observation) 
        done = False; score = 0
        while not done:
            action = np.argmax(q_table[current_obs[0]][current_obs[1]])
            if EXPLORATION_RATE > np.random.random():
                action = env.action_space.sample() 

            observation, reward, done, _, _ = env.step(action)
            new_state = discretizer(observation)

            ## TRAINING
            # learnt_value = agent.new_Q_value(reward , new_state )
            # old_value = agent.Q_table[current_state][action]
            # agent.Q_table[current_state][action] = (1-agent.lr)*old_value + agent.lr*learnt_value


            current_state = new_state
            score += reward
            env.render()

            # print(current_state, observation)
        print(f"Episode {e}, score {score}")

    env.close()


