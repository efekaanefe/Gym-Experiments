import gymnasium as gym
import numpy as np


if __name__=="__main__":
    # https://www.youtube.com/watch?v=JNKvJEzuNsc
    # https://github.com/udacity/reinforcement-learning/blob/master/notebooks/Discretization.ipynb
    # https://www.youtube.com/watch?v=yMk_XtIEzH8&list=PLQVvvaa0QuDezJFIOU5wDdfy4e9vdnx-7

    # env = gym.make("CartPole-v1", render_mode = "human")
    env = gym.make("CartPole-v1")

    # only working with obs[0] and obs[2], since others can have value of +-inf, not good for q-learning
    max_obs_values = env.observation_space.high
    min_obs_values = env.observation_space.low
    max_obs_values = np.array([max_obs_values[0], np.rad2deg(max_obs_values[2])])/2
    min_obs_values = np.array([min_obs_values[0], np.rad2deg(min_obs_values[2])])/2

    ## DISCRETIZING
    DISCRETE_OBS_SPACE_SIZE = [20]* len(max_obs_values) # these are the what the max values will corresponds to
    discrete_obs_space_step_size = (max_obs_values - min_obs_values) / DISCRETE_OBS_SPACE_SIZE

    def discretizer(obs):
        obs = np.array([obs[0], obs[1]])
        discrete_obs = (obs - min_obs_values)/discrete_obs_space_step_size
        return tuple(discrete_obs.astype(np.int16)) # tuple to make indexing easier
    
    ## CONSTANTS, these may be adjusted based on episode number
    LEARNING_RATE = 0.1
    DISCOUNT = 0.95
    EXPLORATION_RATE = 1
    EXPLORATION_DECAY_RATE = 0.000001

    ## Q-table
    q_table = np.zeros(DISCRETE_OBS_SPACE_SIZE + [env.action_space.n])
    num_episodes = 100000
    scores = []
    # e = 0
    # while True:
    for e in range(num_episodes):
        observation, info = env.reset()
        current_obs = discretizer(observation) 
        done = False; score = 0
        while not done:
            action = np.argmax(q_table[current_obs])
            if EXPLORATION_RATE > np.random.random():
                 action = env.action_space.sample() 

            observation, reward, done, _, _ = env.step(action)
            new_obs = discretizer(observation)

            ## TRAINING
            if not done:
                max_future_q = np.max(q_table[new_obs,:])
                print(max_future_q)
                current_q = q_table[current_obs + (action,)]
                new_q = (1-LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
                q_table[current_obs + (action,)] = new_q
                # print(new_q, current_q) # new_q isn't changing

            current_obs = new_obs
            score += reward
            EXPLORATION_RATE -= np.max(EXPLORATION_RATE - EXPLORATION_DECAY_RATE, 0)
            # env.render()
            # print(current_obs, observation)
        scores.append(score)
        # print(f"Episode {e}, score {score}")
        if score > 1500:
            break
        # e += 1

    env.close()

    import matplotlib.pyplot as plt
    plt.plot(scores)
    plt.show()
