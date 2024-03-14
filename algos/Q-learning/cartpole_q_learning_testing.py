import gymnasium as gym
import numpy as np
import pickle

if __name__=="__main__":
    # https://www.youtube.com/watch?v=JNKvJEzuNsc
    # https://github.com/udacity/reinforcement-learning/blob/master/notebooks/Discretization.ipynb
    # https://www.youtube.com/watch?v=yMk_XtIEzH8&list=PLQVvvaa0QuDezJFIOU5wDdfy4e9vdnx-7

    env = gym.make("CartPole-v1", render_mode = "human")

    max_obs_values = np.array([2.4, 4, 0.2099 , 4])
    min_obs_values = np.array([-2.4, -4, -0.2099, -4])
    DISCRETE_OBS_SPACE_SIZE = [10]* len(max_obs_values) # these are the what the max values will corresponds to
    discrete_obs_space_step_size = (max_obs_values - min_obs_values) / DISCRETE_OBS_SPACE_SIZE

    def discretizer(obs):
        # obs = np.array([obs[0], obs[2]])
        discrete_obs = (obs - min_obs_values-0.1)/discrete_obs_space_step_size
        return tuple(discrete_obs.astype(np.int16)) # tuple to make indexing easier

    with open("q-table.pck", "rb") as file:
        q_table = pickle.load(file)
  
    num_episodes = 2

    for e in range(num_episodes):
        state, _ = env.reset()
        done = False; score = 0
        while not done:
            discrete_state = discretizer(state) 
            action = np.argmax(q_table[discrete_state])
            new_state, reward, done, _, _ = env.step(action)
            new_discrete_state = discretizer(new_state)
            state = new_state
            score += reward
            env.render()
            if score % 100 == 0:
                print(f"Episode {e}, score {score}")

    env.close()
