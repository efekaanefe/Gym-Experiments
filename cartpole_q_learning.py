import gymnasium as gym
import numpy as np


if __name__=="__main__":
    # https://www.youtube.com/watch?v=JNKvJEzuNsc
    # https://github.com/udacity/reinforcement-learning/blob/master/notebooks/Discretization.ipynb
    # https://www.youtube.com/watch?v=yMk_XtIEzH8&list=PLQVvvaa0QuDezJFIOU5wDdfy4e9vdnx-7

    # env = gym.make("CartPole-v1", render_mode = "human")
    env = gym.make("MountainCar-v0", render_mode = "human")
    
    # only will look to 0th and 2nd observations
    DISCRETE_OBS_SPACE_SIZE = [20] * len(env.observation_space.high)
    discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/DISCRETE_OBS_SPACE_SIZE
    print(discrete_os_win_size)
    print(DISCRETE_OBS_SPACE_SIZE+ [env.action_space.n])
    # every combination of observations and the action spaces
    q_table = np.random.uniform(low=-2, high=2, size=(DISCRETE_OBS_SPACE_SIZE + [env.action_space.n]))

    
    def discretizer(position , _ , angle, __): # only two observations
        return 0

    num_episodes = 1
    for e in range(num_episodes):
        observation, info = env.reset()
        current_state = discretizer(*observation) 
        done = False; score = 0
        while not done:
            # action = choose_action(current_state)
            # if exploration_rate(e) > np.random.random():
            action = env.action_space.sample() 

            observation, reward, done, _, _ = env.step(action)
            new_state = discretizer(*observation)

            # update
            # learnt_value = agent.new_Q_value(reward , new_state )
            # old_value = agent.Q_table[current_state][action]
            # agent.Q_table[current_state][action] = (1-agent.lr)*old_value + agent.lr*learnt_value


            current_state = new_state
            score += reward
            env.render()

            # print(current_state, observation)
        print(f"Episode {e}, score {score}")

    env.close()


