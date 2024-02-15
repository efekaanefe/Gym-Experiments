import gymnasium as gym
import numpy as np


env = gym.make("CartPole-v1", render_mode = "human")

observation, info = env.reset()

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

class PIDCartPoleAgent:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0
        self.integral = 0

    def choose_action(self, observation):
        position, velocity, angle, angular_velocity = observation
        target_position = np.array([0,0,0,0])
        error = target_position - position

        self.integral += error

        derivative = error - self.prev_error
        self.prev_error = error

        pid = self.kp * error + self.ki * self.integral + self.kd * derivative

        # if pid < 0:
        #     action = 0  # Move left
        # else:
        #     action = 1  # Move right
        action = sigmoid(pid)
        action = np.round(action).astype(np.int32)
        return action

P, I, D = [1/150, 1/950, 0.1, 0.01], [0.0005, 0.001, 0.01, 0.0001], [0.2, 0.0001, 0.5, 0.005]
agent = PIDCartPoleAgent(P, I, D)

observation, info = env.reset()
done = False
while not done:
    env.render()
    action = agent.choose_action(observation)
    observation, reward, terminated, done , info = env.step(action)
env.close()


