import numpy as np
import matplotlib.pyplot as plt

import collections
import gym
import cv2

def plot_learning_curve(x, scores, figure_file):
    """
    """
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)


class RepeatActionAndMaxFrame(gym.Wrapper):
    """
    """
    def __init__(self, env=None, repeat=4) -> None:
        super().__init__(env)
        self.repeat = repeat
        self.shape = env.observation_space.low.shape
        self.frame_buffer = np.zeros_like((2, self.shape))

    def step(self, action):
        """
        """
        t_reward = 0.0
        done = False
        for i in range(self.repeat):
            obs, reward, done, info = self.env.step(action)
            t_reward += reward
            idx = i%2
            self.frame_buffer[idx] = obs
            if done:
                break

        max_frame = np.maximum(self.frame_buffer[0], self.frame_buffer[1])
        return max_frame, t_reward, done, info
    
    def reset(self):
        """
        """
        obs = self.env.reset()
        self.frame_buffer = np.zeros_like((2, self.shape))
        self.frame_buffer[0] = obs
        return obs

class PreprocessFrame(gym.ObservationWrapper):
    """
    """
    def __init__(self, shape, env=None) -> None:
        super().__init__(env)
        self.shape = (shape[2], shape[0], shape[1])
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=self.shape, dtype=np.float32)
    
    def observation(self, obs):
        new_frame = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized_screen = cv2.resize(new_frame, self.shape[1:], interpolation=cv2.INTER_AREA)
        new_obs = np.array(resized_screen, dtype=np.uint8).reshape(self.shape)
        new_obs = new_obs/255.0
        return new_obs
    
class StackFrames(gym.ObservationWrapper):
    """
    """
    def __init__(self, env, repeat) -> None:
        super().__init__(env)
        self.observation_space = gym.spaces.Box(env.observation_space.low.repeat(repeat, acis=0),
                                                env.observation_space.high.repeat(repeat, acis=0),
                                                dtype = np.float32)
        self.stack = collections.deque(maxlen=repeat)

    def reset(self):
        self.stack.clear()
        observation = self.env.reset()
        for _ in range(self.stack.maxlen):
            self.stack.append(observation)
        
        return np.array(self.stack).reshape(self.observation_space.low.shape)

    def observation(self, observation):
        """
        """
        self.stack.append(observation)
        return np.array(self.stack).reshape(self.observation_space.low.shape)


def make_env(env_name, shape=(84,84,1), repeat=4):
    """
    """
    env = env.make(env_name)
    env = RepeatActionAndMaxFrame(env, repeat)
    env = PreprocessFrame(shape, env)
    env = StackFrames(env, repeat)

    return env

