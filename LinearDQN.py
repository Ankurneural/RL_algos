import numpy as np
import gym
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T

from utils import plot_learning_curve

class LinearDQN(nn.Module):
    """
    """
    def __init__(self, lr, n_actions, input_dims) -> None:
        """
        """
        super().__init__()
        self.lr = lr
        self.fc1 = nn.Linear(*input_dims, 128)
        self.fc2 = nn.Linear(128, n_actions)

        self.optim = optim.Adam(self.parameters(), lr=self.lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available else 'cpu')
        self.to(self.device)

    def forward(self, states):
        """
        """
        layer1 = F.relu(self.fc1(states))
        return self.fc2(layer1)


class Agent:
    """
    """
    def __init__(self, input_dims, n_actions, lr=0.0001, gamma=0.99,
                epsilon=1.0, eps_dec = 1e-05, eps_min=0.01) -> None:
        """
        """
        self.input_dims= input_dims
        self.n_actions= n_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_dec = eps_dec
        self.eps_min = eps_min
        self.action_space = [i for i in range(self.n_actions)]

        self.Q = LinearDQN(lr=self.lr, n_actions=self.n_actions, input_dims=self.input_dims)
    
    def choose_action(self, obser):
        """
        """
        if np.random.random() > self.epsilon:
            state = T.tensor(obser, dtype=T.float).to(self.Q.device)
            actions = self.Q.forward(state)
            action = actions.cpu().data.numpy().argmax()
        else:
            action = np.random.choice(self.action_space)
        
        return action
    
    def decrement_epsilon(self):
        """
        """
        if self.epsilon > self.eps_min:
            self.epsilon -= self.eps_dec
        else:
            self.eps_min

    def learn(self, state, action, reward, state_):
        """
        """
        self.Q.optim.zero_grad()
        states = T.tensor(state, dtype=T.float).to(self.Q.device)
        actions = T.tensor(action, dtype=T.float).to(self.Q.device)
        rewards = T.tensor(reward, dtype=T.float).to(self.Q.device)
        states_ = T.tensor(state_, dtype=T.float).to(self.Q.device)

        # print(f"actions: {actions.long()}")
        q_pred = self.Q.forward(states)[actions.long()]

        q_next = self.Q.forward(states_).max()
        q_target = rewards + self.gamma*q_next

        loss = self.Q.loss(q_target, q_pred).to(self.Q.device)
        loss.backward()
        self.Q.optim.step()
        self.decrement_epsilon()


if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    n_games = 10000
    scores, eps_his = [], []
    agent = Agent(input_dims=env.observation_space.shape, n_actions=env.action_space.n)

    for i in range(n_games):
        score = 0
        done = False
        obs = env.reset()

        while not done:
            action = agent.choose_action(obs)
            obs_, reward, done, info = env.step(action)
            score += reward
            agent.learn(obs, action, reward, obs_)
            obs = obs_
        
        scores.append(score)
        eps_his.append(agent.epsilon)

        if i%100==0:
            avg_score = np.mean(scores[-100:])
            print(f"Episode {i}: score: {avg_score} epsilon {agent.epsilon}")
        
    
    filename = 'plots/cartpole_dqn.png'
    x = [i+1 for i in range(n_games)]
    plot_learning_curve(x, scores, filename)



            

