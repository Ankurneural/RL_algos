import os
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp

class Actor_Critic(keras.Model):
    """
    """
    def __init__(self, n_actions, fc1_dims=1024, fc2_dims=512,
                     name='actor_critic', chkpt_dir = 'tmp/actor_critic') -> None:
        super().__init__()
        self.n_actions = n_actions
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.model_name = name
        self.chkpt_dir = chkpt_dir
        self.chkpt_dir_file = os.path.join(self.chkpt_dir, name + '_ac')

        self.layer1 = Dense(self.fc1_dims, activation='relu')
        self.layer2 = Dense(self.fc2_dims, activation='relu')

        self.v = Dense(1, activation=None)
        self.pi = Dense(n_actions, activation='softmax')

    def call(self, obs):
        """
        """
        value = self.layer1(obs)
        value = self.layer2(value)

        v = self.v(value)
        pi = self.pi(value)

        return v, pi

class Agent:
    def __init__(self, alpha=0.003, gamma=0.90, n_actions=2):
        self.alpha = alpha
        self.gamma = gamma
        self.n_actions = n_actions

        self.action = None
        self.action_space = [i for i in range(self.n_actions)]

        self.actor_critic = Actor_Critic(n_actions=n_actions)
        self.actor_critic.compile(optimizer = Adam(learning_rate=alpha))
    
    def get_action(self, obs):
        """
        """
        obs = tf.convert_to_tensor([obs])
        _, probs = self.actor_critic(obs)

        action_probs = tfp.distributions.Categorical(probs)
        action = action_probs.sample()
        self.action = action

        return action.numpy()[0]
    
    def save_model(self):
        """
        """
        self.actor_critic.save_weights(self.actor_critic.chkpt_dir_file)
    
    def load_model(self):
        """
        """
        self.actor_critic.load_weights(self.actor_critic.chkpt_dir_file)

    def learn(self, state, reward, state_, done):
        """
        """
        state = tf.convert_to_tensor([state])
        state_ = tf.convert_to_tensor([state_])
        reward = tf.convert_to_tensor(reward)
        
        with tf.GradientTape() as tape:
            state_value, probs = self.actor_critic(state)
            state_value_, _ = self.actor_critic(state_)
            state_value = tf.squeeze(state_value)
            state_value_ = tf.squeeze(state_value_)

            action_probs = tfp.distributions.Categorical(probs)
            log_prob = action_probs.log_prob(self.action)

            delta = reward + self.gamma*state_value_*(1-int(done)) - state_value
            actor_loss = -log_prob*delta
            critic_loss = delta **2
            total_loss = actor_loss + critic_loss

        grad = tape.gradient(total_loss, self.actor_critic.trainable_variables)

        self.actor_critic.optimizer.apply_gradients(zip(
            grad, self.actor_critic.trainable_variables))
