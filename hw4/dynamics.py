import tensorflow as tf
import numpy as np
import math

# Predefined function to build a feedforward neural network
def build_mlp(input_placeholder, 
              output_size,
              scope, 
              n_layers=2, 
              size=500, 
              activation=tf.tanh,
              output_activation=None
              ):
    out = input_placeholder
    with tf.variable_scope(scope):
        for _ in range(n_layers):
            out = tf.layers.dense(out, size, activation=activation)
        out = tf.layers.dense(out, output_size, activation=output_activation)
    return out

class NNDynamicsModel():
    def __init__(self, 
                 env, 
                 n_layers,
                 size, 
                 activation, 
                 output_activation, 
                 normalization,
                 batch_size,
                 iterations,
                 learning_rate,
                 sess
                 ):
        """ YOUR CODE HERE """
        """ Note: Be careful about normalization """
        self.mean_obs, self.std_obs, self.mean_deltas, self.std_deltas, self.mean_action, self.std_action = normalization
        self.sess = sess
        self.batch_size = batch_size
        self.iter = iterations

        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        self.s_a_ph = tf.placeholder(shape = [None, self.state_dim + self.action_dim], name = 's_a_ph', dtype = tf.float32)
        self.delta_ph = tf.placeholder(shape = [None, self.state_dim], name = 'delta_ph', dtype = tf.float32)
        self.delta_pred = build_mlp(self.s_a_ph, self.state_dim, "dynamics", n_layers = n_layers, size = size, 
                                activation = activation, output_activation = output_activation)
        self.loss = tf.reduce_mean(tf.square(self.delta_pred - self.delta_ph))
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

    def fit(self, data):
        """
        Write a function to take in a dataset of (unnormalized)states, 
        (unnormalized)actions, (unnormalized)next_states and fit the dynamics model going from normalized states, 
        normalized actions to normalized state differences (s_t+1 - s_t)
        """

        un_states = np.concatenate([d['observations'] for d in data])
        un_actions = np.concatenate([d['actions'] for d in data])
        un_states_next = np.concatenate([d['next_observations'] for d in data])
        N = un_states.shape[0]
        indices = np.arange(N)

        n_states = (un_states - self.mean_obs) / (self.std_obs + 1e-7)
        n_deltas = ((un_states_next - un_states) - self.mean_deltas) / (self.std_deltas + 1e-7)
        n_actions = (un_actions - self.mean_action) / (self.std_action + 1e-7)

        state_action = np.concatenate((n_states, n_actions), axis = 1)

        for _ in range(self.iter):
            np.random.shuffle(indices)
            batches = int (math.ceil(N / self.batch_size))
            for i in range(batches):
                start_idx = i * self.batch_size
                idxs = indices[start_idx : start_idx + self.batch_size]
                batch_s_a = state_action[idxs, :]
                batch_delta = n_deltas[idxs, :]
                self.sess.run(self.optimizer, feed_dict = {self.s_a_ph : batch_s_a, self.delta_ph : batch_delta})

    def predict(self, states, actions):
        """ Write a function to take in a batch of (unnormalized) states and (unnormalized) 
        actions and return the (unnormalized) next states as predicted by using the model """
        
        n_states = (states - self.mean_obs) / (self.std_obs + 1e-7)
        n_actions = (actions - self.mean_action) / (self.std_action + 1e-7)
        print(n_states.shape)
        print(n_actions.shape)
        state_action = np.concatenate((n_states, n_actions), axis = 1)

        expected_deltas = self.sess.run(self.delta_pred, feed_dict = {self.s_a_ph : state_action})

        return expected_deltas * self.std_deltas + self.mean_deltas + states
