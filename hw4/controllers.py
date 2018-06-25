import numpy as np
from cost_functions import trajectory_cost_fn
import time

class Controller():
	def __init__(self):
		pass

	# Get the appropriate action(s) for this state(s)
	def get_action(self, state):
		pass


class RandomController(Controller):
	def __init__(self, env):
		self.env = env

	def get_action(self, state):
		""" Your code should randomly sample an action uniformly from the action space """
		return self.env.action_space.sample()


class MPCcontroller(Controller):
	""" Controller built using the MPC method outlined in https://arxiv.org/abs/1708.02596 """
	def __init__(self, 
				 env, 
				 dyn_model, 
				 horizon=5, 
				 cost_fn=None, 
				 num_simulated_paths=10,
				 ):
		self.env = env
		self.dyn_model = dyn_model
		self.horizon = horizon
		self.cost_fn = cost_fn
		self.num_simulated_paths = num_simulated_paths

	def get_action(self, state):
		""" Note: be careful to batch your simulations through the model for speed """
		
		state_batch, states, next_states, actions = [], [], [], []

		#state batches have dimension (K, dim(state))
		for _ in range(self.num_simulated_paths):
			state_batch.append(state)

		for _ in range(self.horizon):
			action = []
			for _ in range(self.num_simulated_paths):
				action.append(self.env.action_space.sample())
			actions.append(action)
			states.append(state_batch)
			#use batch for speed
			state_batch = self.dyn_model.predict(np.array(state_batch), np.array(action))

			next_states.append(state_batch)

		costs = trajectory_cost_fn(self.cost_fn, np.array(states), np.array(actions), np.array(next_states))
		j_star = np.argmin(np.array(costs))
		return actions[0][j_star]

