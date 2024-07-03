import numpy as np
import torch

class Minibatch:

    def __init__(self, state, action, reward, next_state, next_action, terminal, indices):
        # Inits the Minibatch object and initializes the attributes with given values.
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.next_action = next_action
        self.terminal = terminal
        self.indices = indices


class ReplayBuffer:

    def __init__(self, buffer_size=1e4, prior_exp=0.5):

        # Inits a few attributes with 0 or the default values.
        self.buffer_size = int(buffer_size)
        self.n = 0
        self.write_index = 0
        self.max_td_error = 1000.0
        self.prior_exp = prior_exp

    def _init_nparray(self, state, action, reward, next_state, next_action, terminal):

        # Initialize numpy arrays of all_xxx attributes to one transaction repeated buffer_size times.
        self.all_states = np.array([state] * self.buffer_size)
        self.all_action = np.array([action] * self.buffer_size)
        self.all_rewards = np.array([reward] * self.buffer_size)
        self.all_next_states = np.array([next_state] * self.buffer_size)
        self.all_next_actions = [next_action] * self.buffer_size
        self.all_terminals = np.array([terminal] * self.buffer_size)
        self.all_td_errors = np.array([self.max_td_error] * self.buffer_size)

        # Set the counters to 1 as one transaction is stored.
        self.n = 1
        self.write_index = 1

    def store_transition(self, state, action, reward, next_state, next_action, terminal):

        # Add a new transaction to a replay buffer.
        # If buffer arrays are not yet initialized, initialize it.
        if self.n == 0:
            self._init_nparray(state, action, reward, next_state, next_action, terminal)
            return
        
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        if isinstance(reward, torch.Tensor):
            reward = reward.cpu().numpy()
        if isinstance(next_state, torch.Tensor):
            next_state = next_state.cpu().numpy()
        if isinstance(next_action, torch.Tensor):
            next_action = next_action.cpu().numpy()
        if isinstance(terminal, torch.Tensor):
            terminal = terminal.cpu().numpy()
            
        # Write a transaction at a write_index position.
        self.all_states[self.write_index] = state
        self.all_action[self.write_index] = action
        self.all_rewards[self.write_index] = reward
        self.all_next_states[self.write_index] = next_state
        self.all_next_actions[self.write_index] = next_action
        self.all_terminals[self.write_index] = terminal
        self.all_td_errors[self.write_index] = self.max_td_error

        # Keep track of the index for writing.
        self.write_index += 1
        if self.write_index >= self.buffer_size:
            self.write_index = 0

        # Keep track of the max index to be used for sampling.
        if self.n < self.buffer_size:
            self.n += 1

    def sample_minibatch(self, batch_size=32):

        # Get td error of samples that were written in the buffer.
        td_errors_to_consider = self.all_td_errors[:self.n]

        # Scale and normalize the td error to turn it into a probability for sampling.
        p = np.power(td_errors_to_consider, self.prior_exp) / np.sum(np.power(td_errors_to_consider, self.prior_exp))

        # Choose indices to sample according to the computed probability.
        # The higher the td error is, the more likely it is that the sample will be selected.
        minibatch_indices = np.random.choice(range(self.n), size=batch_size, replace=True, p=p)

        minibatch = Minibatch(
            self.all_states[minibatch_indices],
            [self.all_action[i] for i in minibatch_indices],
            self.all_rewards[minibatch_indices],
            self.all_next_states[minibatch_indices],
            [self.all_next_actions[i] for i in minibatch_indices],
            self.all_terminals[minibatch_indices],
            minibatch_indices,
        )

        return minibatch

    def update_td_errors(self, td_errors, indices):

        # Set the values for prioritized replay to the most recent td errors.
        self.all_td_errors[indices] = np.ravel(np.absolute(td_errors))

        # Find the max error from the replay buffer that will be used as a default value for new transactions.
        self.max_td_error = np.max(self.all_td_errors)