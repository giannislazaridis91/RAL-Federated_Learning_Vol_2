import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from batch_estimator import Estimator

class DQN:

    def __init__(self,
                 observation_length=6,
                 learning_rate=1e-3,
                 batch_size=32,
                 target_copy_factor=0.001,
                 bias_average=0,
                 round = 1):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.i_train = 0
        self.i_actions_taken = 0
        self.batch_size = batch_size
        self.target_copy_factor = target_copy_factor
        self.round = round

        # TARGET ESTIMATOR
        self.target_estimator = Estimator(observation_length, is_target_dqn=True, bias_average=bias_average).to(
            self.device)

        # ESTIMATOR
        self.estimator = Estimator(observation_length, is_target_dqn=False, bias_average=bias_average).to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(self.estimator.parameters(), lr=learning_rate)

        # Loss function
        self.loss_fn = nn.MSELoss()

        # Mode flag to initialize model parameters
        self._initialized = False

        # Check if we need to load weights.
        if self.round > 1:
            self.load_weights()

    def _check_initialized(self):
        if not self._initialized:
            self.estimator.apply(self._weights_init)
            self.target_estimator.load_state_dict(self.estimator.state_dict())
            self._initialized = True

    def _weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    # Save DQN weights.
    def save_weights(self, filename='dqn_weights.pth'):
        torch.save({
            'estimator_state_dict': self.estimator.state_dict(),
            'target_estimator_state_dict': self.target_estimator.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'round': self.round
        }, filename)
        print(f"Saved weights to '{filename}'.")

    # Load DQN weights.
    def load_weights(self, filename='dqn_weights.pth'):
        checkpoint = torch.load(filename, map_location=self.device)
        self.estimator.load_state_dict(checkpoint['estimator_state_dict'])
        self.target_estimator.load_state_dict(checkpoint['target_estimator_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.round = checkpoint['round']
        print(f"Loaded weights from '{filename}' for round {self.round}.")

    def get_action(self, dataset, model, state, next_action_batch, next_action_unlabeled_data):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.i_actions_taken += 1
        self._check_initialized()

        if not isinstance(state, np.ndarray):
            state = state.cpu().numpy()
        if not isinstance(next_action_batch, np.ndarray):
            if isinstance(next_action_batch, list):
                next_action_batch = torch.tensor(next_action_batch).float().to(device)
            next_action_batch = next_action_batch.cpu().numpy()
        if not isinstance(next_action_unlabeled_data, np.ndarray):
            next_action_unlabeled_data = next_action_unlabeled_data.cpu().numpy()

        state = np.repeat([state], len(next_action_batch), axis=0)

        state_tensor = torch.from_numpy(state).float().to(device)
        next_action_batch_tensor = torch.from_numpy(next_action_batch).float().to(device)

        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)
        if next_action_batch_tensor.dim() == 1:
            next_action_batch_tensor = next_action_batch_tensor.unsqueeze(0)

        with torch.no_grad():
            predictions = self.estimator(state_tensor, next_action_batch_tensor)

        selected_batch = np.random.choice(np.where(predictions.cpu().numpy() == predictions.max().item())[0])
        selected_batch = int(next_action_batch[selected_batch])

        input_data = dataset.agent_data[next_action_unlabeled_data, :]

        # Print input data shape for debugging
        print(f"Input data shape: {input_data.shape}")

        # Correcting input data shape to check the last dimension
        if input_data.shape[-1] != 3:
            raise ValueError(f"Expected input data to have 3 channels, but got {input_data.shape[-1]} channels instead")

        # Ensure input_data is a NumPy array
        if isinstance(input_data, torch.Tensor):
            input_data = input_data.cpu().numpy()

        input_data_tensor = torch.from_numpy(input_data).float().to(device)

        with torch.no_grad():
            train_predictions = model(input_data_tensor)

        uncertainty_scores = -np.abs(train_predictions[:, 0].cpu().numpy())
        selected_indices = np.argsort(uncertainty_scores)[:selected_batch]

        if isinstance(selected_batch, torch.Tensor):
            selected_batch = selected_batch.cpu()

        return selected_batch, selected_indices

    def train(self, minibatch):
        self._check_initialized()
        max_prediction_batch = []

        for i in range(len(minibatch.next_state)):
            next_state = np.repeat([minibatch.next_state[i]], len(minibatch.next_action[i]), axis=0)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32).to(self.device)
            next_action_tensor = torch.tensor(minibatch.next_action[i], dtype=torch.float32).to(self.device)
            state_tensor = torch.tensor(minibatch.state, dtype=torch.float32).to(self.device)
            action_tensor = torch.tensor([[x] for x in minibatch.action], dtype=torch.float32).to(self.device)

            with torch.no_grad():
                target_predictions = self.target_estimator(next_state_tensor, next_action_tensor)
                predictions = self.estimator(state_tensor, action_tensor)

            # Find the best action by estimator.
            if len(predictions) == 0:
                # Handle case where predictions is empty (no valid actions).
                best_action_by_estimator = 0  # Choose a default action or handle appropriately.
            else:
                best_action_by_estimator = torch.argmax(predictions).item()

            # Ensure best_action_by_estimator is within valid range
            if best_action_by_estimator >= len(target_predictions):
                best_action_by_estimator = len(target_predictions) - 1  # Clamp to the last valid index.

            max_target_prediction_i = target_predictions[best_action_by_estimator].item()
            max_prediction_batch.append(max_target_prediction_i)

        state_tensor = torch.tensor(minibatch.state, dtype=torch.float32).to(self.device)
        action_tensor = torch.tensor([[x] for x in minibatch.action], dtype=torch.float32).to(self.device)
        reward_tensor = torch.tensor(minibatch.reward, dtype=torch.float32).to(self.device)
        terminal_tensor = torch.tensor(minibatch.terminal, dtype=torch.bool).to(self.device)
        max_prediction_tensor = torch.tensor(max_prediction_batch, dtype=torch.float32).to(self.device)

        self.optimizer.zero_grad()

        predictions = self.estimator(state_tensor, action_tensor)
        terminal_mask = ~terminal_tensor
        masked_target_predictions = max_prediction_tensor * terminal_mask

        actions_taken_targets = reward_tensor + masked_target_predictions
        actions_taken_targets = actions_taken_targets.view(self.batch_size, 1)

        td_error = actions_taken_targets - predictions
        loss = torch.sum(td_error ** 2)

        loss.backward()
        self.optimizer.step()

        # Update target estimator
        for target_param, param in zip(self.target_estimator.parameters(), self.estimator.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.target_copy_factor) + param.data * self.target_copy_factor)

        return td_error.cpu().detach().numpy()