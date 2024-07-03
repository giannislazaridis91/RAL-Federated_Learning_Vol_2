import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LalEnv(object):

    def __init__(self, dataset, model, epochs, classifier_batch_size, target_precision):

        # Define the device and move model to device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.dataset = dataset
        self.model = model
        self.epochs = epochs
        self.classifier_batch_size = classifier_batch_size
        self.target_precision = target_precision
        self.number_of_classes = np.size(np.unique(self.dataset.state_labels))
        self.episode_qualities = []
        self.rewards_bank = []
        self.batch_bank = []
        self.precision_bank = []

    def reset(self, number_of_first_samples=10, code_state="", target_precision=0.0, target_budget=1.0):
        self.dataset.regenerate()
        self.episode_qualities.append(0)
        self.rewards_bank.append(0)
        self.batch_bank.append(0.1)
        self.precision_bank.append(0)
        self.code_state = code_state
        self.target_precision = target_precision
        self.target_budget = target_budget

        if self.code_state == "Warm-Start":
            if number_of_first_samples < self.number_of_classes:
                print(f'number_of_first_samples {number_of_first_samples} is less than the number of classes {self.number_of_classes}, so we change it.')
                number_of_first_samples = self.number_of_classes

            self.indices_known = []
            self.indices_unknown = []
            for i in np.unique(self.dataset.warm_start_labels.numpy()):  # Convert torch.Tensor to numpy array.
                cl = np.nonzero(self.dataset.warm_start_labels.numpy() == i)[0]
                indices = np.random.permutation(cl)
                self.indices_known.append(indices[0])
                self.indices_unknown.extend(indices[1:])
            self.indices_known = np.array(self.indices_known)
            self.indices_unknown = np.array(self.indices_unknown)
            self.indices_unknown = np.random.permutation(self.indices_unknown)

            if number_of_first_samples > self.number_of_classes:
                self.indices_known = np.concatenate(
                    (self.indices_known, self.indices_unknown[:number_of_first_samples - self.number_of_classes]))
                self.indices_unknown = self.indices_unknown[number_of_first_samples - self.number_of_classes:]

            known_data = self.dataset.warm_start_data[self.indices_known, :]
            known_labels = self.dataset.warm_start_labels[self.indices_known]
            known_labels_one_hot_encoding = one_hot(torch.tensor(known_labels),
                                                    num_classes=self.dataset.number_of_classes).float()

            train_loader = DataLoader(TensorDataset(torch.tensor(known_data).float().to(self.device), known_labels_one_hot_encoding.to(self.device)),
                            batch_size=self.classifier_batch_size, shuffle=True)

            self.model.train()
            optimizer = optim.Adam(self.model.parameters())
            criterion = nn.CrossEntropyLoss()

            best_val_acc = 0
            patience_counter = 0
            for epoch in range(self.epochs):
                epoch_loss = 0
                for batch_data, batch_labels in train_loader:
                    batch_data, batch_labels = batch_data.to(self.device), batch_labels.to(self.device)
                    batch_data = batch_data.permute(0, 3, 1, 2).flatten(1, 2)  # This is where the correction is made
                    optimizer.zero_grad()
                    outputs = self.model(batch_data)
                    loss = criterion(outputs, batch_labels)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

                val_acc = self._evaluate_model()
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(self.model.state_dict(), 'best_model.pth')
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter > 5:
                        self.model.load_state_dict(torch.load('best_model.pth'))
                        break

            self.model.eval()
            with torch.no_grad():
                predictions = self.model(torch.tensor(self.dataset.test_data).float().permute(0, 3, 1, 2).to(self.device))
                predicted_labels = torch.argmax(predictions, dim=1).cpu().numpy()
                test_labels_one_hot_encoding = one_hot(torch.tensor(self.dataset.test_labels),
                                                    num_classes=self.dataset.number_of_classes).float()
                true_labels = np.argmax(test_labels_one_hot_encoding, axis=1)
                precision_scores = precision_score(true_labels, predicted_labels, average='weighted', zero_division=0)

            self.episode_qualities.append(best_val_acc)
            self.precision_bank.append(precision_scores)
            self.batch_bank.append(number_of_first_samples / len(self.dataset.warm_start_data))
        
        elif self.code_state == "Agent":
            if number_of_first_samples < self.number_of_classes:
                print(f'number_of_first_samples {number_of_first_samples} is less than the number of classes {self.number_of_classes}, so we change it.')
                number_of_first_samples = self.number_of_classes

            self.indices_known = []
            self.indices_unknown = []
            for i in np.unique(self.dataset.agent_labels.numpy()):  # Convert torch.Tensor to numpy array.
                cl = np.nonzero(self.dataset.agent_labels.numpy() == i)[0]
                indices = np.random.permutation(cl)
                self.indices_known.append(indices[0])
                self.indices_unknown.extend(indices[1:])
            self.indices_known = np.array(self.indices_known)
            self.indices_unknown = np.array(self.indices_unknown)
            self.indices_unknown = np.random.permutation(self.indices_unknown)

            if number_of_first_samples > self.number_of_classes:
                self.indices_known = np.concatenate(
                    (self.indices_known, self.indices_unknown[:number_of_first_samples - self.number_of_classes]))
                self.indices_unknown = self.indices_unknown[number_of_first_samples - self.number_of_classes:]

            known_data = self.dataset.agent_data[self.indices_known, :]
            known_labels = self.dataset.agent_labels[self.indices_known]
            known_labels_one_hot_encoding = one_hot(torch.tensor(known_labels),
                                                    num_classes=self.dataset.number_of_classes).float()

            train_loader = DataLoader(TensorDataset(torch.tensor(known_data).float().to(self.device), known_labels_one_hot_encoding.to(self.device)),
                            batch_size=self.classifier_batch_size, shuffle=True)

            self.model.train()
            optimizer = optim.Adam(self.model.parameters())
            criterion = nn.CrossEntropyLoss()

            best_val_acc = 0
            patience_counter = 0
            for epoch in range(self.epochs):
                epoch_loss = 0
                for batch_data, batch_labels in train_loader:
                    batch_data, batch_labels = batch_data.to(self.device), batch_labels.to(self.device)
                    batch_data = batch_data.permute(0, 3, 1, 2).flatten(1, 2)  # This is where the correction is made
                    optimizer.zero_grad()
                    outputs = self.model(batch_data)
                    loss = criterion(outputs, batch_labels)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

                val_acc = self._evaluate_model()
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(self.model.state_dict(), 'best_model.pth')
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter > 5:
                        self.model.load_state_dict(torch.load('best_model.pth'))
                        break

            self.model.eval()
            with torch.no_grad():
                predictions = self.model(torch.tensor(self.dataset.test_data).float().permute(0, 3, 1, 2).to(self.device))
                predicted_labels = torch.argmax(predictions, dim=1).cpu().numpy()
                test_labels_one_hot_encoding = one_hot(torch.tensor(self.dataset.test_labels),
                                                    num_classes=self.dataset.number_of_classes).float()
                true_labels = np.argmax(test_labels_one_hot_encoding, axis=1)
                precision_scores = precision_score(true_labels, predicted_labels, average='weighted', zero_division=0)

            self.episode_qualities.append(best_val_acc)
            self.precision_bank.append(precision_scores)
            self.batch_bank.append(number_of_first_samples / len(self.dataset.agent_data))

        state, next_action = self._get_state()
        self.n_actions = np.size(self.indices_unknown)

        return state, next_action

    def step(self, batch_actions_indices):
        # The batch_actions_indices value indicates the positions
        # of the batch of data points in self.indices_unknown that we want to sample in unknown_data.
        # The index in train_data should be retrieved.

        if self.code_state == "Warm-Start":

            selection_absolute = self.indices_unknown[batch_actions_indices]
            self.indices_known = self.indices_known.flatten()
            selection_absolute = selection_absolute.flatten()
            self.indices_known = np.concatenate((self.indices_known, selection_absolute))
            self.indices_unknown = np.delete(self.indices_unknown, batch_actions_indices)
            # Train a model with new labeled data:
            known_data = self.dataset.warm_start_data[self.indices_known, :]
            known_labels = self.dataset.warm_start_labels[self.indices_known]
            known_labels_one_hot_encoding = one_hot(torch.tensor(known_labels),
                                                    num_classes=self.dataset.number_of_classes).float()

            train_loader = DataLoader(TensorDataset(torch.tensor(known_data).float(), known_labels_one_hot_encoding),
                                    batch_size=self.classifier_batch_size, shuffle=True)

            self.model.train()
            optimizer = optim.Adam(self.model.parameters())
            criterion = nn.CrossEntropyLoss()

            best_val_acc = 0
            patience_counter = 0
            for epoch in range(self.epochs):
                epoch_loss = 0
                for batch_data, batch_labels in train_loader:
                    batch_data, batch_labels = batch_data.to(self.device), batch_labels.to(self.device)
                    batch_data = batch_data.permute(0, 3, 1, 2).flatten(1, 2)  # This is where the correction is made
                    optimizer.zero_grad()
                    outputs = self.model(batch_data)
                    loss = criterion(outputs, batch_labels)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

                val_acc = self._evaluate_model()
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(self.model.state_dict(), 'best_model.pth')
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter > 5:
                        self.model.load_state_dict(torch.load('best_model.pth'))
                        break

            # Compute the quality of the current classifier.
            new_score = best_val_acc  # Testing accuracy.
            self.episode_qualities.append(new_score)

            # Compute the precision:
            self.model.eval()
            with torch.no_grad():
                predictions = self.model(torch.tensor(self.dataset.test_data).float().permute(0, 3, 1, 2).to(self.device))            
                predicted_labels = torch.argmax(predictions, dim=1).cpu().numpy()
                test_labels_one_hot_encoding = one_hot(torch.tensor(self.dataset.test_labels),
                                                    num_classes=self.dataset.number_of_classes).float()
                true_labels = np.argmax(test_labels_one_hot_encoding, axis=1)
                precision_scores = precision_score(true_labels, predicted_labels, average='weighted', zero_division=0)

            self.precision_bank.append(precision_scores)

            # Update batch bank and training losses.
            self.batch_bank.append(len(batch_actions_indices) / len(self.dataset.warm_start_data))
    
            # Get the new state and the next action.
            state, next_action = self._get_state()

            # Compute the reward.
            reward = self._compute_reward()

            # Check if this episode terminated.
            done = self._compute_is_terminal()

            if isinstance(state, torch.Tensor):
                state = state.cpu()
            if isinstance(next_action, torch.Tensor):
                next_action = next_action.cpu()
            if isinstance(reward, torch.Tensor):
                reward = reward.cpu()
            if isinstance(done, torch.Tensor):
                done = done.cpu()

        if self.code_state == "Agent":

            selection_absolute = self.indices_unknown[batch_actions_indices]
            self.indices_known = self.indices_known.flatten()
            selection_absolute = selection_absolute.flatten()
            self.indices_known = np.concatenate((self.indices_known, selection_absolute))
            self.indices_unknown = np.delete(self.indices_unknown, batch_actions_indices)
            # Train a model with new labeled data:
            known_data = self.dataset.agent_data[self.indices_known, :]
            known_labels = self.dataset.agent_labels[self.indices_known]
            known_labels_one_hot_encoding = one_hot(torch.tensor(known_labels),
                                                    num_classes=self.dataset.number_of_classes).float()

            train_loader = DataLoader(TensorDataset(torch.tensor(known_data).float(), known_labels_one_hot_encoding),
                                    batch_size=self.classifier_batch_size, shuffle=True)

            self.model.train()
            optimizer = optim.Adam(self.model.parameters())
            criterion = nn.CrossEntropyLoss()

            best_val_acc = 0
            patience_counter = 0
            for epoch in range(self.epochs):
                epoch_loss = 0
                for batch_data, batch_labels in train_loader:
                    batch_data, batch_labels = batch_data.to(self.device), batch_labels.to(self.device)
                    batch_data = batch_data.permute(0, 3, 1, 2).flatten(1, 2)  # This is where the correction is made
                    optimizer.zero_grad()
                    outputs = self.model(batch_data)
                    loss = criterion(outputs, batch_labels)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

                val_acc = self._evaluate_model()
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(self.model.state_dict(), 'best_model.pth')
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter > 5:
                        self.model.load_state_dict(torch.load('best_model.pth'))
                        break

            # Compute the quality of the current classifier.
            new_score = best_val_acc  # Testing accuracy.
            self.episode_qualities.append(new_score)

            # Compute the precision:
            self.model.eval()
            with torch.no_grad():
                predictions = self.model(torch.tensor(self.dataset.test_data).float().permute(0, 3, 1, 2).to(self.device))            
                predicted_labels = torch.argmax(predictions, dim=1).cpu().numpy()
                test_labels_one_hot_encoding = one_hot(torch.tensor(self.dataset.test_labels),
                                                    num_classes=self.dataset.number_of_classes).float()
                true_labels = np.argmax(test_labels_one_hot_encoding, axis=1)
                precision_scores = precision_score(true_labels, predicted_labels, average='weighted', zero_division=0)

            self.precision_bank.append(precision_scores)

            # Update batch bank and training losses.
            self.batch_bank.append(len(batch_actions_indices) / len(self.dataset.agent_data))
    
            # Get the new state and the next action.
            state, next_action = self._get_state()

            # Compute the reward.
            reward = self._compute_reward()

            # Check if this episode terminated.
            done = self._compute_is_terminal()

            if isinstance(state, torch.Tensor):
                state = state.cpu()
            if isinstance(next_action, torch.Tensor):
                next_action = next_action.cpu()
            if isinstance(reward, torch.Tensor):
                reward = reward.cpu()
            if isinstance(done, torch.Tensor):
                done = done.cpu()

        return state, next_action, self.indices_unknown, reward, done

    def _get_state(self):
        with torch.no_grad():
            predictions = self.model(torch.tensor(self.dataset.state_data).float().permute(0, 3, 1, 2).to(self.device)).cpu().numpy()[:, 0]            
            idx = np.argsort(predictions)
            state = predictions[idx]
        # Compute next_action.
        if self.code_state == "Warm-Start":
            unknown_data = self.dataset.warm_start_data[self.indices_unknown,:]
        elif self.code_state == "Agent":
            unknown_data = self.dataset.agent_data[self.indices_unknown,:]     
        next_action = [np.array([i]) for i in range(1, len(unknown_data) + 1)]
        self.n_actions = len(unknown_data)
        return state, next_action

    def _compute_reward(self):
        reward = 0.0
        return reward

    def _compute_is_terminal(self):
        done = self.n_actions == 0
        return done


class LalEnvFirstAccuracy(LalEnv):

    def __init__(self, dataset, model, epochs, classifier_batch_size, target_precision):
        super().__init__(dataset, model, epochs, classifier_batch_size, target_precision)

    def reset(self, number_of_first_samples=10, code_state="", target_precision=0.0, target_budget=1.0):
        state, next_action = super().reset(number_of_first_samples=number_of_first_samples, code_state=code_state, target_precision=target_precision,
                                           target_budget=target_budget)
        current_reward = self._compute_reward()
        self.rewards_bank.append(current_reward)
        if isinstance(state, torch.Tensor):
            state = state.cpu()
        if isinstance(next_action, torch.Tensor):
            next_action = next_action.cpu()
        if isinstance(current_reward, torch.Tensor):
            current_reward = current_reward.cpu()

        return state, next_action, self.indices_unknown, current_reward

    def _compute_reward(self):
        new_score = self.precision_bank[-1] / self.batch_bank[-1]
        previous_score = self.precision_bank[-2] / self.batch_bank[-2]
        reward = new_score - previous_score + 0.01 * (1 - np.random.rand())
        self.rewards_bank.append(reward)
        return reward

    def _compute_is_terminal(self):
        done = super()._compute_is_terminal()
        percentage_precision = 100
        percentage_budget = 50
        if self.code_state!="Warm-Start":
            if (((percentage_precision * self.target_precision) / 100) <= self.precision_bank[-1]) and (
                    ((percentage_budget * self.target_budget) / 100) > self.batch_bank[-1]):
                print("-- Exceed target precision with lower budget, so this is the end of the episode.")
                done = True
        else:
            if len(self.rewards_bank) >= 4 and all(
                    x < y for x, y in zip(self.rewards_bank[-3:], self.rewards_bank[-4:-1])):
                done = True
        return done

    def return_episode_qualities(self):
        return self.episode_qualities

    def return_episode_precisions(self):
        return self.precision_bank

    def _evaluate_model(self):
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(torch.tensor(self.dataset.test_data).float().permute(0, 3, 1, 2).to(self.device))
            predicted_labels = torch.argmax(predictions, dim=1).cpu().numpy()
            test_labels_one_hot_encoding = one_hot(torch.tensor(self.dataset.test_labels), num_classes=self.dataset.number_of_classes).float()
            true_labels = torch.tensor(np.argmax(test_labels_one_hot_encoding, axis=1)).to(self.device)
            correct = (predicted_labels == true_labels.cpu().numpy()).sum()
            accuracy = correct / len(true_labels)
        return accuracy