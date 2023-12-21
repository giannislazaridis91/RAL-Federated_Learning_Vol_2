import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from network import Network
from replay_buffer import ReplayBuffer

GAMMA = 0.95
EXPLORATION_MAX = 1.0
EXPLORATION_DECAY = 0.999
EXPLORATION_MIN = 0.001
DEVICE = torch.device("cpu")
BATCH_SIZE = 64

class DQN:

    def __init__(self, State_Length=30, Action_Length=3, learning_rate=1e-3):
        self.network = Network(state_length=State_Length, action_length=Action_Length)
        self.exploration_rate = EXPLORATION_MAX

    def get_action(self, state, action):
        self.i_actions_taken += 1
        classifier_state = np.repeat([classifier_state], np.shape(action_state)[1], axis=0)
        classifier_state = torch.from_numpy(classifier_state).to(self.device)
        action_state = torch.from_numpy(action_state.T).to(self.device)
        with torch.no_grad():
            p = self.estimator(classifier_state, action_state)
        p = p.cpu().numpy()
        action = np.random.choice(np.where(p == p.min())[0])
        return action

    def train(self, minibatch):
        self.i_train += 1
        self.optimizer.zero_grad()
        mpb = []

        states = torch.tensor(minibatch.states , dtype=torch.float32).to(DEVICE)
        actions = torch.tensor(minibatch.actions, dtype=torch.long).to(DEVICE)
        rewards = torch.tensor(minibatch.rewards, dtype=torch.float32).to(DEVICE)
        _states_ = torch.tensor(minibatch._state, dtype=torch.float32).to(DEVICE)
        _actions = torch.tensor(minibatch._actions, dtype=torch.float32).to(DEVICE)
        dones = torch.tensor(dones, dtype=torch.bool).to(DEVICE)
        batch_indices = np.arange(BATCH_SIZE, dtype=np.int64)



        for ncs, next_action_state in zip(mini.ncs, mini.next_action_state):
            n_next_actions = np.shape(next_action_state)[1]
            ncs = np.repeat([ncs], n_next_actions, axis=0)
            ncs = torch.from_numpy(ncs).to(self.device)
            next_action_state = torch.from_numpy(next_action_state.T).to(self.device)
            tp = self.target_estimator(ncs, next_action_state)
            p = self.estimator(ncs, next_action_state)
            tp = tp.cpu().numpy().ravel()
            p = p.cpu().numpy().ravel()
            bestabe = np.random.choice(np.where(p == np.amax(p))[0])
            mtpi = tp[bestabe]
            mpb.append(mtpi)
        nmas = [np.mean(action_state.T, axis=0) for action_state in mini.action_state]
        nmas = np.array(nmas)
        classifier_state = torch.from_numpy(mini.classifier_state).to(self.device)
        nmas = torch.from_numpy(nmas).to(self.device)
        mpb = torch.tensor(mpb, dtype=torch.float32).to(self.device)
        reward = torch.tensor(mini.reward, dtype=torch.float32).to(self.device)
        terminal = torch.tensor(mini.terminal, dtype=torch.bool).to(self.device)
        target_classifier_state = torch.from_numpy(mini.ncs).to(self.device)
        with torch.set_grad_enabled(True):
            output = self.estimator(classifier_state, nmas)
            loss = self.criterion(output, reward + mpb * (1 - terminal))
            loss.backward()
            self.optimizer.step()
            self._update_target_estimator(target_classifier_state, nmas)
        return output.detach().cpu().numpy()

    def _update_target_estimator(self, target_classifier_state, nmas):
        for target_param, param in zip(self.target_estimator.parameters(), self.estimator.parameters()):
            target_param.data.copy_(self.target_update_factor * param.data + (1 - self.target_update_factor) * target_param.data)