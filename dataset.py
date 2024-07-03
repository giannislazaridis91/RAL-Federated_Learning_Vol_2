import torch
from torch.utils.data import Dataset
from torchvision import datasets
import numpy as np

class CIFAR10Dataset(Dataset):
    def __init__(self, root_dir, length_of_client_data, data, labels):
        self.root_dir = root_dir
        self.number_of_state_data = int(15*length_of_client_data/100)
        self.number_of_warm_start_data = int(15*length_of_client_data/100)
        self.number_of_agent_data = int(70*length_of_client_data/100)
        self.data = data
        self.labels =labels
        self.regenerate()
        
    def regenerate(self):

        train_data = self.data
        train_labels = self.labels

        # Split train_data and train_labels to subsets per class.
        class_data = [[] for _ in range(10)]
        class_labels = [[] for _ in range(10)]
        
        for i in range(len(train_labels)):
            class_data[train_labels[i]].append(train_data[i])
            class_labels[train_labels[i]].append(train_labels[i])

        # Sample subset data from each class based on provided numbers.
        state_data = []
        state_labels = []
        warm_start_data = []
        warm_start_labels = []
        agent_data = []
        agent_labels = []

        for data_class in range(10):
            state_data.extend(class_data[data_class][:int(self.number_of_state_data / 10)])
            state_labels.extend(class_labels[data_class][:int(self.number_of_state_data / 10)])

            warm_start_data.extend(class_data[data_class][int(self.number_of_state_data / 10):int(self.number_of_state_data / 10) + int(self.number_of_warm_start_data / 10)])
            warm_start_labels.extend(class_labels[data_class][int(self.number_of_state_data / 10):int(self.number_of_state_data / 10) + int(self.number_of_warm_start_data / 10)])

            agent_data.extend(class_data[data_class][int(self.number_of_state_data / 10) + int(self.number_of_warm_start_data / 10):int(self.number_of_state_data / 10) + int(self.number_of_warm_start_data / 10) + int(self.number_of_agent_data / 10)])
            agent_labels.extend(class_labels[data_class][int(self.number_of_state_data / 10) + int(self.number_of_warm_start_data / 10):int(self.number_of_state_data / 10) + int(self.number_of_warm_start_data / 10) + int(self.number_of_agent_data / 10)])

        # Convert lists to numpy arrays and then to PyTorch tensors.
        self.state_data = torch.tensor(np.array(state_data))
        self.warm_start_data = torch.tensor(np.array(warm_start_data))
        self.agent_data = torch.tensor(np.array(agent_data))

        self.state_labels = torch.tensor(np.array(state_labels))
        self.warm_start_labels = torch.tensor(np.array(warm_start_labels))
        self.agent_labels = torch.tensor(np.array(agent_labels))

        # Load CIFAR-10 test dataset for evaluation.
        test_dataset = datasets.CIFAR10(root=self.root_dir, train=False, download=True, transform=None)
        self.test_data = torch.tensor(test_dataset.data)
        self.test_labels = torch.tensor(test_dataset.targets)

        self.number_of_classes = 10

        self._normalization()

    def _normalization(self):
        # Data normalization (assuming RGB images, divide by 255)
        self.state_data = self.state_data.float() / 255.0
        self.warm_start_data = self.warm_start_data.float() / 255.0
        self.agent_data = self.agent_data.float() / 255.0
        self.test_data = self.test_data.float() / 255.0

    def __len__(self):
        return len(self.state_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {
            'state_data': self.state_data[idx],
            'warm_start_data': self.warm_start_data[idx],
            'agent_data': self.agent_data[idx],
            'state_labels': self.state_labels[idx],
            'warm_start_labels': self.warm_start_labels[idx],
            'agent_labels': self.agent_labels[idx],
        }

        return sample