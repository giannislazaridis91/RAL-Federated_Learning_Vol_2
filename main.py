import random
import numpy as np
import copy
import os
import shutil
import wandb

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision.datasets import CIFAR10

from dataset import CIFAR10Dataset
from central_server_ral import CentralFed
from torchvision import transforms
from batch_dqn import DQN
from batch_envs import LalEnvFirstAccuracy
from batch_helpers import ReplayBuffer

wandb.login()

# Classifier parameters.

CLASSIFIER_NUMBER_OF_CLASSES = 10
CLASSIFIER_NUMBER_OF_EPOCHS = 10
CLASSIFIER_LEARNING_RATE = 0.01
CLASSIFIER_BATCH_SIZE = 64

# Parameters for both agents.

REPLAY_BUFFER_SIZE = 5e4
PRIOROTIZED_REPLAY_EXPONENT = 3
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
TARGET_COPY_FACTOR = 0.01
BIAS_INITIALIZATION = 0

# BatchAgent's parameters.

DIRNAME = './batch_agent/' # The resulting batch_agent of this experiment will be written in a file.
WARM_START_EPISODES_BATCH_AGENT = 5
NN_UPDATES_PER_EPOCHS_BATCH_AGENT = 50
TRAINING_EPOCHS_BATCH_AGENT = 25
TRAINING_EPISODES_PER_EPOCH_BATCH_AGENT = 5

# FL parameters.
FED_ROUNDS = 20
N_CLIENTS = 5


cwd = os.getcwd()

# Delete following directories if they exist.
for directory in [cwd+'/__pycache__', cwd+'/wandb', cwd+'/batch_agent', cwd+'/libact', cwd+'/AL_results', cwd+'/checkpoints', cwd+'/summaries', cwd+'/data', cwd+'/data_client']:
    if os.path.exists(directory):
        shutil.rmtree(directory, ignore_errors=True)



def get_cifar10_splited_big_common(num_clients, trans,
                                   root='./data',
                                   special_client_size=0):
    special_indices_per_class_from_total = int(
        special_client_size / 10)
    if num_clients < 2:
        raise ValueError("Number of clients must be at least 2.")
    trainset = CIFAR10(root=root, train=True, download=True, transform=trans)
    indices = torch.randperm(len(trainset)).tolist()
    class_indices = [[] for _ in range(10)]
    for idx in indices:
        _, label = trainset[idx]
        class_indices[label].append(idx)
    special_client_indices = []
    for class_list in class_indices:
        special_client_indices.extend(class_list[:special_indices_per_class_from_total])
        del class_list[:special_indices_per_class_from_total]
    remaining_images_per_class = len(class_indices[0])
    images_per_class_per_client = remaining_images_per_class // (num_clients - 1)
    client_indices = [special_client_indices]
    for _ in range(num_clients - 1):
        client_subset = []
        for class_list in class_indices:
            client_subset.extend(class_list[:images_per_class_per_client])
            del class_list[:images_per_class_per_client]
        client_indices.append(client_subset)
    return client_indices, trainset

print("# Define the number of clients and the size of the special client's dataset.")
# Define the number of clients and the size of the special client's dataset.
num_clients = 5
special_client_size = 10000
root = './data'

print("# Define the transformations to be applied to the dataset.")
# Define the transformations to be applied to the dataset.
trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

print("# Call the function.")

# Call the function.
client_indices, trainset = get_cifar10_splited_big_common(
    num_clients=num_clients,
    trans=trans,
    root=root,
    special_client_size=special_client_size
)

print("# Print some information about the output.")
# Print some information about the output.
for i, indices in enumerate(client_indices):
    print(f"Client {i} has {len(indices)} images.")

# Example to access the dataset for a specific client.
client_data = torch.utils.data.Subset(trainset, client_indices[0])
first_client_indices = client_indices[0]
subset_data_first_client = torch.tensor(trainset.data[first_client_indices])
subset_labels_first_client = torch.tensor([trainset.targets[i] for i in first_client_indices])
dataset = CIFAR10Dataset(root_dir= './data_client',length_of_client_data=len(client_data), data = subset_data_first_client, labels = subset_labels_first_client)
datasets = [CIFAR10Dataset(root_dir= './data_client',
                           length_of_client_data=len(torch.utils.data.Subset(trainset, client_indices[i])),
                           data = subset_data_first_client,
                           labels = subset_labels_first_client) for i in range(N_CLIENTS)]



print("Warm-start data are {}.".format(len(dataset.warm_start_data)))
print("State data are {}.".format(len(dataset.state_data)))
print("Agent data are {}.".format(len(dataset.agent_data)))
print("Test data are {}.".format(len(dataset.test_data)))

run = wandb.init(
    # Set the project where this run will be logged.
    project="RAL_Image_Classification_3",
    # Track hyperparameters and run metadata.
    config={
        "Learning_rate": CLASSIFIER_LEARNING_RATE,
        "Classifier_epochs": CLASSIFIER_NUMBER_OF_EPOCHS,
        "Number_of_classes": CLASSIFIER_NUMBER_OF_CLASSES,
        "Dataset": "CIFAR10",
        "Classifier": "Pre-trained ResNet50",
        "Warm-start episodes for the BatchAgent": WARM_START_EPISODES_BATCH_AGENT,
        "Training epochs for BatchAgent": TRAINING_EPOCHS_BATCH_AGENT,
        "Training episodes per epoch for BatchAgent": TRAINING_EPISODES_PER_EPOCH_BATCH_AGENT,
        "Number of clients": N_CLIENTS,
        "Size of data per client": special_client_size,
        "FL Rounds": FED_ROUNDS,
        "Extra Information": "Federated learning."
    }
)

print("# Initialize the model.")
# Initialize the model.
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.resnet18 = models.resnet18(pretrained=True)
        for param in self.resnet18.parameters():
            param.requires_grad = False

        # Modify the layers to handle smaller input sizes.
        self.resnet18.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet18.maxpool = nn.Identity()  # Remove the max pooling layer.

        num_ftrs = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(num_ftrs, 10)

    def forward(self, x):
        x = x.reshape(-1, 3, 32, 32)
        return self.resnet18(x)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
classifier = CNNClassifier()
classifiers = [CNNClassifier() for i in range(N_CLIENTS)]
classifier.to(device)

print("# Initiliaze DQN.")
# Initiliaze DQN.
batch_agents = [DQN(
                observation_length=len(dataset.state_data),
                learning_rate=LEARNING_RATE,
                batch_size=BATCH_SIZE,
                target_copy_factor=TARGET_COPY_FACTOR,
                bias_average=BIAS_INITIALIZATION) for i in range(N_CLIENTS)]

print("# Define the loss function and optimizer.")
# Define the loss function and optimizer.
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(classifier.parameters(), lr=0.01)

TARGET_PRECISION = 0.0


torch.cuda.empty_cache()  # Clear unused memory after each episode.

samples_per_client = []
for set in datasets:
    print(f'Subsets lens: {len(set)}')
    samples_per_client.append(len(set))

# Check if model state_dict returns the expected dictionary
model_example = classifiers[0].state_dict()
print(model_example)  # Should print the dictionary of model parameters
print(type(model_example))  # Should print <class 'collections.OrderedDict'>

# Check the sample list
print(samples_per_client)  # Should print a list of integers


print("central_server = CentralFed(models[0].state_dict(), samples_per_client)")
central_server = CentralFed(classifiers[0].state_dict(), samples_per_client)
print("optimizers = [torch.optim.Adam(classifiers[i].parameters()) for i in range(N_CLIENTS)]")
optimizers = [torch.optim.Adam(classifiers[i].parameters()) for i in range(N_CLIENTS)]
print("criterions = []")
criterions = [nn.CrossEntropyLoss() for i in range(N_CLIENTS)]

print("for round in range(FED_ROUNDS):")
for round in range(FED_ROUNDS):
    print("round", round)
    print("    for client in range(N_CLIENTS):")
    model_classifiers = []
    for client in range(N_CLIENTS):
        print("client", client)

        client_data = torch.utils.data.Subset(trainset, client_indices[client])
        first_client_indices = client_indices[client]
        subset_data_first_client = torch.tensor(trainset.data[first_client_indices])
        subset_labels_first_client = torch.tensor([trainset.targets[i] for i in first_client_indices])
        root_dir = './data_client_'+str(client+1)
        dataset = CIFAR10Dataset(root_dir= root_dir,length_of_client_data=len(client_data), data = subset_data_first_client, labels = subset_labels_first_client)

        batch_env = LalEnvFirstAccuracy(dataset, classifier, epochs=CLASSIFIER_NUMBER_OF_EPOCHS, classifier_batch_size=CLASSIFIER_BATCH_SIZE, target_precision=TARGET_PRECISION)
        replay_buffer = ReplayBuffer(buffer_size=REPLAY_BUFFER_SIZE, prior_exp=PRIOROTIZED_REPLAY_EXPONENT)

        model = classifiers[client]
        dataset = datasets[client]
        optimizer = optimizers[client]
        criterion = criterions[client]

        print("# Initialize the variables.")
        # Initialize the variables.
        episode_durations = []
        episode_scores = []
        episode_number = 1
        episode_losses = []
        episode_precisions = []
        batches = []

        model.to(device)

        print("# Warm start procedure.")
        # Warm start procedure.
        for _ in range(WARM_START_EPISODES_BATCH_AGENT):

            print("Episode {}.".format(episode_number))

            state, next_action, indicies_unknown, reward = batch_env.reset(code_state="Warm-Start", target_precision=TARGET_PRECISION, target_budget=1.0)
            done = False
            episode_duration = CLASSIFIER_NUMBER_OF_CLASSES

            while not done:
                if batch_env.n_actions==1:
                    batch = batch_env.n_actions
                else:
                    batch = torch.randint(1, batch_env.n_actions + 1, (1,)).item()

                batches.append(batch)
                input_numbers = range(0, batch_env.n_actions)
                batch_actions_indices = torch.tensor(np.random.choice(input_numbers, batch, replace=False))
                action = batch
                next_state, next_action, indicies_unknown, reward, done = batch_env.step(batch_actions_indices)

                if next_action == []:
                    next_action.append(np.array([0]))

                print("# Store the transition in the replay buffer.")
                # Store the transition in the replay buffer.
                replay_buffer.store_transition(state, action, reward, next_state, next_action, done)

                print("# Get ready for the next step.")
                # Get ready for the next step.
                state = next_state
                episode_duration += batch

            print("# Calculate the final accuracy and precision of the episode.")
            # Calculate the final accuracy and precision of the episode.
            episode_final_acc = batch_env.return_episode_qualities()
            episode_scores.append(episode_final_acc[-1])
            episode_final_precision = batch_env.return_episode_precisions()
            episode_precisions.append(episode_final_precision[-1])
            episode_durations.append(episode_duration)
            episode_number += 1

            wandb.log({"Warm-start | Precision | Round: " + str(round) + " | Client: " + str(client): episode_final_precision[-1], "Warm-start | Budget | Round: " + str(round) + " | Client: " + str(client): (episode_durations[-1]/len(dataset.warm_start_data))*100})

            torch.cuda.empty_cache()  # Clear unused memory after each episode.

        print("# Compute the average episode duration of episodes generated during the warm start procedure.")
        # Compute the average episode duration of episodes generated during the warm start procedure.
        av_episode_duration = np.mean(episode_durations)
        BIAS_INITIALIZATION = - av_episode_duration / 2

        print("# Convert the list to a PyTorch tensor.")
        # Convert the list to a PyTorch tensor.
        episode_precisions = torch.tensor(episode_precisions)
        max_precision = torch.max(episode_precisions)

        warm_start_batches = []
        i=0
        for precision in episode_precisions:
            if precision >= max(episode_precisions):
                warm_start_batches.append(episode_durations[i])
            i+=1
        TARGET_BUDGET = min(warm_start_batches)/(len(dataset.warm_start_data))
        print("Target budget is {}.".format(TARGET_BUDGET))
        TARGET_PRECISION = max(episode_precisions)
        print("Target precision is {}.".format(TARGET_PRECISION))

        for update in range(NN_UPDATES_PER_EPOCHS_BATCH_AGENT):
            print("Update:", update+1)
            minibatch = replay_buffer.sample_minibatch(BATCH_SIZE)
            td_error = batch_agents[client].train(minibatch)
            replay_buffer.update_td_errors(td_error, minibatch.indices)
            torch.cuda.empty_cache()  # Clear unused memory after each update.

        print("BatchAgent training.")
        # BatchAgent training.

        print("# Initialize the agent.")
        # Initialize the agent.
        agent_epoch_durations = []
        agent_epoch_scores = []
        agent_epoch_precisions = []

        print("for epoch in range(TRAINING_EPOCHS_BATCH_AGENT):")
        for epoch in range(TRAINING_EPOCHS_BATCH_AGENT):
            
            print("Training epoch {}.".format(epoch + 1))

            # Simulate training episodes.
            agent_episode_durations = []
            agent_episode_scores = []
            agent_episode_precisions = []

            for training_episode in range(TRAINING_EPISODES_PER_EPOCH_BATCH_AGENT):

                print("- Training episode {}.".format(training_episode + 1))

                print("# Reset the environment to start a new episode.")
                # Reset the environment to start a new episode.
                state, action_batch, action_unlabeled_data, reward = batch_env.reset(code_state="Agent", target_precision=TARGET_PRECISION, target_budget=TARGET_BUDGET)
                done = False
                episode_duration = CLASSIFIER_NUMBER_OF_CLASSES
                first_batch = True

                while not done:
                    if first_batch:
                        next_batch = action_batch
                        next_unlabeled_data = action_unlabeled_data
                        first_batch = False
                    else:
                        next_batch = next_action_batch_size
                        next_unlabeled_data = next_action_unlabeled_data

                    selected_batch, selected_indices = batch_agents[client].get_action(dataset=dataset, model=classifier, state=state,
                                                                              next_action_batch=next_batch,
                                                                              next_action_unlabeled_data=next_unlabeled_data)
                    next_state, next_action_batch_size, next_action_unlabeled_data, reward, done = batch_env.step(
                        selected_indices)
                    if next_action_batch_size == []:
                        next_action_batch_size.append(np.array([0]))

                    replay_buffer.store_transition(state, selected_batch, reward, next_state, next_action_batch_size, done)

                    print("# Change the state of the environment.")
                    # Change the state of the environment.
                    state = torch.tensor(next_state, dtype=torch.float32).to(device)
                    episode_duration += selected_batch

                print("\n")

                agent_episode_final_acc = batch_env.return_episode_qualities()
                agent_episode_scores.append(agent_episode_final_acc[-1])
                agent_episode_final_precision = batch_env.return_episode_precisions()
                agent_episode_precisions.append(agent_episode_final_precision[-1])
                agent_episode_durations.append(episode_duration)

            maximum_epoch_precision = max(agent_episode_precisions)
            minimum_batches_for_the_maximum_epoch_precision = []
            accuracy_for_the_maximum_epoch_precision = []
            for i in range(len(agent_episode_precisions)):
                if agent_episode_precisions[i] == maximum_epoch_precision:
                    minimum_batches_for_the_maximum_epoch_precision.append(agent_episode_durations[i])
                    accuracy_for_the_maximum_epoch_precision.append(agent_episode_scores[i])
            agent_epoch_precisions.append(maximum_epoch_precision)
            agent_epoch_scores.append(accuracy_for_the_maximum_epoch_precision)
            agent_epoch_durations.append(min(minimum_batches_for_the_maximum_epoch_precision))

            wandb.log({"BatchAgent | Precision | Round: " + str(round) + " | Client: " + str(client): agent_epoch_precisions[-1], "BatchAgent | Budget | Round: " + str(round) + " | Client: " + str(client): (agent_epoch_durations[-1]/len(dataset.agent_data))*100})

            torch.cuda.empty_cache()  # Clear unused memory after each episode.

            print("# Neural networks updates.")
            # Neural networks updates.
            for update in range(NN_UPDATES_PER_EPOCHS_BATCH_AGENT):
                minibatch = replay_buffer.sample_minibatch(BATCH_SIZE)
                td_error = batch_agents[client].train(minibatch)
                replay_buffer.update_td_errors(td_error, minibatch.indices)
                torch.cuda.empty_cache()  # Clear unused memory after each update.

        model.to('cpu')
        model_classifiers.append(model)

    og_models = [copy.deepcopy(model) for model in model_classifiers]
    models_params = central_server.get_local_parameters(model_classifiers)
    aggregated_model = central_server._fed_avg(models_params)
    _ = central_server._update_local_models(aggregated_model, model_classifiers)
