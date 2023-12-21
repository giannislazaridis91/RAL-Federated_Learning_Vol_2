import numpy as np
from scipy.special import entr

# This file contains a set of functions that are needed for testing RL agents and other strategies

def check_performance(all_scores):
    """This function computes the statistics on the duration of the episodes.
    
    Given quality scores for iterations of multiple episodes, 
    compute the average duration, standard deviation on it, 
    median duration and the maximum duration. The statistics
    are printed out.
    
    Args:
        all_scores: A list of lists of floats where 
            all_scores[0] contains accuracies of episode 0
            len(all_scores) is the number of episodes andz
            len(all_scores[0]) is a list of scores of the 0th episode.
    Returns:
        all_scores: The same as input.
        all_durations: A list of durations of all the episodes.
    """
    all_durations = []
    for score in all_scores:
        all_durations.append(len(score))
    all_durations = np.array(all_durations)
    print('Mean is ', np.mean(all_durations), ' and std is ', np.std(all_durations), '!')
    print('Median is ', np.median(np.array(all_durations)), '!')
    print('Maximum is ', max(all_durations), '!')
    return all_scores, all_durations

# Functions for various AL strategies
def policy_random(n_actions):
    """Random sampling selects a datapoint at random.
    
    Args:
        n_actions: Number of available for labelling datapoints.
    Returns:
        action: An action to be taken in the environment: the index of 
            the selected datapoint."""
    action = np.random.randint(0, n_actions)
    return action

def policy_uncertainty(next_action_prob):
    """Select an action according to uncertainty sampling strategy.
    
    Args:
        next_action_prob: A numpy.ndarray of size 1 x # datapoints
            available for labelling. Contains the probability to belong 
            to one of the classes for each of the available datapoints.
    Returns:
        action: An action to be taken in the environment: the index of 
            the selected datapoint.
    """
    # compute the distance to the boundary
    criterion = abs(next_action_prob-0.5)
    # select one of the datapoints that is the closest to the boundary
    max_action = np.random.choice(np.where(criterion == criterion.min())[0])
    action = max_action
    return action

def policy_rl(agent, state, next_action_state, batch):
    """Select an action accourding to a RL agent.
    
    Args:
        agent: An object of DQN class.
        state: A numpy.ndarray characterizing the current classifier 
            The size is number of datasamples in dataset.state_data
        next_action_state: A numpy.ndarray 
            of size #features characterising actions (currently, 3) x #unlabelled datapoints 
            where each column corresponds to the vector characterizing each possible action.
    Returns:
        action: An action to be taken in the environment: the index of 
            the selected datapoint.
    """
    action = agent.get_action(state, next_action_state, batch)
    return action
