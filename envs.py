import numpy as np
from sklearn.base import clone
import collections
from sklearn.ensemble import RandomForestClassifier
import random



class LalEnv(object):

    """
    The base class for the LAL environment.

    Following the conventions of OpenAI gym, this class implements the environment which simulates labeling of a given 
    annotated dataset. The classes differ by the way how the reward is computed and when the terminal state is reached.
    It implements the environment that simulates labeling of a given annotated dataset. 

    Attributes:
        dataset:                An object of class Dataset.
        model:                  A classifier from sklearn. Should implement fit, predict and predict_proba.
        model_rf:               A random forest classifier that was fit to the same data as the data used for the model.
        quality_method:         A function that computes the quality of the prediction. For example, can be metrics.accuracy_score or metrics.f1_score.                
        n_classes:              An integer indicating the number of classes in a dataset. Typically 2.
        episode_qualities:      A list of floats with the errors of classifiers at various steps.
        n_actions:              An integer indicating the possible number of actions (the number of remaining unlabeled points).
        indices_known:          A list of indices of datapoints whose labels can be used for training.
        indices_unknown:        A list of indices of datapoint whose labels cannot be used for training yet.
        rewards_bank:           A list of floats with the rewards after each iteration in an episode.
    """
    


    def __init__(self, dataset, model, quality_method):

        # Inits environment with attributes: dataset, model, quality function and other attributes.

        self.dataset = dataset
        self.model = model
        self.quality_method = quality_method

        # Compute the number of classes as a number of unique labels in train dataset:
        self.n_classes = np.size(np.unique(self.dataset.train_labels))

        # Initialize a list where quality at each iteration will be written:
        self.episode_qualities = []

        # Set first element as 0 to be able to compute the reward later.
        self.episode_qualities.append(0)

        # Rewards bank to store the rewards.
        self.rewards_bank = []
    


    def for_lal(self):

        # Function that is used to compute features for lal-regr.
        # Fits RF classifier to the data.

        known_data = self.dataset.train_data[self.indices_known,:]
        known_labels = self.dataset.train_labels[self.indices_known]
        known_labels = np.ravel(known_labels)
        self.model_rf = RandomForestClassifier(50, oob_score=True, n_jobs=1)
        self.model_rf.fit(known_data, known_labels)


    
    def reset(self, n_start=6):

        """
        Resets the environment.
        
        1) The dataset is regenerated according to its method.
        2) n_start datapoints are selected, at least one datapoint from each class is included.
        3) The model is trained on the initial dataset and the corresponding state of the problem is computed.

        Args:
            n_start:            An integer indicating the size of the annotated set at the beginning. The initial batch size.
            
        Returns:
            classifier_state:   A numpy.ndarray characterizing the current classifier of size of number of features for the state,
                                in this case it is the size of the number of data samples in dataset.state_data.
            next_action_state:  A numpy.ndarray of size #features characterizing actions (currently, 3) x #unlabeled datapoints
                                where each column corresponds to the vector characterizing each possible action.
        """

        # SAMPLE INITIAL DATAPOINTS.
        self.dataset.regenerate()
        self.episode_qualities = []
        self.episode_qualities.append(0)
        self.rewards_bank = []
        self.rewards_bank.append(0)

        # To train an initial classifier we need at least self.n_classes samples.
        if n_start < self.n_classes:
            print('n_start', n_start, ' number of points is less than the number of classes', self.n_classes, ', so we change it.')
            n_start = self.n_classes

        # Sample n_start datapoints.
        self.indices_known = []
        self.indices_unknown = []
        for i in np.unique(self.dataset.train_labels):
            # First get 1 point from each class.
            cl = np.nonzero(self.dataset.train_labels==i)[0]
            # Insure that we select random datapoints.
            indices = np.random.permutation(cl)
            self.indices_known.append(indices[0])
            self.indices_unknown.extend(indices[1:])
        self.indices_known = np.array(self.indices_known)
        self.indices_unknown = np.array(self.indices_unknown)

        # The self.indices_unknown now contains first all points of class_1, then all points of class_2 etc.
        # So, we permute them.
        self.indices_unknown = np.random.permutation(self.indices_unknown)

        # Then, sample the rest of the datapoints at random.
        if n_start > self.n_classes:
            self.indices_known = np.concatenate(([self.indices_known, self.indices_unknown[0:n_start-self.n_classes]]))
            self.indices_unknown = self.indices_unknown[n_start-self.n_classes:]
            
        # BUILD AN INITIAL MODEL.

        # Get the data corresponding to the selected indices.
        known_data = self.dataset.train_data[self.indices_known,:]
        known_labels = self.dataset.train_labels[self.indices_known]
        unknown_data = self.dataset.train_data[self.indices_unknown,:]
        unknown_labels = self.dataset.train_labels[self.indices_unknown]

        # Train a model using data corresponding to indices_known:
        known_labels = np.ravel(known_labels)
        self.model.fit(known_data, known_labels)

        # Compute the quality score:
        test_prediction = self.model.predict(self.dataset.test_data)
        new_score = self.quality_method(self.dataset.test_labels, test_prediction)
        self.episode_qualities.append(new_score) 

        # Get the features categorizing the state.     
        classifier_state, next_action_state = self._get_state()
        self.n_actions = np.size(self.indices_unknown)    

        return classifier_state, next_action_state
        


    def step(self, action):

        """
        Make a step in the environment.

        Follow the action, in this environment it means labeling a datapoint 
        at position 'action' in indices_unknown.
        
        Args:
            action:             An integer indicating the position of a datapoint to label.
            
        Returns:
            classifier_state:   A numpy.ndarray of size # features characterising state = # datasamples in dataset.state_data
                                that characterizes the current classifier.
            next_action_state:  A numpy.ndarray of size # features characterising actions (currently, 3) x # unlabeled datapoint,
                                where each column corresponds to the vector characterizing each possible action.
            reward:             A float with the reward after adding a new datapoint.
            done:               A boolean indicator if the episode is terminated.
        """

        # Action indicates the position of a datapoint in self.indices_unknown 
        # that we want to sample in unknown_data
        # The index in train_data should be retrieved 
        selection_absolute = self.indices_unknown[action]

        # Label a datapoint: add its index to known samples and remove from unknown.
        self.indices_known = np.concatenate((self.indices_known, selection_absolute))    
        self.indices_unknown = np.delete(self.indices_unknown, action)    

        # Train a model with new labeled data:
        known_data = self.dataset.train_data[self.indices_known,:]
        known_labels = self.dataset.train_labels[self.indices_known]
        known_labels = np.ravel(known_labels)
        self.model.fit(known_data, known_labels)

        # Get a new state.
        classifier_state, next_action_state = self._get_state() 

        # Update the number of available actions:
        self.n_actions = np.size(self.indices_unknown)

        # Compute the quality of the current classifier:
        test_prediction = self.model.predict(self.dataset.test_data)
        new_score = self.quality_method(self.dataset.test_labels, test_prediction)
        self.episode_qualities.append(new_score)

        # Compute the reward.
        reward = self._compute_reward()

        # Check if this episode terminated.
        done = self._compute_is_terminal()
        if done:
            print("Final achieved ACC is {}.".format(self.episode_qualities[-1]))

        return classifier_state, next_action_state, reward, done
      


    def _get_state(self):

        """
        Private function for computing the state depending on the classifier and next available actions.
        
        This function computes:
        1) classifier_state that characterizes the current state of the classifier and it is computed as a function of predictions on the hold-out dataset 
        2) next_action_state that characterizes all possible actions (unlabeled datapoints) that can be taken at the next step.
        
        Returns:
            classifier_state:   A numpy.ndarray of size of number of datapoints in dataset.state_data 
                                characterizing the current classifier and, thus, the state of the environment.
            next_action_state:  A numpy.ndarray of size #features characterizing actions (currently, 3) x #unlabeled datapoints 
                                where each column corresponds to the vector characterizing each possible action.
        """

        # COMPUTE CLASSIFIER_STATE.
        predictions = self.model.predict_proba(self.dataset.state_data)[:,0]
        predictions = np.array(predictions)
        idx = np.argsort(predictions)

        # The state representation is the *sorted* list of scores.
        classifier_state = predictions[idx]
        
        # COMPUTE ACTION_STATE.
        unknown_data = self.dataset.train_data[self.indices_unknown,:]

        # Prediction (score) of classifier on each unlabeled sample:
        a1 = self.model.predict_proba(unknown_data)[:,0]
        # Average distance to every unlabeled datapoint:
        a2 = np.mean(self.dataset.distances[self.indices_unknown,:][:,self.indices_unknown],axis=0)
        # Average distance to every labeled data point.
        a3 = np.mean(self.dataset.distances[self.indices_known,:][:,self.indices_unknown],axis=0)
        # Compute the next_action_state.
        next_action_state = np.concatenate(([a1], [a2], [a3]), axis=0)

        return classifier_state, next_action_state
    


    def _compute_reward(self):

        """
        Private function to compute the reward.
        
        Default function always returns 0.
        Every sub-class should implement its own reward function.
        
        Returns:
            reward: A float reward.
        """

        reward = 0.0
        
        return reward
    


    def _compute_is_terminal(self):

        """
        Private function to compute if the episode has reached the terminal state.
        
        By default episode terminates when all the data is labeled.
        Every sub-class should implement its own episode termination function.
        
        Returns:
            done: A boolean that indicates if the episode is finished.
        """

        # The self.n_actions contains a number of unlabeled datapoints that are left.
        if self.n_actions==2:
            print('We ran out of samples!')
            done = True
        else:
            done = False
   
        return done
        
    
    


class LalEnvFirstAccuracy(LalEnv): 

    """
    The LAL environment class where the episode lasts until a classifier reaches a predifined quality.

    This class inherits from LalEnv. 
    The reward is -1 at every step. 
    The terminal state is reached 
    when the predefined classificarion 
    quality is reached. Classification 
    quality is defined as a proportion 
    of the final quality (that is obtained 
    when all data is labelled).

    Attributes:
        tolerance_level: A float indicating what proportion of the maximum reachable score 
                         should be attained in order to terminate the episode.
    """
    


    def __init__(self, dataset, model, quality_method):

        # Inits environment with its normal attributes.
        LalEnv.__init__(self, dataset, model, quality_method)
    

        
    def reset(self, n_start=6):

        """
        Resets the environment.
        
        Args:
            n_start: An integer indicating the size of annotated set at the beginning.
            
        Returns:
            The same as the parent class.
        """

        classifier_state, next_action_state = LalEnv.reset(self, n_start=n_start)
        current_reward = self._compute_reward()

        # Store the current rewatd.
        self.rewards_bank.append(current_reward)
        
        return classifier_state, next_action_state, current_reward
       


    def _compute_reward(self):

        # Find the reward as new_score - previous_score.
        new_score = self.episode_qualities[-1]
        previous_score = self.episode_qualities[-2]
        reward = new_score - previous_score
        self.rewards_bank.append(reward)
        return reward
    
    

    def _compute_is_terminal(self):

        """
        Computes if the episode has reached the terminal state.
        
        Returns:
            done: A boolean that indicates if the episode is finished.
        """

        # By default the episode will terminate when all samples are labelled.
        done = LalEnv._compute_is_terminal(self)
 
        # If the last three rewards are declining, then terminate the episode.
        if len(self.rewards_bank) >= 4:
            if self.rewards_bank[-1] < self.rewards_bank[-2] and self.rewards_bank[-2] < self.rewards_bank[-3] and self.rewards_bank[-3] < self.rewards_bank[-4]:
                done = True
                return done
        return done