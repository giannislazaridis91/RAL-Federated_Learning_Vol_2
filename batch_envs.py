import numpy as np
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from sklearn.metrics import precision_score

class BatchAgentLalEnv(object):

    def __init__(self, dataset, epochs, classifier_batch_size):
        
        # Initialize the environment with attributes: dataset, model, quality function and other attributes.
        self.dataset = dataset
        self.model = load_model('./classifiers/classifier_batch_agent.h5')
        self.epochs = epochs
        self.classifier_batch_size = classifier_batch_size  
        self.number_of_classes = np.size(np.unique(self.dataset.train_labels)) # # The number of classes as a number of unique labels in the train dataset.    
        self.episode_qualities = [] # A list where testing quality at each iteration will be written.
        self.episode_losses= [] # A list where loss for testing at each iteration will be written:
        self.episode_training_qualities = [] # A list where training quality at each iteration will be written:
        self.episode_training_losses = [] # A list where training loss at each iteration will be written.
        self.rewards_bank = [] # Rewards bank to store the rewards.
        self.batch_bank = [] # Batch bank to store the batch size.
        self.precision_bank = [] # Precision bank to store the precision per class.
    
    def reset(self, number_of_first_samples=10):
        
        self.model = load_model('./classifiers/classifier_batch_agent.h5')
        
        # Sample initial data points.
        self.dataset.regenerate()
        self.episode_qualities.append(0)
        self.rewards_bank.append(0)
        self.batch_bank.append(0.1)
        self.precision_bank.append(0)
        self.episode_losses.append(2)
        
        # To train an initial classifier we need at least self.number_of_classes samples.
        if number_of_first_samples < self.number_of_classes:
            print('number_of_first_samples', number_of_first_samples, ' number of points is less than the number of classes', self.number_of_classes, ', so we change it.')
            number_of_first_samples = self.number_of_classes
        
        # Sample number_of_first_samples data points.
        self.indices_known = []
        self.indices_unknown = []
        
        for i in np.unique(self.dataset.train_labels):
            
            # First get 1 point from each class.
            cl = np.nonzero(self.dataset.train_labels==i)[0]
            
            # Ensure that we select random data points.
            indices = np.random.permutation(cl)
            self.indices_known.append(indices[0])
            self.indices_unknown.extend(indices[1:])
        self.indices_known = np.array(self.indices_known)
        self.indices_unknown = np.array(self.indices_unknown)
        
        # The self.indices_unknown now contains first all points of class_1, then all points of class_2 etc.
        # So, we permute them.
        self.indices_unknown = np.random.permutation(self.indices_unknown)
        
        # Then, sample the rest of the data points at random.
        if number_of_first_samples > self.number_of_classes:
            self.indices_known = np.concatenate(([self.indices_known, self.indices_unknown[0:number_of_first_samples-self.number_of_classes]]))
            self.indices_unknown = self.indices_unknown[number_of_first_samples-self.number_of_classes:]
        
        # BUILD AN INITIAL MODEL.
        # Get the data corresponding to the selected indices.
        known_data = self.dataset.train_data[self.indices_known,:]
        
        #print("known_data", known_data)
        known_labels = self.dataset.train_labels[self.indices_known]
        
        #print("known_labels", known_labels)
        unknown_data = self.dataset.train_data[self.indices_unknown,:]
        unknown_labels = self.dataset.train_labels[self.indices_unknown]
        known_labels_one_hot_encoding = keras.utils.to_categorical(known_labels, num_classes = self.dataset.number_of_classes)
        
        # Train a model using data corresponding to indices_known:
        early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
        checkpoint = ModelCheckpoint('weights.h5', monitor='val_accuracy', verbose=0, save_best_only=True, mode='max')
        callbacks = [early_stopping, checkpoint]
        self.model._ckpt_saved_epoch = 0
        history = self.model.fit(known_data, known_labels_one_hot_encoding, batch_size=self.classifier_batch_size, epochs=self.epochs, verbose=0,
                                 validation_data=(self.dataset.test_data, self.dataset.test_labels_one_hot_encoding), callbacks=callbacks)
        # Testing accuracy.
        new_score = history.history['val_accuracy'][-1]
        self.episode_qualities.append(new_score)
        
        # Compute the precision:
        predictions = self.model.predict(self.dataset.test_data)
        predicted_labels = np.argmax(predictions, axis=1)
        true_labels = np.argmax(self.dataset.test_labels_one_hot_encoding, axis=1)
        precision_scores = precision_score(true_labels, predicted_labels, average='weighted', zero_division=0)
        self.precision_bank.append(precision_scores)

        # Batch.
        self.batch_bank.append(number_of_first_samples/len(self.dataset.train_data))

        # Testing loss.
        self.episode_losses.append(history.history['val_loss'][-1])

        # Training accuracy.
        self.episode_training_qualities.append(history.history['accuracy'][-1])
        
        # Training loss.
        self.episode_training_losses.append(history.history['loss'][-1])

        # Get the features categorizing the state.
        state, next_action = self._get_state()
        self.n_actions = np.size(self.indices_unknown)
        self.model.save('./classifiers/classifier_batch_agent.h5')
        return state, next_action
        
    def step(self, batch=0, batch_actions_indices=[], isBatchAgent=False):
        
        self.isBatchAgent = isBatchAgent
        
        if self.isBatchAgent:
            self.model = load_model('./classifiers/classifier_batch_agent.h5')
            train_predictions = self.model.predict(self.dataset.train_data[self.indices_unknown,:])
            uncertainty_scores = np.min(train_predictions, axis=1)
            sorted_indices = np.argsort(uncertainty_scores)[:batch]
            
            # Label a datapoint: add its index to known samples and remove from unknown.
            self.indices_known = np.concatenate((self.indices_known, sorted_indices))
            self.indices_unknown = np.delete(self.indices_unknown, sorted_indices)
            
            # Train a model with new labeled data:
            known_data = self.dataset.train_data[self.indices_known,:]
            known_labels = self.dataset.train_labels[self.indices_known]
            known_labels_one_hot_encoding = keras.utils.to_categorical(known_labels, num_classes = self.dataset.number_of_classes)
        
        else:
            self.model = load_model('./classifiers/classifier_batch_agent.h5')
            
            # The batch_actions_indices value indicates the positions
            # of the batch of data points in self.indices_unknown that we want to sample in unknown_data.
            # The index in train_data should be retrieved.
            selection_absolute = self.indices_unknown[batch_actions_indices]
            
            # Label a datapoint: add its index to known samples and remove from unknown.
            self.indices_known = np.concatenate((self.indices_known, selection_absolute))
            self.indices_unknown = np.delete(self.indices_unknown, batch_actions_indices)
            
            # Train a model with new labeled data:
            known_data = self.dataset.train_data[self.indices_known,:]
            known_labels = self.dataset.train_labels[self.indices_known]
            known_labels_one_hot_encoding = keras.utils.to_categorical(known_labels, num_classes = self.dataset.number_of_classes)
        
        early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
        checkpoint = ModelCheckpoint('weights.h5', monitor='val_accuracy', verbose=0, save_best_only=True, mode='max')
        callbacks = [early_stopping, checkpoint]
        self.model._ckpt_saved_epoch = 0
        history = self.model.fit(known_data, known_labels_one_hot_encoding, batch_size=self.classifier_batch_size, epochs=self.epochs, verbose=0,
                                 validation_data=(self.dataset.test_data, self.dataset.test_labels_one_hot_encoding), callbacks=callbacks)
        # Get a new state.
        state, next_action = self._get_state()
        
        # Update the number of available actions.
        self.n_actions = np.size(self.indices_unknown)
        
        # Compute the quality of the current classifier.
        new_score = history.history['val_accuracy'][-1] # Testing accuracy.
        self.episode_qualities.append(new_score)
        
        # Compute the precision:
        predictions = self.model.predict(self.dataset.test_data)
        predicted_labels = np.argmax(predictions, axis=1)
        true_labels = np.argmax(self.dataset.test_labels_one_hot_encoding, axis=1)
        precision_scores = precision_score(true_labels, predicted_labels, average='weighted', zero_division=0)
        self.precision_bank.append(precision_scores)
        self.batch_bank.append(len(batch_actions_indices)/len(self.dataset.train_data)) # Batch.
        self.episode_losses.append(history.history['val_loss'][-1]) # Testing loss.
        self.episode_training_qualities.append(history.history['accuracy'][-1]) # Training accuracy.
        self.episode_training_losses.append(history.history['loss'][-1]) # Training loss.
        
        # Compute the reward.
        reward = self._compute_reward()
        
        # Check if this episode terminated.
        done = self._compute_is_terminal()
        self.model.save('./classifiers/classifier_batch_agent.h5')
        return state, next_action, reward, done
      
    def _get_state(self):
        
        # Compute current state. Use margin score.
        predictions = self.model.predict(self.dataset.state_data)
        predictions = np.array(predictions)
        part = np.partition(-predictions, 1, axis=1)
        margin = - part[:, 0] + part[:, 1]
        idx = np.argsort(margin)

        # The state representation is the "sorted" list of margin scores.
        state = margin[idx]

        # Compute next_action.
        unknown_data = self.dataset.train_data[self.indices_unknown,:]
        next_action = []
        
        for i in range(1, len(unknown_data)+1):
            next_action.append(np.array([i]))
            
        return state, next_action
    
    def _compute_reward(self):

        reward = 0.0
        return reward
    
    def _compute_is_terminal(self):
        # The self.n_actions contains a number of unlabeled data points that are left.
        if self.n_actions==0:
            print('We ran out of samples!')
            done = True
        else:
            done = False
        return done

class BatchAgentLalEnvFirstAccuracy(BatchAgentLalEnv): 

    def __init__(self, dataset, epochs, classifier_batch_size):
        # Initialize the environment with its normal attributes.
        BatchAgentLalEnv.__init__(self, dataset, epochs, classifier_batch_size)
    
    def reset(self, number_of_first_samples=10):
        
        state, next_action = BatchAgentLalEnv.reset(self, number_of_first_samples=number_of_first_samples)
        current_reward = self._compute_reward()

        # Store the current rewatd.
        self.rewards_bank.append(current_reward)
        return state, next_action, current_reward
       
    def _compute_reward(self):
        
        # Find the reward as new_score - previous_score.
        new_score = self.episode_qualities[-1] / self.batch_bank[-1]
        previous_score = self.episode_qualities[-2] / self.batch_bank[-2]
        reward = new_score - previous_score
        self.rewards_bank.append(reward)
        return reward
    
    def _compute_is_terminal(self):

        # By default the episode will terminate when all samples are labelled.
        done = BatchAgentLalEnv._compute_is_terminal(self)
        if len(self.rewards_bank) >= 2:
            if self.rewards_bank[-1] < self.rewards_bank[-2]:
                done = True
                return done
        return done
