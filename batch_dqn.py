import tensorflow as tf
from tensorflow.compat.v1 import disable_eager_execution, variable_scope, placeholder
from tensorflow.keras.optimizers import Adam
import numpy as np
import os
from batch_agent_estimator import BatchAgentEstimator
if not tf.executing_eagerly():
    tf.compat.v1.enable_eager_execution()
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

class BatchAgentDQN:

    def __init__(self, experiment_dir, observation_length, learning_rate, batch_size, target_copy_factor, bias_average, session):

        self.session = session
        self.i_train = 0
        self.i_actions_taken = 0
        self._initialized = False
        
        # TARGET ESTIMATOR
        with variable_scope("target_dqn"):

            self.target_estimator = BatchAgentEstimator(observation_length, is_target_dqn=False, var_scope_name="target_dqn", bias_average=bias_average)

        # ESTIMATOR
        with variable_scope("dqn"):

            self.estimator = BatchAgentEstimator(observation_length, is_target_dqn=True, var_scope_name="dqn", bias_average=bias_average)
            
            # Placeholders for transactions from the replay buffer.
            self._reward_placeholder = placeholder(dtype=tf.float32, shape=(batch_size))
            self._terminal_placeholder = placeholder(dtype=tf.bool, shape=(batch_size))
            
            # Placeholder for the max of the next prediction by target estimator.
            self._next_best_prediction = placeholder(dtype=tf.float32, shape=(batch_size))
            ones = tf.ones(shape=(batch_size))
            zeros = tf.zeros(shape=(batch_size))
            
            # Contains 1 where not terminal, 0 where terminal.
            # Dimensionality: (batch_size x 1).
            terminal_mask = tf.where(self._terminal_placeholder, zeros, ones)
            
            # For samples that are not terminal, masked_target_predictions contains 
            # max next step action value predictions. 
            # dimensionality (batch_size x 1)
            masked_target_predictions = self._next_best_prediction * terminal_mask
            
            # Target values for actions taken (actions_taken_targets)
            #  = r + Q_target(s', a')  , for non-terminal transitions
            #  = r                     , for terminal transitions
            # Dimensionality: (batch_size x 1).
            actions_taken_targets = self._reward_placeholder + masked_target_predictions
            actions_taken_targets = tf.reshape(actions_taken_targets, (batch_size, 1))
            
            # Define temporal difference error .
            self._td_error = actions_taken_targets - self.estimator.predictions
            
            # Loss function.
            self._loss = tf.reduce_sum(tf.square(self._td_error))
            
            # Training operation with Adam optimiser.
            # Create a tf.GradientTape() context.
            with tf.GradientTape() as tape:

                # Compute the loss.
                loss = tf.reduce_sum(tf.square(self._td_error))

            # Create the optimizer.
            opt = Adam(learning_rate)

            # Get the gradients.
            gradients = tape.gradient(loss, tf.compat.v1.get_collection('dqn'))

            # Apply the gradients.
            self._train_op = opt.apply_gradients(zip(gradients, tf.compat.v1.get_collection('dqn')))
            
            # Operation to copy parameter values (partially) to target estimator.
            copy_factor_complement = 1 - target_copy_factor
            self._copy_op = [target_var.assign(target_copy_factor * my_var + copy_factor_complement * target_var)
                            for (my_var, target_var)
                            in zip(tf.compat.v1.get_collection('dqn'), tf.compat.v1.get_collection('target_dqn'))]
    
    def _check_initialized(self):
        
        if not self._initialized:
            self.session.run(tf.compat.v1.global_variables_initializer())      
            self._initialized = True

    def get_action(self, state, next_action):
        
        # Counter of how many times this function was called.
        self.i_actions_taken += 1
        self._check_initialized()
        
        # Repeat classification_state so that we have a copy of classification state for each possible action.
        state = np.repeat([state], len(next_action), axis=0)
        
        # Predict q-values with current estimator.
        predictions = self.session.run(
            self.estimator.predictions,
            feed_dict = {self.estimator.classifier_placeholder: state, 
                         self.estimator.action_placeholder: next_action})
        
        max_action = np.random.choice(np.where(predictions == predictions.max())[0])
        return max_action
         
    def train(self, minibatch):

        # NEXT BEST Q-VALUES
        # For bootstrapping, the target for update function depends on the q-function 
        # of the best next action. So, compute max_prediction_batch that represents Q_target_estimator(s', a_best_by estimator).
        self._check_initialized()
        max_prediction_batch = []
        i = 0

        # Counter of how many times this function was called.
        self.i_train += 1

        # For every transaction in minibatch.
        for next_state in minibatch.next_state:

            # Predict q-value function value for all available actions.
            n_next_actions = len(minibatch.next_action[i])
            next_state = np.repeat([next_state], n_next_actions, axis=0)

            # Use target_estimator.
            target_predictions = self.session.run(
                [self.target_estimator.predictions],
                feed_dict = {self.target_estimator.classifier_placeholder: next_state, 
                             self.target_estimator.action_placeholder: minibatch.next_action[i]})
            
            # Use estimator.
            predictions = self.session.run(
                [self.estimator.predictions],
                feed_dict = {self.estimator.classifier_placeholder: next_state, self.estimator.action_placeholder: minibatch.next_action[i]})
            target_predictions = np.ravel(target_predictions)
            predictions = np.ravel(predictions)
            
            # Follow Double Q-learning idea of van Hasselt, Guez, and Silver 2016.
            # Select the best action according to the predictions of estimator.
            best_action_by_estimator = np.random.choice(np.where(predictions == np.amax(predictions))[0])
            
            # During the estimation of q-value of the best action, 
            # take the prediction of target estimator for the selecting action.
            max_target_prediction_i = target_predictions[best_action_by_estimator]
            max_prediction_batch.append(max_target_prediction_i)
            i += 1

        # OPTIMIZE
        # Update Q-function value estimation.
        _, _td_error = self.session.run(
            [self._train_op, self._td_error],
            feed_dict = {self.estimator.classifier_placeholder: minibatch.state,
                self.estimator.action_placeholder: [[x] for x in minibatch.action],
                self._next_best_prediction: max_prediction_batch,
                self._reward_placeholder: minibatch.reward,
                self._terminal_placeholder: minibatch.terminal,
                self.target_estimator.classifier_placeholder: minibatch.next_state,
                self.target_estimator.action_placeholder: [[x] for x in minibatch.action]})
        
        # Update target_estimator by partially copying the parameters of the estimator.
        self.session.run(self._copy_op)

        return _td_error
