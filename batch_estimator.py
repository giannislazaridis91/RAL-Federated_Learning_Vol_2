import tensorflow as tf
from tensorflow.compat.v1 import disable_eager_execution, placeholder, variable_scope
from tensorflow.keras.layers import Dense
import numpy as np

class BatchAgentEstimator:
    
    def __init__(self, classifier_state_length, is_target_dqn, var_scope_name, bias_average):
        
        disable_eager_execution()
        self.classifier_placeholder = placeholder(tf.float32, shape=[None, classifier_state_length], name="X_classifier")
        self.action_placeholder = placeholder(tf.float32, shape=[None, 1], name="X_datapoint")
        
        with variable_scope(var_scope_name):
            fc1 = Dense(10, activation='sigmoid', trainable=not is_target_dqn, name='fc1')(self.classifier_placeholder)
            fc2concat = tf.concat([fc1, self.action_placeholder], 1)
            fc3 = Dense(5, activation='sigmoid', trainable=not is_target_dqn, name='fc3')(fc2concat)
            self.predictions = Dense(1, activation=None, trainable=not is_target_dqn, name='predictions')(fc3)
