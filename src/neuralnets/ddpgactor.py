import os

import numpy as np
import tensorflow as tf
#from tensorflow.initializers import he_uniform
from tensorflow.keras.initializers import HeUniform

from src.neuralnet import NeuralNet
from src.neuralnets.varstate import VariableState
from src.picklefuncs import save_data, load_data
from src.helper_funcs import check_and_make_dir

class DDPGActorNet:
    def __init__(self, input_d, hidden_d, hidden_act, output_d, output_act, lr, lre, name, batch_size, sess):
        #create model and all necessary parts
        with tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE):
            self.input = tf.compat.v1.placeholder(tf.float32,
                                        shape=[None, input_d],
                                        name='inputs')
                                                                                                     
            self.action_gradient = tf.compat.v1.placeholder(tf.float32,
                                          shape=[None, output_d],
                                          name='gradients')
                                                                                                     
            self.is_training = tf.compat.v1.placeholder(tf.bool, name='is_training')

            dense1 = tf.compat.v1.layers.dense(self.input, units=hidden_d[0],
                                     kernel_initializer=HeUniform())

            batch1 = tf.compat.v1.layers.batch_normalization(dense1, training=self.is_training)
            layer1_activation = tf.nn.elu(batch1)
            dense2 = tf.compat.v1.layers.dense(layer1_activation, units=hidden_d[1],
                                     kernel_initializer=HeUniform())

            batch2 = tf.compat.v1.layers.batch_normalization(dense2, training=self.is_training)
            layer2_activation = tf.nn.elu(batch2)
            mu = tf.compat.v1.layers.dense(layer2_activation, units=output_d,
                            activation='tanh',
                            kernel_initializer=HeUniform())

            self.mu = mu
                                                                                                     
            self.params = tf.compat.v1.trainable_variables(scope=name)
            #print(name)

            self.unnormalized_actor_gradients = tf.gradients(self.mu, self.params, -self.action_gradient, unconnected_gradients='zero')
            
            self.actor_gradients = list(map(lambda x: tf.math.divide(x, batch_size), self.unnormalized_actor_gradients))

            # Ensure batch norm running stats are updated during training
            update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS, scope=name)
            with tf.control_dependencies(update_ops):
                self.optimize = tf.compat.v1.train.AdamOptimizer(learning_rate = lr, epsilon=lre).apply_gradients(zip(self.actor_gradients, self.params))

            #for training/saving
            self.varstate = VariableState(sess, self.params)

class DDPGActor(NeuralNet):
    def __init__(self, input_d, hidden_d, hidden_act, output_d, output_act, lr, lre, tau, learner=False, name='', batch_size=32, sess=None):
        self.lr = lr
        self.lre = lre
        self.sess = sess
        self.batch_size = batch_size
        self.name = name
        super().__init__(input_d, hidden_d, hidden_act, output_d, output_act, learner=learner)
        self.tau = tau
        self.new_w = []

        if learner:
            self.update_actor = [self.models['target'].params[i].assign(
                                 tf.multiply(self.models['online'].params[i], self.tau)
                                 + tf.multiply(self.models['target'].params[i], 1. - self.tau))
                                 for i in range(len(self.models['target'].params))]

    def create_model(self, input_d, hidden_d, hidden_act, output_d, output_act):
        return DDPGActorNet(input_d, hidden_d, hidden_act, output_d, output_act, self.lr, self.lre, self.name, self.batch_size, self.sess)

    def forward(self, x, nettype):
        return self.sess.run(self.models[nettype].mu, feed_dict={self.models[nettype].input: x,
                                                                 self.models[nettype].is_training: False})

    def backward(self, states, grads):
        self.sess.run(self.models['online'].optimize,
                      feed_dict={self.models['online'].input: states,
                                 self.models['online'].action_gradient: grads,
                                 self.models['online'].is_training: True})

    def transfer_weights(self):
        """ Transfer model weights to target model with a factor of Tau
        """
        self.sess.run(self.update_actor)

    def get_weights(self, nettype):
        #return self.sess.run(self.models[nettype].params)
        return self.models[nettype].varstate.export_variables()

    def set_weights(self, weights, nettype):
        '''
        if 'actor-12' == self.name:
            print('trying to import weights==========')
            #print(weights)
            print(len(weights))
            for w in weights:
                print(w.shape)
            print('GETTING WAITS FROM NETWORK------')
            w = self.get_weights('online')
            for _ in w:
                print(_.shape)
            #print(w)
            print(len(w))
        '''
        self.models[nettype].varstate.import_variables(weights)

    def save_weights(self, nettype, path, fname):
        check_and_make_dir(path)
        weights = self.get_weights(nettype)
        save_data(path+fname+'.p', weights)

    def load_weights(self, path):
        path += '.p'
        if os.path.exists(path):
            weights = load_data(path)
            self.set_weights(weights, 'online')
        else:
            #raise not found exceptions
            assert 0, 'Failed to load weights, supplied weight file path '+str(path)+' does not exist.'
