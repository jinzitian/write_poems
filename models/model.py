# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 13:22:59 2017

@author: Jinzitian
"""

import tensorflow as tf


class Rnn_model(object):

    def __init__(self, model, input_data, output_data, vocab_size, cell_size, num_layers):
        """
        construct rnn seq2seq model.
        :param model: model class
        :param input_data: input data placeholder
        :param output_data: output data placeholder
        :param vocab_size:
        :param cell_size:
        :param num_layers:
        :return:
        """

        batch_size, num_steps = input_data.shape.as_list()
            
        self.vector_size = cell_size * 2
        if model == 'rnn':
            cell_unit = tf.contrib.rnn.BasicRNNCell
        elif model == 'gru':
            cell_unit = tf.contrib.rnn.GRUCell
        elif model == 'lstm':
            cell_unit = tf.contrib.rnn.BasicLSTMCell
        
        cell = cell_unit(cell_size, state_is_tuple=True)
        cell = tf.contrib.rnn.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
        
        initial_state = cell.zero_state(batch_size, tf.float32)
        self.initial_state = initial_state
        
        with tf.device("/cpu:0"):
            embedding = tf.get_variable('embedding', shape = [vocab_size, self.vector_size], dtype=tf.float32)
            inputs = tf.nn.embedding_lookup(embedding, input_data)
        
        # [batch_size, ?, cell_size] = [64, ?, 128]
        outputs, last_state = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state)
        self.last_state = last_state
        output = tf.reshape(outputs, [-1, cell_size])
        
        weights = tf.Variable(tf.truncated_normal([cell_size, vocab_size]))
        bias = tf.Variable(tf.zeros(shape=[vocab_size]))
        logits = tf.nn.bias_add(tf.matmul(output, weights), bias=bias)
        # [?, vocab_size] ，softmax的操作是在axis = -1上操作，这里是在axis = 1上
        softmax_output = tf.nn.softmax(logits)
        self.predict = softmax_output
        if output_data is not None:
            # output_data must be one-hot encode
            labels = tf.one_hot(tf.reshape(output_data, [-1]), depth=vocab_size)
            loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
            average_loss = tf.reduce_mean(loss)
            train_op = tf.train.AdamOptimizer(0.01).minimize(average_loss)

            self.train_op = train_op
            self.loss = average_loss

