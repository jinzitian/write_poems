# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 13:22:59 2017

@author: Jinzitian
"""
import os
import numpy as np
import pickle

import tensorflow as tf
from models.model import Rnn_model
from train.train import FLAGS

from dict import __path__ as dict_path

def next_word(predict, vocabs):
    picklefile = open(os.path.join(dict_path[0],'word2id_dictionary.pkl'), 'rb')
    word2id = pickle.load(picklefile)
    picklefile.close()
    t = np.cumsum(predict)
    s = np.sum(predict)
    sample = int(np.searchsorted(t, np.random.rand(1) * s))
    if sample >= len(vocabs) or sample == 0:
        sample = word2id[']']
    return vocabs[sample]


def generate_poem():

    picklefile = open(os.path.join(dict_path[0],'word2id_dictionary.pkl'), 'rb')
    word2id = pickle.load(picklefile)
    picklefile.close()    
    
    picklefile = open(os.path.join(dict_path[0],'words_tuple.pkl'), 'rb')
    words_tuple = pickle.load(picklefile)
    picklefile.close()   
    
    with tf.Graph().as_default():
        #设置整个graph的初始化方式
        initializer = tf.random_uniform_initializer(-FLAGS.init_scale, FLAGS.init_scale)

        x_train = tf.placeholder(tf.int32, shape=[FLAGS.predict_batch_size, None])
        y_train = None            
        with tf.variable_scope("Model", reuse=None, initializer=initializer):
            m = Rnn_model('lstm', x_train, y_train, len(words_tuple), FLAGS.cell_size, FLAGS.num_layers)
        
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoints_dir)
        saver = tf.train.Saver()
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            saver.restore(session, ckpt.model_checkpoint_path)
            poem = ''            
            word = '['
            last_state = session.run(m.initial_state)
            while word != ']':
                predict, last_state = session.run([m.predict, m.last_state],feed_dict={x_train: np.array([[word2id[word]]]), m.initial_state: last_state})
                word = next_word(predict[0], words_tuple)
                if word != ']':
                    poem += word
                
            return poem
    
    

def generate_your_poem(header_string):
    
    picklefile = open(os.path.join(dict_path[0],'word2id_dictionary.pkl'), 'rb')
    word2id = pickle.load(picklefile)
    picklefile.close()    
    
    picklefile = open(os.path.join(dict_path[0],'words_tuple.pkl'), 'rb')
    words_tuple = pickle.load(picklefile)
    picklefile.close()    
    
    with tf.Graph().as_default():
        #设置整个graph的初始化方式
        initializer = tf.random_uniform_initializer(-FLAGS.init_scale, FLAGS.init_scale)

        x_train = tf.placeholder(tf.int32, shape=[FLAGS.predict_batch_size, None])
        y_train = None            
        with tf.variable_scope("Model", reuse=None, initializer=initializer):
            m = Rnn_model('lstm', x_train, y_train, len(words_tuple), FLAGS.cell_size, FLAGS.num_layers)
        
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoints_dir)
        saver = tf.train.Saver()
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            saver.restore(session, ckpt.model_checkpoint_path)
            
            poem = ''    
            last_state = session.run(m.initial_state)
            for head_word in header_string:               
                word = head_word
                predict, last_state = session.run([m.predict, m.last_state],feed_dict={x_train: np.array([[word2id[word]]]), m.initial_state: last_state})
                poem_piece = ''            
                while word != ']':
                    poem_piece += word
                    if word == '。':
                        break
                    word = next_word(predict[0], words_tuple)
                    predict, last_state = session.run([m.predict, m.last_state],feed_dict={x_train: np.array([[word2id[word]]]), m.initial_state: last_state})
                poem += poem_piece                   

            return poem
