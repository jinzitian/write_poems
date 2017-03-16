# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 13:22:59 2017

@author: Jinzitian
"""

import pickle
import numpy as np
import tensorflow as tf

from word2id.word2id import build_dataset
from models.model import Rnn_model

tf.app.flags.DEFINE_integer('cell_size', 128, 'cell_size')
tf.app.flags.DEFINE_string('poem_file_name', './data/poems.txt', 'data_file_name.')
tf.app.flags.DEFINE_float('learning_rate', 1.0, 'learning_rate')
tf.app.flags.DEFINE_integer('num_layers', 2, 'num_layers')
tf.app.flags.DEFINE_integer('nb_epoch', 50, 'nb_epoch')
tf.app.flags.DEFINE_integer('train_batch_size', 64, 'train_batch_size')
tf.app.flags.DEFINE_integer('predict_batch_size', 1, 'predict_batch_size')
tf.app.flags.DEFINE_float('init_scale', 0.1, 'init_scale')
tf.app.flags.DEFINE_string('checkpoints_dir', './checkpoints', 'checkpoints save path.')
tf.app.flags.DEFINE_string('model_prefix', 'Rnn_model', 'model save filename.')

FLAGS = tf.app.flags.FLAGS

def process_poems(file_name):
    # 诗集
    poems = []
    with open(file_name, "r", encoding='utf-8') as f:
        for line in f.readlines():
            try:
                title, content = line.strip().split(':')
                content = content.replace(' ', '')
                if '_' in content or '(' in content or '（' in content or '《' in content or '[' in content or \
                        '[' in content or ']' in content:
                    continue
                if len(content) < 5 or len(content) > 79:
                    continue
                content = '[' + content + ']'
                poems.append(content)
            except ValueError as e:
                pass

    #这里排序的目的是为了后面分批次的时候尽量让长度相近的诗在一个批次，方便训练
    poems = sorted(poems, key=lambda l: len(l))
    # 统计每个字出现次数
    all_words = []
    for poem in poems:
        all_words += list(poem)        
    word2id, words = build_dataset(all_words)
    poems_vector = [list(map(lambda word: word2id.get(word, 0), poem)) for poem in poems]

    return poems_vector, word2id, words


def generate_batch(batch_size, poems_vec, word_to_int):
    # 每次取64首诗进行训练
    n_chunk = len(poems_vec) // batch_size
    x_batches = []
    y_batches = []
    for i in range(n_chunk):
        start_index = i * batch_size
        end_index = (i+1) * batch_size

        batches = poems_vec[start_index:end_index]
        # 找到这个batch的所有poem中最长的poem的长度
        length = max(map(len, batches))
        # 填充一个这么大小的空batch，空的地方放空格对应的index标号
        x_data = np.full((batch_size, length), word_to_int[' '], np.int32)
        for row in range(batch_size):
            # 每一行就是一首诗，在原本的长度上把诗还原上去
            x_data[row, :len(batches[row])] = batches[row]
        y_data = np.copy(x_data)
        # y的话就是x向左边也就是前面移动一个
        y_data[:, :-1] = x_data[:, 1:]
        """
        x_data             y_data
        [6,2,4,6,9]       [2,4,6,9,9]
        [1,4,2,8,5]       [4,2,8,5,5]
        """
        x_batches.append(x_data)
        y_batches.append(y_data)
    return x_batches, y_batches



def train():

    poems_vector, word2id, words_tuple = process_poems(FLAGS.poem_file_name)
    x_batches, y_batches = generate_batch(FLAGS.train_batch_size, poems_vector, word2id)
    
    #用pickle保存word2id字典，为了后续预测数据时转换使用
    output = open('./dict/word2id_dictionary.pkl', 'wb')
    pickle.dump(word2id, output)
    output.close()
    
    output1 = open('./dict/words_tuple.pkl', 'wb')
    pickle.dump(words_tuple, output1)
    output1.close()
    
    with tf.Graph().as_default():
        #设置整个graph的初始化方式
        initializer = tf.random_uniform_initializer(-FLAGS.init_scale, FLAGS.init_scale)

        x_train = tf.placeholder(tf.int32, shape=[FLAGS.train_batch_size, None])
        y_train = tf.placeholder(tf.int32, shape=[FLAGS.train_batch_size, None])                
        with tf.variable_scope("Model", reuse=None, initializer=initializer):
            m = Rnn_model('lstm', x_train, y_train, len(words_tuple), FLAGS.cell_size, FLAGS.num_layers)
        
        saver = tf.train.Saver()
            
        with tf.Session() as session:
            session.run(tf.global_variables_initializer()) 
         
            for j in range(FLAGS.nb_epoch): 
                for i in range(len(x_batches)):
                    _, loss = session.run([m.train_op, m.loss], feed_dict = {x_train: x_batches[i], y_train: y_batches[i]})  
                    print("Epoch: %d batch_num: %d loss: %.3f" % (j, i, loss))

            #保存模型时一定注意保存的路径必须是英文的，中文会报错
            save_path = saver.save(session, FLAGS.checkpoints_dir + '/'+ FLAGS.model_prefix)
            print("Model saved in file: ", save_path)



def main(argv):
    
    train()
    

if __name__ == '__main__':
    
    
    tf.app.run()

    