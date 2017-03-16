# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 11:47:19 2017

@author: Jinzitian
"""

import sys
import numpy as np
import collections

from train.train import train
from generate.generate import generate_poem,generate_your_poem

        
def main(args):
    
    if args[1] == 'train':
        train()
        
    elif args[1] == 'generate_poem':
        for i in range(10):
            poem = generate_poem()
            count = collections.Counter(poem)
            t = poem.replace('，','。')
            a = np.array(list(map(len, t.split('。')[:-1])))
            if np.sum(np.abs(a - a[0])) == 0 and count['，'] == count['。']:
                for i in poem.split('。')[:-1]:
                    print(i + '。')
                break
            if i == 9:
                print('not lucky , please try again~')
                
    elif args[1] == 'generate_your_poem':
        for i in range(100):
            try:
                poem = generate_your_poem(args[2])
            except Exception as e:
                print('maybe your words are used not quite often, please change some words')
                break
            count = collections.Counter(poem)
            t = poem.replace('，','。')
            a = np.array(list(map(len, t.split('。')[:-1])))
            if np.sum(np.abs(a - a[0])) == 0 and count['，'] == count['。']:
                for i in poem.split('。')[:-1]:
                    print(i + '。')
                break
            if i == 99:
                print('not lucky , please try again~')
    
    else:
        print('you can try:')
        print('python main.py train')
        print('python main.py generate_poem')
        print('python main.py generate_your_poem XXXXXXX')
        
        
if __name__ == '__main__':
    
    main(sys.argv)
    


    
    
    
    