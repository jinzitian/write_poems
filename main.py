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
        
    if args[1] == 'generate_poem':
        for i in range(10):
            poem = generate_poem()
            count = collections.Counter(poem)
            t = poem.replace('，','。')
            a = np.array(list(map(len, t.split('。')[:-1])))
            if (a - a[0]).sum() == 0 and count['，'] == count['。']:
                for i in poem.split('。')[:-1]:
                    print(i + '。')
                break
            if i == 9:
                print('not lucky , please try again~\n')
                
    if args[1] == 'generate_your_poem':
        for i in range(100):
            poem = generate_your_poem(args[2])
            count = collections.Counter(poem)
            t = poem.replace('，','。')
            a = np.array(list(map(len, t.split('。')[:-1])))
            if (a - a[0]).sum() == 0 and count['，'] == count['。']:
                for i in poem.split('。')[:-1]:
                    print(i + '。')
                break
            if i == 99:
                print('not lucky , please try again~\n')
        
if __name__ == '__main__':
    
    main(sys.argv)
    


    
    
    
    