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
    
    if sys.version[0] == '2':
        reload(sys)
        sys.setdefaultencoding('utf-8')
        
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
                print('\nnot lucky , please try again~\n')
                                
    elif args[1] == 'generate_your_poem':
        for i in range(100):
            try:
                if sys.version[0] == '2':
                    poem = generate_your_poem(unicode(args[2],'utf-8'))
                else:
                    poem = generate_your_poem(args[2])
            except Exception as e:
                print('\nmaybe your words are used not quite often, please change some words\n')
                break
            count = collections.Counter(poem)
            t = poem.replace('，','。')
            a = np.array(list(map(len, t.split('。')[:-1])))
            if np.sum(np.abs(a - a[0])) == 0 and count['，'] == count['。']:
                for i in poem.split('。')[:-1]:
                    print(i + '。')
                break
            if i == 99:
                print('\nnot lucky , please try again~\n')
    
    else:
        print('\nyou can try:')
        print('python main.py train')
        print('python main.py generate_poem')
        print('python main.py generate_your_poem XXXXXXX\n')
        
        
if __name__ == '__main__':
    
    main(sys.argv)
    


    
    
    
    