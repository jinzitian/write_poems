# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 11:47:19 2017

@author: Jinzitian
"""

import sys
import numpy as np

from train.train import train
from generate.generate import generate_poem

        
def main(args):
    
    if args[1] == 'train':
        train()
    if args[1] == 'generate_poem':
        for i in range(10):
            poem = generate_poem()
            a = np.array(list(map(len, poem.split('。')[:-1])))
            if (a - a[0]).sum() == 0:
                for i in poem.split('。')[:-1]:
                    print(i + '。')
                break
            if i == 9:
                print('not lucky , please try again~')
        
if __name__ == '__main__':
    
    main(sys.argv)
    


    
    
    
    