'''
Created on May 22, 2009

@author: Oso
'''
from os.path import join
import numpy
import scipy.io as sio
import scipy.sparse as sp


def read_input_files():
    input_file = open(join('..','..','data','classic400.txt'),'r')
    
    print(dir(sio))
    sM = sio.loadmat(input_file)
    M = numpy.matrix()
    M = sparse.lil_matrix()
    for line in input_file:
        pass#M.
    label_file = open(join('.','data','truelabels.csv'),'r')
    
    

if __name__ == '__main__':
     read_input_files()
