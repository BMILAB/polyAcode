#!/usr/bin/env python

import sys

import numpy as np
from sklearn import linear_model, metrics
from modshogun import BinaryLabels, StringCharFeatures, DNA, WeightedDegreePositionStringKernel, SVMLight
from matplotlib import pyplot as plt

C_LST = [0.001,0.01,0.1,1.0,2.0]
TOL = 0.00001
D = 658
K = 10
SEQ_LEN = 201


def parse(infile):
    infile = open(infile)
    
    X = []
    Y = []
    
    for line in infile:
        fields = line.strip('\n').split('\t')
        X.append(fields[0])
        Y.append(float(fields[1]))
    
    infile.close()
    
    return (X, Y)


def main():
    data_train = './data_seq_train.txt'
    data_test = './data_seq_test.txt'
    
    X_train, Y_train = parse(data_train)
    X_test, Y_test = parse(data_test)

    X_train = StringCharFeatures(X_train, DNA)
    X_test = StringCharFeatures(X_test, DNA)
    Y_train = BinaryLabels(np.array(Y_train, dtype=np.float64))
    
    d = 8
    C = 0.1
    
    kernel = WeightedDegreePositionStringKernel(X_train, X_train, d)
    kernel.set_shifts(10*np.ones(SEQ_LEN, dtype=np.int32))
    kernel.set_position_weights(np.ones(SEQ_LEN, dtype=np.float64))
    kernel.init(X_train, X_train)
    
    model = SVMLight(C, kernel, Y_train)
    model.train()
    
    Y_test_pred = model.apply(X_test)
    Y_test_pred_label = Y_test_pred.get_labels()
    Y_test_pred_dist = Y_test_pred.get_values()
    
    print np.where((Y_test - Y_test_pred_label) == 0)[0].size

    
    
if __name__ == '__main__':
    main()

