#!/usr/bin/env python

import sys

import numpy as np
from sklearn import metrics
from modshogun import BinaryLabels, StringCharFeatures, DNA, WeightedDegreePositionStringKernel, SVMLight
from matplotlib import pyplot as plt

C_LST = [0.01,0.1,0.5,1.0]
DEGREE_LST = [5, 10, 15, 20]
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


def cross_validation(X, Y, d, c):
    N = len(Y)
    n = N/K
    
    accuracy_list = []
    
    for k in range(0, K):
        print 'degree = %s\tC = %s\tcross_validation_iter = %s/%s' % (d, c, k+1, K)
        sys.stdout.flush()
        
        X_test = list(X[k:k+n])
        Y_test = list(Y[k:k+n])
        X_train = []
        X_train.extend(X[:k])
        X_train.extend(X[k+n:])
        Y_train = []
        Y_train.extend(Y[:k])
        Y_train.extend(Y[k+n:])
        
        X_train = StringCharFeatures(X_train, DNA)
        X_test = StringCharFeatures(X_test, DNA)
        Y_train = BinaryLabels(np.array(Y_train, dtype=np.float64))
        Y_test = np.array(Y_test)
        
        args_tuple = (X_train, Y_train, X_test, Y_test, d, c)
        accuracy, Y_test_proba = svm_process(args_tuple)
        accuracy_list.append(accuracy)
    
    return np.array(accuracy_list).mean()


def svm_process(args_tuple):
    X_train, Y_train, X_test, Y_test, d, c = args_tuple
    
    kernel = WeightedDegreePositionStringKernel(X_train, X_train, d)
    kernel.set_shifts(np.ones(SEQ_LEN, dtype=np.int32))
    kernel.set_position_weights(np.ones(SEQ_LEN, dtype=np.float64))
    kernel.init(X_train, X_train)
     
    model = SVMLight(c, kernel, Y_train)
    model.train()
        
    Y_test_pred = model.apply(X_test).get_labels()
    Y_test_dist = model.apply(X_test).get_values()
    Y_test_proba = np.exp(Y_test_dist)/(1 + np.exp(Y_test_dist))
        
    accuracy = np.where(Y_test_pred - Y_test == 0)[0].size*1.0/Y_test.size
    
    return (accuracy, Y_test_proba)


def main():
    base_name = sys.argv[1]
    
    data_train = './data_seq_train.txt'
    data_test = './data_seq_test.txt'
    
    X_train, Y_train = parse(data_train)
    X_test, Y_test = parse(data_test)
    
    D_best = 0
    C_best = 0
    accuracy_cv_best = 0
    
    for d in DEGREE_LST:        
        for c in C_LST:
            accuracy_cv = cross_validation(X_train, Y_train, d, c)
            
            if accuracy_cv > accuracy_cv_best:
                D_best = d
                C_best = c
                accuracy_cv_best = accuracy_cv
        
            print 'degree = %s\tC = %s\tcross_validation_accuracy = %.1f%%' % (d, c, accuracy_cv*100)
            sys.stdout.flush()
        
    print 'degree_best = %s\tC_best = %s\tcross_validation_accuracy_best = %.1f%%' % (D_best, C_best, accuracy_cv_best*100)
    
    X_train = StringCharFeatures(X_train, DNA)
    X_test = StringCharFeatures(X_test, DNA)
    Y_train = BinaryLabels(np.array(Y_train, dtype=np.float64))
    Y_test = np.array(Y_test)
        
    args_tuple = (X_train, Y_train, X_test, Y_test, D_best, C_best)
    accuracy, Y_test_proba = svm_process(args_tuple)
    
    auc = metrics.roc_auc_score(Y_test, Y_test_proba)
    fpr, tpr, thresholds = metrics.roc_curve(Y_test, Y_test_proba)
     
    np.save(base_name + '_fpr.npy', fpr)
    np.save(base_name + '_tpr.npy', tpr)
    outfile = open(base_name + '_results.txt', 'w')
    outfile.write('accuracy\t' + str(accuracy) + '\n')
    outfile.write('AUC\t' + str(auc) + '\n')
    outfile.close()
    
#     plt.figure(figsize=(10, 8))
#     plt.plot(fpr, tpr, 'b.')
#     plt.title('WD SVM; accuracy = %.1f%%; AUC = %.3f' % (accuracy*100, auc))
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.xlim((0, 1))
#     plt.ylim((0, 1))
#     plt.show()

    
    
if __name__ == '__main__':
    main()

