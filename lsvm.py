#!/usr/bin/env python

import sys

import numpy as np
from sklearn import svm, metrics, cross_validation
from matplotlib import pyplot as plt

C_LST = [0.0001, 0.001,0.01,0.1,1.0,2.0]
TOL = 0.00001
D = 658
K = 10


def main():
    base_name = sys.argv[1]
    N = int(sys.argv[2])
    
    data_train = np.load('./data_feature_train.npy')
    data_test = np.load('./data_feature_test.npy')
    X_train = data_train[:, 0:-1]
    Y_train = data_train[:, -1]
    X_test = data_test[:, 0:-1]
    Y_test = data_test[:, -1]

    C_best = 0
    accuracy_cv_best = 0
    
    for c in C_LST:
        model = svm.LinearSVC(C=c, penalty='l2', dual=False, tol=TOL)
        accuracy_cv = cross_validation.cross_val_score(model, X_train, Y_train, cv=K, n_jobs=N).mean()
        
        if accuracy_cv > accuracy_cv_best:
            C_best = c
            accuracy_cv_best = accuracy_cv
        
        print 'C = %s\tcross_validation_accuracy = %.1f%%' % (c, accuracy_cv*100)
        sys.stdout.flush()
        
    print 'C_best = %s\tcross_validation_accuracy_best = %.1f%%' % (C_best, accuracy_cv_best*100)
    
    model = svm.LinearSVC(C=C_best, penalty='l2', dual=False, tol=TOL)
    model.fit(X_train, Y_train)
    
    Y_test_dist = model.decision_function(X_test)
    Y_test_proba = np.exp(Y_test_dist)/(1 + np.exp(Y_test_dist))
    accuracy = model.score(X_test, Y_test)
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
#     plt.title('Linear SVM; accuracy = %.1f%%; AUC = %.3f' % (accuracy*100, auc))
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.xlim((0, 1))
#     plt.ylim((0, 1))
#     plt.show()
    
if __name__ == '__main__':
    main()

