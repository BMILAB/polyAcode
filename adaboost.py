#!/usr/bin/env python

import sys
import os

import numpy as np
from sklearn import ensemble, tree, metrics, cross_validation
from matplotlib import pyplot as plt

os.system("taskset -p 0xff %d" % os.getpid())

MAX_DEPTH = 4
MIN_LEAF = 1
D = 658
K = 10
N = 5

ESTIMATOR_LST = [10, 50, 100, 150, 200]
RATE_LST = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64]


def main():
    base_name = sys.argv[1]
    N = int(sys.argv[2])
    
    data_train = np.load('./data_features_train.npy')
    data_test = np.load('./data_features_test.npy')
    X_train = data_train[:, 0:-1]
    Y_train = data_train[:, -1]
    X_test = data_test[:, 0:-1]
    Y_test = data_test[:, -1]
    
    dt = tree.DecisionTreeClassifier(max_depth=MAX_DEPTH, min_samples_leaf=MIN_LEAF)
    dt.fit(X_train, Y_train)
    
    estimator_best = 0
    rate_best = 0
    accuracy_cv_best = 0
    
    for estimator in ESTIMATOR_LST:
        for rate in RATE_LST:
            model = ensemble.AdaBoostClassifier(base_estimator=dt, learning_rate=rate, n_estimators=estimator, algorithm="SAMME.R")
            accuracy_cv = cross_validation.cross_val_score(model, X_train, Y_train, cv=K, n_jobs=N).mean()
            
            if accuracy_cv > accuracy_cv_best:
                estimator_best = estimator
                rate_best = rate
                accuracy_cv_best = accuracy_cv
                
            print 'n_estimator = %s\tlearning_rate = %s\tcross_validation_accuracy = %.1f%%' % (estimator, rate, accuracy_cv*100)
            sys.stdout.flush()
    
    print 'n_estimator_best = %s\tlearning_rate_best = %s\tcross_validation_accuracy_best = %.1f%%' % (estimator_best, rate_best, accuracy_cv_best*100)
    
    model = ensemble.AdaBoostClassifier(base_estimator=dt, learning_rate=rate_best, n_estimators=estimator_best, algorithm="SAMME.R")    
    model.fit(X_train, Y_train)
    
    Y_test_proba = model.predict_proba(X_test)[:, 1]
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
#     plt.title('AdaBoost; accuracy = %.1f%%; AUC = %.3f' % (accuracy*100, auc))
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.xlim((0, 1))
#     plt.ylim((0, 1))
#     plt.show()
    
if __name__ == '__main__':
    main()

