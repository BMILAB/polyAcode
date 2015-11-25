README for polyAcode
====================


INSTALL
=======

Prerequisites
-------------
**Mandatory** 

* Python (2.7). [Python 2.7.6](http://www.python.org/download/releases/2.7.6/) is recommended.

* [Numpy](http://www.numpy.org/)(>=1.6.1). 

* [Scipy](http://www.scipy.org/)(>=0.10). 

* [Scikit-learn](http://scikit-learn.org/stable/)(>=0.16.1).

**Optional** 
* [Shogun](http://www.shogun-toolbox.org/)(>=3.2.0). Required for wdsvm.py


DATA
====

**features_annotation.xls** contains detailed annotation about the 658 features.

MayrLab
-------
* polyAsites_specific.txt contains the 2276 tissue-specific PASs described in the manuscript. 
* polyAsites_unspecific.txt contains the 3903 tissue-unspecific(constitutive) PASs described in the manuscript.
* features_specific.txt contains the 658 RNA features for the 2276 tissue-specific PASs.
* features_unspecific.txt contains the 658 RNA features for the 3903 tissue-unspecific PASs.
* seq_specific.txt contains the +/- 100nt sequence for the 2276 tissue-specific PASs.
* seq_unspecific.txt contains the +/- 100nt sequence for the 3903 tissue-unspecific PASs.
* data_features_train.npy is the numpy format for training lr.py, lsvm.py adaboost.py. It is a 4000x659 matrix with 4000 corresponding to 2000 tissue-specific PASs and 2000 tissue-unspecific PASs, and 659 corresponding 658 features and the last column representing the label(+1 for tissue-specific and -1 for tissue-unspecific).
* data_features_test.npy is the numpy format for testing lr.py lsvm.py adaboost.py. It is a 552x659 matrix with 276 tissue-specific PASs and 276 tissue-unspecific PASs.
* data_features_Bcell_train.npy, data_features_Bcell_test.npy are similarly defined as above but only for Bcell-specific PASs. 
* data_features_Testis_train.npy, data_features_Testis_test.npy are similarly defined as above but only for Testis-specific PASs. 
* data_seq_train.txt is the +/- 100nt sequence with the label for training wdsvm.py. It contains 2000 tissue-specific PASs and 2000 tissue-unspecific PASs.
* data_seq_test.txt is the +/- 100nt sequence with the label for testing wdsvm.py. It contains 276 tissue-specific PASs and 276 tissue-unspecific PASs.


JohnLab
-------
* features_specific.txt contains the 658 RNA features for the 706 tissue-specific PASs.
* features_unspecific.txt contains the 658 RNA features for the 1114 tissue-unspecific PASs.
* data_features_train.npy is the numpy format for training lr.py, lsvm.py adaboost.py. It is a 1000x659 matrix with 1000 corresponding to 500 tissue-specific PASs and 500 tissue-unspecific PASs, and 659 corresponding 658 features and the last column representing the label(+1 for tissue-specific and -1 for tissue-unspecific).
* data_features_test.npy is the numpy format for testing lr.py lsvm.py adaboost.py. It is a 412x659 matrix with 206 tissue-specific PASs and 206 tissue-unspecific PASs.






