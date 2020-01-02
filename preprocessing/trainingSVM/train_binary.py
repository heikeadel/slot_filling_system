#!/usr/bin/python

#####################################
### CIS SLOT FILLING SYSTEM      ####
### 2014-2015                    ####
### Author: Heike Adel           ####
#####################################

import sys
from scipy.io import mmread
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
import cPickle

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

if len(sys.argv) != 4:
  logger.error("please pass the basename of the feature file (without .mtx) for training, the basename of the feature file for development and the file where the model should be written to as parameter")
  exit()

trainfile = sys.argv[1]
devfile = sys.argv[2]
modelfile = sys.argv[3]

# read features
X_train = mmread(trainfile + ".mtx")

train_labels = []
f = open(trainfile + ".labels")
for line in f:
  line = line.strip()
  train_labels.append(int(line))
f.close()
y_train = np.array(train_labels)

X_test = mmread(devfile + ".mtx")
dev_labels = []
f = open(devfile + ".labels")
for line in f:
  line = line.strip()
  dev_labels.append(int(line))
f.close()
y_test = np.array(dev_labels)

# optimize C with cross validation
myCList = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]

bestC = -1
bestScore = -1
for myC in myCList:
  clf = LinearSVC(C=myC, random_state=84447)
  clf.fit(X_train, y_train)
  predictions = clf.predict(X_test)
  f1 = f1_score(y_test, predictions)
  logger.info("C = " + str(myC) + " :: " + "F1 = " + str(f1))
  if f1 > bestScore:
    bestScore = f1
    bestC = myC
    # store classifier
    save_file = open(modelfile, 'wb')
    cPickle.dump(clf, save_file, -1)
    save_file.close()
logger.info("best C = " + str(bestC) + " with F1 = " + str(bestScore))
