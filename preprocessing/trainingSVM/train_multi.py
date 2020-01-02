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
from sklearn.externals import joblib

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

def getF1(tp, numHypo, numRef):
    precision = 1
    recall = 0
    f1 = 0
    if numHypo > 0:
      precision = 1.0 * tp / numHypo
    if numRef > 0:
      recall = 1.0 * tp / numRef
    if precision + recall > 0:
      f1 = 2 * precision * recall / (precision + recall)
    logger.info("P = " + str(precision) + ", R = " + str(recall) + ", F1 = " + str(f1))
    return f1

def getFscore(hypos, refs):
    class2tp = {}
    class2numHypo = {}
    class2numRef = {}
    total = 0
    tp_tn = 0
    for i in range(len(hypos)):
        hypo = hypos[i]
        ref = refs[i]
        if hypo > 0 and not hypo in class2numHypo:
          class2numHypo[hypo] = 0
          class2numRef[hypo] = 0
          class2tp[hypo] = 0
        if ref > 0 and not ref in class2numHypo:
          class2numHypo[ref] = 0
          class2numRef[ref] = 0
          class2tp[ref] = 0
        if hypo > 0:
          class2numHypo[hypo] += 1
          if hypo == ref:
            class2tp[hypo] += 1
        if ref > 0:
          class2numRef[ref] += 1
        if hypo == ref:
          tp_tn += 1
        total += 1
    logger.info("accurracy: " + str(100.0 * tp_tn / total))
    f1sum = 0
    for cl in sorted(class2tp.keys()):
      logger.info("class " + str(cl))
      f1 = getF1(class2tp[cl], class2numHypo[cl], class2numRef[cl])
      f1sum += f1
    macroF1 = f1sum / len(class2tp.keys())
    logger.info("macro averaged F1: " + str(macroF1))
    logger.info("---")
    return macroF1

if __name__ == '__main__':

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
    clf = LinearSVC(C=myC, random_state=84447, class_weight='auto') # might need to be changed to balanced, depending on scikit version
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    f1 = getFscore(predictions, y_test)
    logger.info("C = " + str(myC) + " :: " + "F1 = " + str(f1))
    if f1 > bestScore:
      bestScore = f1
      bestC = myC
      # store classifier
      joblib.dump(clf, modelfile)
  logger.info("best C = " + str(bestC) + " with F1 = " + str(bestScore))
