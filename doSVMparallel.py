#!/usr/bin/python
# -*- coding: utf-8 -*-

#####################################
### CIS SLOT FILLING SYSTEM      ####
### 2014-2015                    ####
### Author: Heike Adel           ####
#####################################

from __future__ import unicode_literals
import codecs, sys
reload(sys)
sys.setdefaultencoding('utf-8')
sys.stdout = codecs.getwriter('utf8')(sys.stdout)
sys.stderr = codecs.getwriter('utf8')(sys.stderr)
import numpy
from sklearn import svm
from sklearn.externals import joblib
import os.path
import cPickle
from scipy import sparse
import math
from multiprocessing import Process, Queue
from scipy import sparse
from scipy.io import mmwrite
from sklearn.feature_extraction.text import CountVectorizer
import re

import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
logger.setLevel(logging.DEBUG)

####################### this script calls the classifiers for each slot and returns all results with confidence values over a given threshold ############

slotList = ["N", "per:age","per:alternate_names","per:children","per:cause_of_death","per:date_of_birth","per:date_of_death","per:employee_or_member_of","per:location_of_birth","per:location_of_death","per:locations_of_residence","per:origin","per:schools_attended", "per:siblings", "per:spouse", "per:title", "org:alternate_names", "org:date_founded", "org:founded_by", "org:location_of_headquarters", "org:members", "org:parents", "org:top_members_employees"]

def sigmoid(x):
  sigValue = 1.0 / (1 + math.exp(-x))
  return sigValue

def getFeatures_skip(slot, vocab, candidateAndFillerAndOffsetList, skipVocab, slot2inverse):
    ngram_vectorizer = CountVectorizer(
         ngram_range=(1,3),
         lowercase=False,
         binary=True,
         token_pattern=u'[^ ]+',
         vocabulary=vocab
    )

    examplesLeft = []
    examplesRight = []
    examplesMiddle = []
    examples = []
    flagValues = []
    capValues = []
    flagRows = []
    flagCols = []
    skipNValues = []
    skipNRows = []
    skipNCols = []
    index = 0
    for cf in candidateAndFillerAndOffsetList:
      filler = cf[0]
      numCap = 0
      for fi in filler.split():
        if fi[0].isupper():
          numCap += 1
      fillerCapRatio = numCap * 1.0 / len(filler.split())
      curEx = cf[1]
      if slot in slot2inverse:
        # reverse name and filler
        c_tmp = re.sub(ur' \<name\> ', ' NAME ', ' ' + curEx + ' ', re.UNICODE)
        c_tmp = re.sub(ur' \<filler\> ', ' <name> ', c_tmp, re.UNICODE)
        c_tmp = re.sub(ur' NAME ', ' <filler> ', c_tmp, re.UNICODE)
        curEx = c_tmp.strip()
      examples.append(curEx)
      curExList = curEx.split()
      fillerIndices = [i for i, x in enumerate(curExList) if x == "<filler>"]
      nameIndices = [i for i, x in enumerate(curExList) if x == "<name>"]
      fillerInd = -1
      nameInd = -1
      distanceNameFiller = len(curExList)

      for fi in fillerIndices:
        for ni in nameIndices:
          distance = abs(ni - fi)
          if distance < distanceNameFiller:
            distanceNameFiller = distance
            nameInd = ni
            fillerInd = fi
      minInd = 0
      maxInd = 0
      nameBeforeFiller = -1
      if fillerInd < nameInd:
        nameBeforeFiller = 0
        minInd = fillerInd
        maxInd = nameInd
      else:
        nameBeforeFiller = 1
        maxInd = fillerInd
        minInd = nameInd
      flagRows.append(index)
      flagCols.append(0)
      flagValues.append(nameBeforeFiller)
      capValues.append(fillerCapRatio)
      examplesLeft.append(" ".join(curExList[0:minInd]))
      examplesMiddle.append(" ".join(curExList[minInd+1:maxInd]))
      examplesRight.append(" ".join(curExList[maxInd+1:]))

      mcList = curExList[minInd + 1:maxInd]
      foundSkipNgram = False
      for n in range(3,5):
        for i in range(0,len(mcList) + 1 - n):
          curContext = []
          for j in range(0, n):
            if j == 0 or j == n-1:
              curContext.append(mcList[i+j])
          curContextString = " ".join(curContext)
          if curContextString in skipVocab:
            curIndex = skipVocab[curContextString]
            skipNRows.append(index)
            skipNCols.append(curIndex)
            skipNValues.append(1)
            foundSkipNgram = True
      if foundSkipNgram == False:
        skipNRows.append(index)
        skipNCols.append(0)
        skipNValues.append(0)

      index += 1

    bowMatrixWhole = ngram_vectorizer.transform(examples)
    bowMatrixLeft = ngram_vectorizer.transform(examplesLeft)
    bowMatrixMiddle = ngram_vectorizer.transform(examplesMiddle)
    bowMatrixRight = ngram_vectorizer.transform(examplesRight)
    flagMatrix = sparse.csr_matrix((numpy.array(flagValues), (numpy.array(flagRows), numpy.array(flagCols))), shape = (flagRows[-1] + 1, 1))
    skipNCounts = sparse.csr_matrix((numpy.array(skipNValues), (numpy.array(skipNRows), numpy.array(skipNCols))), shape = (skipNRows[-1] + 1, len(skipVocab.keys())))

    # stack all features:
    counts = sparse.hstack((flagMatrix, bowMatrixWhole))
    counts = sparse.hstack((counts, bowMatrixLeft))
    counts = sparse.hstack((counts, bowMatrixMiddle))
    counts = sparse.hstack((counts, bowMatrixRight))
    featuresArray = sparse.hstack((counts, skipNCounts))

    return featuresArray

def getFeatures_bow(slot, vocab, candidateAndFillerAndOffsetList, slot2inverse):
    unigram_vectorizer = CountVectorizer(
         ngram_range=(1,1),
         lowercase=False,
         binary=True,
         token_pattern=u'[^ ]+',
         vocabulary=vocab
    )

    examplesLeft = []
    examplesRight = []
    examplesMiddle = []
    examples = []
    flagValues = []
    capValues = []
    flagRows = []
    flagCols = []
    index = 0
    for cf in candidateAndFillerAndOffsetList:
      filler = cf[0]
      numCap = 0
      for fi in filler.split():
        if fi[0].isupper():
          numCap += 1
      fillerCapRatio = numCap * 1.0 / len(filler.split())
      curEx = cf[1]
      if slot in slot2inverse:
        # reverse name and filler
        c_tmp = re.sub(ur' \<name\> ', ' NAME ', ' ' + curEx + ' ', re.UNICODE)
        c_tmp = re.sub(ur' \<filler\> ', ' <name> ', c_tmp, re.UNICODE)
        c_tmp = re.sub(ur' NAME ', ' <filler> ', c_tmp, re.UNICODE)
        curEx = c_tmp.strip()
      examples.append(curEx)
      curExList = curEx.split()
      fillerIndices = [i for i, x in enumerate(curExList) if x == "<filler>"]
      nameIndices = [i for i, x in enumerate(curExList) if x == "<name>"]
      fillerInd = -1
      nameInd = -1
      distanceNameFiller = len(curExList)

      for fi in fillerIndices:
        for ni in nameIndices:
          distance = abs(ni - fi)
          if distance < distanceNameFiller:
            distanceNameFiller = distance
            nameInd = ni
            fillerInd = fi
      minInd = 0
      maxInd = 0
      nameBeforeFiller = -1
      if fillerInd < nameInd:
        nameBeforeFiller = 0
        minInd = fillerInd
        maxInd = nameInd
      else:
        nameBeforeFiller = 1
        maxInd = fillerInd
        minInd = nameInd
      flagRows.append(index)
      flagCols.append(0)
      flagValues.append(nameBeforeFiller)
      capValues.append(fillerCapRatio)
      examplesLeft.append(" ".join(curExList[0:minInd]))
      examplesMiddle.append(" ".join(curExList[minInd+1:maxInd]))
      examplesRight.append(" ".join(curExList[maxInd+1:]))
      index += 1

    bowMatrixWhole = unigram_vectorizer.transform(examples)
    bowMatrixLeft = unigram_vectorizer.transform(examplesLeft)
    bowMatrixMiddle = unigram_vectorizer.transform(examplesMiddle)
    bowMatrixRight = unigram_vectorizer.transform(examplesRight)
    flagMatrix = sparse.csr_matrix((numpy.array(flagValues), (numpy.array(flagRows), numpy.array(flagCols))), shape = (flagRows[-1] + 1, 1))

    featuresArray = sparse.hstack((flagMatrix, bowMatrixWhole, bowMatrixLeft, bowMatrixMiddle, bowMatrixRight))

    return featuresArray


def run_binary_bow(slot, candidateAndFillerAndOffsetList, queue, vocab, skipVocab, slot2inverse):

    modelDir = "svm/models_bow"
    logger.info("binary SVMbow: " + slot + ": " + str(len(candidateAndFillerAndOffsetList)) + " candidates")
    if slot in slot2inverse:
      slot_eval = slot2inverse[slot]
      logger.info("using model from slot: " + slot_eval)
    else:
      slot_eval = slot
    if not os.path.isfile(modelDir + "/" + slot_eval + ".model"):
      logger.error("SVM: no model found for " + slot_eval)
      queue.put([])
      return

    results = []

    featuresArray = getFeatures_bow(slot, vocab, candidateAndFillerAndOffsetList)

    # load svm
    modelfile = open(modelDir + "/" + slot_eval + ".model", 'rb')
    clf = cPickle.load(modelfile)
    modelfile.close()
    # evaluate SVM
    confidence = clf.decision_function(featuresArray)
    for co in range(0, len(confidence)):
      myProb = sigmoid(confidence[co])
      results.append([candidateAndFillerAndOffsetList[co][0], candidateAndFillerAndOffsetList[co][1], myProb, candidateAndFillerAndOffsetList[co][2], candidateAndFillerAndOffsetList[co][3], candidateAndFillerAndOffsetList[co][4], candidateAndFillerAndOffsetList[co][5]])

    queue.put(results)
    return

def run_binary_skip(slot, candidateAndFillerAndOffsetList, queue, vocab, skipVocab, slot2inverse):

    modelDir = "svm/models_skip"
    logger.info("INFO: binary SVMskip: " + slot + ": " + str(len(candidateAndFillerAndOffsetList)) + " candidates")
    if slot in slot2inverse:
      slot_eval = slot2inverse[slot]
      logger.info("INFO: using model from slot: " + slot_eval)
    else:
      slot_eval = slot
    modelfilename = modelDir + "/" + slot_eval + ".model"
    if not os.path.isfile(modelfilename):
      logger.error("SVM: no model found for " + slot_eval)
      queue.put([])
      return

    results = []

    featuresArray = getFeatures_skip(slot, vocab, candidateAndFillerAndOffsetList, skipVocab, slot2inverse)

    # load svm
    modelfile = open(modelfilename, 'rb')
    clf = cPickle.load(modelfile)
    modelfile.close()
    # evaluate SVM
    confidence = clf.decision_function(featuresArray)
    for co in range(0, len(confidence)):
      myProb = sigmoid(confidence[co])
      results.append([candidateAndFillerAndOffsetList[co][0], candidateAndFillerAndOffsetList[co][1], myProb, candidateAndFillerAndOffsetList[co][2], candidateAndFillerAndOffsetList[co][3], candidateAndFillerAndOffsetList[co][4], candidateAndFillerAndOffsetList[co][5]])

    queue.put(results)
    return

def run_multi_skip(slot, candidateAndFillerAndOffsetList, queue, vocab, skipVocab, slot2inverse):
    logger.info("multiclass SVMskip: " + slot + ": " + str(len(candidateAndFillerAndOffsetList)) + " candidates")
    if slot in slot2inverse:
      slot_eval = slot2inverse[slot]
      logger.info("using prediction from slot: " + slot_eval)
    else:
      slot_eval = slot
    modelfilename = "svm/models_skip/multiClass.sameOthersAsPositives.model.joblib"
    if not os.path.isfile(modelfilename):
      logger.error("SVM: no model found")
      queue.put([])
      return
    slot_index = slotList.index(slot_eval)

    results = []

    featuresArray = getFeatures_skip(slot, vocab, candidateAndFillerAndOffsetList, skipVocab, slot2inverse)

    # load svm
    clf = joblib.load(modelfilename)
    # evaluate SVM
    confidence = clf.decision_function(featuresArray)
    for co in range(0, len(confidence)):
      myProb = sigmoid(confidence[co][slot_index])
      results.append([candidateAndFillerAndOffsetList[co][0], candidateAndFillerAndOffsetList[co][1], myProb, candidateAndFillerAndOffsetList[co][2], candidateAndFillerAndOffsetList[co][3], candidateAndFillerAndOffsetList[co][4], candidateAndFillerAndOffsetList[co][5]])

    queue.put(results)
    return



def classify(slot2candidates, slot2inverse, SVMversion):

  # SVMversion: one of "binaryBOW", "binarySkip", "multiSkip"
  if SVMversion == "multiSkip":
    runSingleSVM = run_multi_skip
  elif SVMversion == "binaryBOW":
    runSingleSVM = run_binary_bow
  else: # default: binarySkip
    runSingleSVM = run_binary_skip

  numberOfProcesses = 5

  # cleaning slot2candidates: removing entries with empty values:
  slotListToClassify = list(slot2candidates.keys())
  for s in slotListToClassify:
    if len(slot2candidates[s]) == 0:
      del slot2candidates[s]

  slotListToClassify = list(slot2candidates.keys())

  logger.info("got the following slots: " + str(slotListToClassify))

  numberOfLoops = (len(slotListToClassify) + numberOfProcesses - 1) / numberOfProcesses

  slot2candidatesAndFillersAndConfidence = {}

  logger.info("total number of loops to do: " + str(numberOfLoops))

  for n in range(numberOfLoops):

    logger.info(str(n) + "-th loop")

    slotsOfLoop = slotListToClassify[n * numberOfProcesses : min((n + 1) * numberOfProcesses, len(slotListToClassify))]
    proc = []
    queues = []
    # start processes
    for slot in slotsOfLoop:
      if slot in slot2inverse:
        slotEval = slot2inverse[slot]
      else:
        slotEval = slot
      if SVMversion == "multiSkip":
        vocabfile = "svm/vocabs_skip/multiClass.sameOthersAsPositives.vocab"
        skipVocab = {}
        f = open(vocabfile + ".skip", 'r')
        for index, line in enumerate(f):
          line = line.strip()
          skipVocab[line] = index
        f.close()
      elif SVMversion == "binaryBOW":
        vocabfile = "svm/vocabs_bow" + slotEval + ".vocab"
        skipVocab = {}
      else: # default: binarySkip
        vocabfile = "svm/vocabs_skip/" + slotEval + ".vocab"
        skipVocab = {}
        f = open(vocabfile + ".skip", 'r')
        for index, line in enumerate(f):
          line = line.strip()
          skipVocab[line] = index
        f.close()

      vocab = []
      f = open(vocabfile, 'r')
      for line in f:
        line = line.strip()
        vocab.append(line)
      f.close()

      candidateAndFillerAndOffsetList = slot2candidates[slot]
      q = Queue()
      queues.append(q)
      p = Process(target=runSingleSVM, args=(slot, candidateAndFillerAndOffsetList, q, vocab, skipVocab, slot2inverse))
      p.start()
      proc.append(p)

    # collect results
    for i,q in enumerate(queues):
      threadResults = q.get()
      slot2candidatesAndFillersAndConfidence[slotsOfLoop[i]] = threadResults

    # wait until all processes have finished
    for p in proc:
      p.join()

  return slot2candidatesAndFillersAndConfidence


