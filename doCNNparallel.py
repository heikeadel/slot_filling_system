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
from sklearn.externals import joblib
import re
import copy
import os.path
import cPickle
import math
from multiprocessing import Process, Queue
from cnnScripts.testCNN_multiClass_global import CNN as CNNglobal
from cnnScripts.testCNN_multiClass_withJointNER import CNN as CNNjoint
from cnnScripts.testCNN_multiClass_withNERinput import CNN as CNNpipeline
from cnnScripts.testCNN_multiClass import CNN as CNNmulti
from cnnScripts.testCNN_binary import CNN as CNNbinary

import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
logger.setLevel(logging.DEBUG)

####################### this script calls the classifiers for each slot and returns all results with confidence values over a given threshold ############

slotList = ["N", "per:age","per:alternate_names","per:children","per:cause_of_death","per:date_of_birth","per:date_of_death","per:employee_or_member_of","per:location_of_birth","per:location_of_death","per:locations_of_residence","per:origin","per:schools_attended", "per:siblings", "per:spouse", "per:title", "org:alternate_names", "org:date_founded", "org:founded_by", "org:location_of_headquarters", "org:members", "org:parents", "org:top_members_employees"]

def binary_run(slot, candidateAndFillerAndOffsetList, queue, slot2inverse):
    logger.info("binary CNN: " + slot + ": " + str(len(candidateAndFillerAndOffsetList)) + " candidates")
    if slot in slot2inverse:
      slot_eval = slot2inverse[slot]
      logger.info("using model from slot: " + slot_eval)
    else:
      slot_eval = slot
    config = "cnn/configs_binary/config_" + slot_eval
    if not os.path.isfile(config):
      logger.error("CNN: no config found for " + slot_eval)
      queue.put([])
      return

    myCNN = CNNbinary(config)

    results = []

    candidateListToClassify = []
    if slot in slot2inverse:
      for cList in candidateAndFillerAndOffsetList:
        # reverse name and filler
        c_tmp = re.sub(ur' \<name\> ', ' NAME ', ' ' + cList[1] + ' ', re.UNICODE)
        c_tmp = re.sub(ur' \<filler\> ', ' <name> ', c_tmp, re.UNICODE)
        c_tmp = re.sub(ur' NAME ', ' <filler> ', c_tmp, re.UNICODE)
        c = c_tmp.strip()
        candidateListToClassify.append([cList[0], c, cList[2], cList[3], cList[4], cList[5]])
    else:
      candidateListToClassify = copy.deepcopy(candidateAndFillerAndOffsetList)
    confidence = myCNN.classify(candidateListToClassify)

    for co in range(0, len(confidence)):
      myProb = confidence[co][2].item(1)
      results.append([candidateAndFillerAndOffsetList[co][0], candidateAndFillerAndOffsetList[co][1], myProb, candidateAndFillerAndOffsetList[co][2], candidateAndFillerAndOffsetList[co][3], candidateAndFillerAndOffsetList[co][4], candidateAndFillerAndOffsetList[co][5]])

    queue.put(results)
    return

def multi_run(slot, candidateAndFillerAndOffsetList, queue, slot2inverse):

    logger.info("multiclass CNN: " + slot + ": " + str(len(candidateAndFillerAndOffsetList)) + " candidates")
    if slot in slot2inverse:
      slot_eval = slot2inverse[slot]
      logger.info("using predictions for slot: " + slot_eval)
    else:
      slot_eval = slot
    config = "cnn/configs_multi/config.multi"
    if not os.path.isfile(config):
      logger.error("CNN: no config found for " + slot_eval)
      queue.put([])
      return

    slot_index = slotList.index(slot_eval)

    myCNN = CNNmulti(config)

    results = []

    candidateListToClassify = []
    if slot in slot2inverse:
      for cList in candidateAndFillerAndOffsetList:
        # reverse name and filler
        c_tmp = re.sub(ur' \<name\> ', ' NAME ', ' ' + cList[1] + ' ', re.UNICODE)
        c_tmp = re.sub(ur' \<filler\> ', ' <name> ', c_tmp, re.UNICODE)
        c_tmp = re.sub(ur' NAME ', ' <filler> ', c_tmp, re.UNICODE)
        c = c_tmp.strip()
        candidateListToClassify.append([cList[0], c, cList[2], cList[3], cList[4], cList[5]])
    else:
      candidateListToClassify = copy.deepcopy(candidateAndFillerAndOffsetList)
    confidence = myCNN.classify(candidateListToClassify)

    for co in range(0, len(confidence)):
      myProb = confidence[co][2][0][slot_index]
      results.append([candidateAndFillerAndOffsetList[co][0], candidateAndFillerAndOffsetList[co][1], myProb, candidateAndFillerAndOffsetList[co][2], candidateAndFillerAndOffsetList[co][3], candidateAndFillerAndOffsetList[co][4], candidateAndFillerAndOffsetList[co][5]])

    queue.put(results)
    return

def multi_joint_run(slot, candidateAndFillerAndOffsetList, queue, slot2inverse):

    logger.info("jointly trained CNN: " + slot + ": " + str(len(candidateAndFillerAndOffsetList)) + " candidates")
    if slot in slot2inverse:
      slot_eval = slot2inverse[slot]
      logger.info("using predictions for slot: " + slot_eval)
    else:
      slot_eval = slot
    config = "cnn/configs_multi/config.multi_joint"
    if not os.path.isfile(config):
      logger.error("CNN: no config found for " + slot_eval)
      queue.put([])
      return

    slot_index = slotList.index(slot_eval)

    myCNN = CNNjoint(config)

    results = []

    candidateListToClassify = []
    if slot in slot2inverse:
      for cList in candidateAndFillerAndOffsetList:
        # reverse name and filler
        c_tmp = re.sub(ur' \<name\> ', ' NAME ', ' ' + cList[1] + ' ', re.UNICODE)
        c_tmp = re.sub(ur' \<filler\> ', ' <name> ', c_tmp, re.UNICODE)
        c_tmp = re.sub(ur' NAME ', ' <filler> ', c_tmp, re.UNICODE)
        c = c_tmp.strip()
        candidateListToClassify.append([cList[0], c, cList[2], cList[3], cList[4], cList[5]])
    else:
      candidateListToClassify = copy.deepcopy(candidateAndFillerAndOffsetList)
    confidence = myCNN.classify(candidateListToClassify, slot_eval)

    for co in range(0, len(confidence)):
      myProb = confidence[co][2][0][slot_index]
      results.append([candidateAndFillerAndOffsetList[co][0], candidateAndFillerAndOffsetList[co][1], myProb, candidateAndFillerAndOffsetList[co][2], candidateAndFillerAndOffsetList[co][3], candidateAndFillerAndOffsetList[co][4], candidateAndFillerAndOffsetList[co][5]])

    queue.put(results)
    return

def multi_pipeline_run(slot, candidateAndFillerAndOffsetList, queue, slot2inverse):

    logger.info("pipeline CNN: " + slot + ": " + str(len(candidateAndFillerAndOffsetList)) + " candidates")
    if slot in slot2inverse:
      slot_eval = slot2inverse[slot]
      logger.info("using predictions for slot: " + slot_eval)
    else:
      slot_eval = slot
    config = "cnn/configs_multi/config.multi_pipeline"
    if not os.path.isfile(config):
      logger.error("CNN: no config found for " + slot_eval)
      queue.put([])
      return

    slot_index = slotList.index(slot_eval)

    myCNN = CNNpipeline(config)

    results = []

    candidateListToClassify = []
    if slot in slot2inverse:
      for cList in candidateAndFillerAndOffsetList:
        # reverse name and filler
        c_tmp = re.sub(ur' \<name\> ', ' NAME ', ' ' + cList[1] + ' ', re.UNICODE)
        c_tmp = re.sub(ur' \<filler\> ', ' <name> ', c_tmp, re.UNICODE)
        c_tmp = re.sub(ur' NAME ', ' <filler> ', c_tmp, re.UNICODE)
        c = c_tmp.strip()
        candidateListToClassify.append([cList[0], c, cList[2], cList[3], cList[4], cList[5]])
    else:
      candidateListToClassify = copy.deepcopy(candidateAndFillerAndOffsetList)
    confidence = myCNN.classify(candidateListToClassify, slot_eval)

    for co in range(0, len(confidence)):
      myProb = confidence[co][2][0][slot_index]
      results.append([candidateAndFillerAndOffsetList[co][0], candidateAndFillerAndOffsetList[co][1], myProb, candidateAndFillerAndOffsetList[co][2], candidateAndFillerAndOffsetList[co][3], candidateAndFillerAndOffsetList[co][4], candidateAndFillerAndOffsetList[co][5]])

    queue.put(results)
    return

def multi_global_run(slot, candidateAndFillerAndOffsetList, queue, slot2inverse):

    logger.info("globally normalized CNN: " + slot + ": " + str(len(candidateAndFillerAndOffsetList)) + " candidates")
    if slot in slot2inverse:
      slot_eval = slot2inverse[slot]
      logger.info("using predictions for slot: " + slot_eval)
    else:
      slot_eval = slot
    config = "cnn/configs_multi/config.multi_global"
    if not os.path.isfile(config):
      logger.error("CNN: no config found for " + slot_eval)
      queue.put([])
      return

    slot_index = slotList.index(slot_eval)

    myCNN = CNNglobal(config)

    results = []

    candidateListToClassify = []
    if slot in slot2inverse:
      for cList in candidateAndFillerAndOffsetList:
        # reverse name and filler
        c_tmp = re.sub(ur' \<name\> ', ' NAME ', ' ' + cList[1] + ' ', re.UNICODE)
        c_tmp = re.sub(ur' \<filler\> ', ' <name> ', c_tmp, re.UNICODE)
        c_tmp = re.sub(ur' NAME ', ' <filler> ', c_tmp, re.UNICODE)
        c = c_tmp.strip()
        candidateListToClassify.append([cList[0], c, cList[2], cList[3], cList[4], cList[5]])
    else:
      candidateListToClassify = copy.deepcopy(candidateAndFillerAndOffsetList)
    confidence = myCNN.classify(candidateListToClassify, slot_eval)

    for co in range(0, len(confidence)):
      myProb = confidence[co]
      results.append([candidateAndFillerAndOffsetList[co][0], candidateAndFillerAndOffsetList[co][1], myProb, candidateAndFillerAndOffsetList[co][2], candidateAndFillerAndOffsetList[co][3], candidateAndFillerAndOffsetList[co][4], candidateAndFillerAndOffsetList[co][5]])

    queue.put(results)
    return



def classify(slot2candidates, slot2inverse, CNNversion):
  # CNNversion: one of "binary", "multi", "joint", "pipeline", "global"
  if CNNversion == "multi":
    runSingleCNN = multi_run
  elif CNNversion == "joint":
    runSingleCNN = multi_joint_run
  elif CNNversion == "pipeline":
    runSingleCNN = multi_pipeline_run
  elif CNNversion == "global":
    runSingleCNN = multi_global_run
  else: # default: binary
    runSingleCNN = binary_run

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
      candidateAndFillerAndOffsetList = slot2candidates[slot]
      q = Queue()
      queues.append(q)
      p = Process(target=runSingleCNN, args=(slot, candidateAndFillerAndOffsetList, q, slot2inverse))
      p.start()
      proc.append(p)

    # collect results
    logger.debug("collecting results")
    for i,q in enumerate(queues):
      threadResults = q.get()
      slot2candidatesAndFillersAndConfidence[slotsOfLoop[i]] = threadResults
    logger.debug("done")
 
    # wait until all processes have finished
    for p in proc:
      p.join()

  return slot2candidatesAndFillersAndConfidence


