#!/usr/bin/python

#####################################
### CIS SLOT FILLING SYSTEM      ####
### 2014-2015                    ####
### Author: Heike Adel           ####
#####################################

import logging
import theano.tensor as T
import time
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

slotList = ["per:age","per:alternate_names","per:children","per:cause_of_death","per:date_of_birth","per:date_of_death","per:employee_or_member_of","per:location_of_birth","per:location_of_death","per:locations_of_residence","per:origin","per:schools_attended", "per:siblings", "per:spouse", "per:title", "org:alternate_names", "org:date_founded", "org:founded_by", "org:location_of_headquarters", "org:members", "org:parents", "org:top_members_employees"]

typeList = ["O", "PERSON", "LOCATION", "ORGANIZATION", "DATE", "NUMBER"]

def calculateF1(tp, numHypo, numRef):
    precision = 1
    recall = 0
    f1 = 0
    if numHypo > 0:
      precision = 1.0 * tp / numHypo
    if numRef > 0:
      recall = 1.0 * tp / numRef
    if precision + recall > 0:
      f1 = 2 * precision * recall / (precision + recall)
    logger.info(str(time.ctime()) + "\tP = " + str(precision) + ", R = " + str(recall) + ", F1 = " + str(f1))
    return f1

def getFScore(confidence, resultVectorDev, batch_size):
  truePos = 0
  hypoOne = 0
  refOne = 0

  for r in range(0,len(confidence)):
    for r1 in range(0, batch_size):
      hypoResult = confidence[r][0].item(r1)
      refResult = resultVectorDev[r1 + r * batch_size]
      if hypoResult == refResult:
        if hypoResult == 1:
          truePos += 1
          hypoOne += 1
          refOne += 1
      else:
        if hypoResult == 1:
          hypoOne += 1
        if refResult == 1:
          refOne += 1

  return calculateF1(truePos, hypoOne, refOne)


def getFScoreMultiClass(hypos, refs, batch_size):
    class2tp = {}
    class2numHypo = {}
    class2numRef = {}
    total = 0
    tp_tn = 0

    if batch_size == None:
      for r in range(len(hypos)):
        hypo = hypos[r]
        ref = refs[r]
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

    else:
      for r in range(len(hypos)):
        for r1 in range(0, batch_size):
          hypo = hypos[r][0].item(r1)
          ref = refs[r1 + r * batch_size]
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
      f1 = calculateF1(class2tp[cl], class2numHypo[cl], class2numRef[cl])
      f1sum += f1
    logger.info("macro averaged F1: " + str(f1sum / len(class2tp.keys())))
    logger.info("---")
    return f1sum / len(class2tp.keys())

def sgd_updates(params, cost, learning_rate):
    updates = []
    for param in params:
      gp = T.grad(cost, param)
      step = -1.0 * learning_rate * gp
      stepped_param = param + step
      updates.append((param, stepped_param))
    return updates

def normalizeLocation(slot):
    if "cit" in slot or "countr" in slot or "province" in slot:
      slot = re.sub(ur'city', 'location', slot, re.UNICODE)
      slot = re.sub(ur'country', 'location', slot, re.UNICODE)
      slot = re.sub(ur'statesorprovinces', 'locations', slot, re.UNICODE)
      slot = re.sub(ur'cities', 'locations', slot, re.UNICODE)
      slot = re.sub(ur'countries', 'locations', slot, re.UNICODE)
      slot = re.sub(ur'stateorprovince', 'location', slot, re.UNICODE)
    return slot

def getTypes(inputfile, slot2types, binarizer):
    resultVectorNER1 = []
    resultVectorNER2 = []
    f = open(inputfile, 'r')
    for line in f:
      line = line.strip()
      if not "<name>" in line or not "<filler>" in line:
        continue # skip example
      if re.search(r'^\S+ \: ', line):
        parts = line.split(' : ')
      else:
        parts = line.split(" :: ")
      slot = parts[1]
      slot = normalizeLocation(slot)
      type1bin = binarizer.transform([slot2types[slot][0]])
      type2bin = binarizer.transform([slot2types[slot][1]])
      resultVectorNER1.append(type1bin)
      resultVectorNER2.append(type2bin)
    f.close()
    return resultVectorNER1, resultVectorNER2

def getSingleType(inputfile, slot2types, slot2typeVectors):
    resultVectorNER1 = []
    resultVectorNER2 = []
    f = open(inputfile, 'r')
    for line in f:
      line = line.strip()
      if re.search(r'^\S+ \: ', line):
        parts = line.split(' : ')
      else:
        parts = line.split(" :: ")
      slot = parts[1]
      slot = normalizeLocation(slot)
      type1bin, type2bin = slot2typeVectors[slot]
      resultVectorNER1.append(type1bin)
      resultVectorNER2.append(type2bin)
    f.close()
    return resultVectorNER1, resultVectorNER2

