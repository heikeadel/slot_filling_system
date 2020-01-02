#!/usr/bin/python

#####################################
### CIS SLOT FILLING SYSTEM      ####
### 2014-2015                    ####
### Author: Heike Adel           ####
#####################################

import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import doSVMparallel
import doCNNparallel
import doPatternMatching
import re
from sklearn.metrics import precision_recall_fscore_support
import doLoad

import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

if len(sys.argv) != 5:
  logger.error("please pass the slot for optimization, the SVM version, the CNN version and the name of the outputfile as parameter")
  exit()

pwd = os.getcwd()

slot = sys.argv[1]
svmVersion = sys.argv[2]
cnnVersion = sys.argv[3]
outfile = sys.argv[4]

os.chdir("..")

validationdata="data/validationData/" + slot # this can be e.g. the data presented in (Adel and Sch\"{u}tze, NAACL 2016)

slot2filler = {}

# read in validation data
slot2reference = {}
f = open(validationdata, 'r')
# assumed format:
# +/- : slot : query entity : filler candidate : proof sentence\n
for line in f:
  line = line.strip()
  parts = line.split(' : ')
  pos = parts[0]
  ref = -1
  if pos == '+':
    ref = 1
  slot = parts[1]
  name = parts[2]
  fill = parts[3]
  ex = " : ".join(parts[4:]) # for the case that proof sentence also contains ':'

  if "cit" in slot or "countr" in slot or "province" in slot:
        slot = re.sub(r'city', 'location', slot)
        slot = re.sub(r'country', 'location', slot)
        slot = re.sub(r'statesorprovinces', 'locations', slot)
        slot = re.sub(r'cities', 'locations', slot)
        slot = re.sub(r'countries', 'locations', slot)
        slot = re.sub(r'stateorprovince', 'location', slot)

  if not slot in slot2filler:
    slot2filler[slot] = []
  slot2filler[slot].append([fill, ex, "", "", "", ""]) # list expected by doSVM/doCNN scripts
  if not slot in slot2reference:
    slot2reference[slot] = []
  slot2reference[slot].append(ref)
f.close()

slot2inverse = doLoad.getSlot2Inverse()
patternsAllSlots = doLoad.readPatterns()

weight2resultsCMB = {}
resultsSVM = doSVMparallel.classify(slot2filler, slot2inverse, svmVersion)
resultsCNN = doCNNparallel.classify(slot2filler, slot2inverse, cnnVersion)
resultsPattern = doPatternMatching.match(slot2filler, patternsAllSlots, slot2inverse)
for slot in resultsSVM:
  if slot in slot2inverse:
    slot_eval = slot2inverse[slot]
  else:
    slot_eval = slot
  for weightSVM in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    for weightCNN in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
      if weightSVM + weightCNN > 1.0: # make sure weights sum to 1
        continue
      weightPAT = round(1.0 - weightSVM - weightCNN, 1) # make sure weights sum to 1
      if weightPAT == -0.0:
        weightPAT = 0.0
      weightString = str(weightSVM) + "-" + str(weightCNN) + "-" + str(weightPAT)
      for rs, rc, rp in zip(resultsSVM[slot], resultsCNN[slot], resultsPattern[slot]):
        cmb = weightSVM * rs[2] + weightCNN * rc[2] + weightPAT * rp[2]
        if not weightString in weight2resultsCMB:
          weight2resultsCMB[weightString] = []
        weight2resultsCMB[weightString].append([rs[0], rs[1], cmb, rs[3], rs[4], rs[5], rs[6]])

thresholdList = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
if "multi" in svmVersion or "multi" in cnnVersion: # confidences are lower with multiclass models
  thresholdList = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]

bestWeightSVM = 0.33
bestWeightCNN = 0.33
bestWeightPAT = 0.33
bestThreshold = 0.5
bestF1 = -1.0
for weights in weight2resultsCMB:
  for threshold in thresholdList:
    classificationResults = []
    for res in weight2resultsCMB[weights]:
      cmb = res[2]
      if cmb > threshold:
        classificationResults.append(1)
      else:
        classificationResults.append(-1)
    try:
      P,R,F1,num = precision_recall_fscore_support(slot2reference[slot], classificationResults, average='binary')
    except UndefinedMetricWarning:
      P = 0
      R = 0
      F1 = 0
    weightSVM, weightCNN, weightPAT = weights.split('-')
    if F1 > bestF1:
      bestWeightSVM = weightSVM
      bestWeightCNN = weightCNN
      bestWeightPAT = weightPAT
      bestThreshold = threshold
      bestF1 = F1
    logger.debug("SVM:" + weightSVM + ",CNN:" + weightCNN + ",PAT:" + weightPAT + "\t" + str(threshold) + "\t" + str(P) + "\t" + str(R) + "\t" + str(F1))

outThreshold = open(pwd + "/slot2threshold." + outfile, 'a')
outThreshold.write(slot + " : " + str(bestThreshold))
outThreshold.close()

outWeight = open(pwd + "/slot2weight." + outfile, 'a')
outWeight.write(slot + "\t" + str(bestWeightSVM) + "\t" + str(bestWeightCNN) + "\t" + str(bestWeightPAT) + "\n")
outWeight.close()
