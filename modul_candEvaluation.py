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
import copy
import doSVMparallel
import doPatternMatching
import doCNNparallel

class CandidateEvaluation:
  
  def setForProximity(self,forProximity):
    self.forProximity = copy.deepcopy(forProximity)

  def setForClassifier(self,forClassifier):
    self.forClassifier = copy.deepcopy(forClassifier)

  def setForPatternMatcher(self,forPatternMatcher):
    self.forPatternMatcher = copy.deepcopy(forPatternMatcher)

  def evaluateProximity(self):
    for s in self.forProximity:
      if not s in self.globalSlot2fillerCandidatesAndConfidence:
        self.globalSlot2fillerCandidatesAndConfidence[s] = []
      curList = self.forProximity[s]
      if s in self.slot2inverse:
        s_eval = self.slot2inverse[s]
      else:
        s_eval = s
      triggerList = self.slot2proximity[s_eval]
      for candidate in curList:
        # get inverse distance from nearest trigger word to filler candidate
        curExList = candidate[1].split()
        distance = len(curExList)
        foundT = False
        for t in triggerList:
          if t in curExList:
            tIndex = curExList.index(t)
            if s in self.slot2inverse: # reverse name and filler
              fIndex = curExList.index("<name>")
            else:
              fIndex = curExList.index("<filler>")
            curDistance = abs(tIndex - fIndex)
            foundT = True
            if curDistance < distance:
              distance = curDistance
        if foundT:
          invDist = 1.0 / distance
          self.logger.info("proximity: " + s + ": " + candidate[0] + ": " + candidate[1] + " --> " + str(invDist))
          self.globalSlot2fillerCandidatesAndConfidence[s].append([candidate[0], candidate[1], str(invDist), candidate[2], candidate[3], candidate[4], candidate[5]])
        else:
          self.logger.debug("proximity: did not find trigger for " + s + " in curEx " + candidate[1])

  def evaluateClassifiers(self):
    resultsSVM = doSVMparallel.classify(self.forClassifier, self.slot2inverse, self.svmVersion)
    resultsCNN = doCNNparallel.classify(self.forClassifier, self.slot2inverse, self.cnnVersion)
    resultsPattern = doPatternMatching.match(self.forClassifier, self.patternsAllSlots, self.slot2inverse)
    for slot in resultsSVM:
      if slot in self.slot2inverse:
        slot_eval = self.slot2inverse[slot]
      else:
        slot_eval = slot
      weightSVM = self.slot2weightsSVM[slot_eval]
      weightCNN = self.slot2weightsCNN[slot_eval]
      weightPAT = self.slot2weightsPAT[slot_eval]
      if not slot in self.globalSlot2fillerCandidatesAndConfidence:
        self.globalSlot2fillerCandidatesAndConfidence[slot] = []
      for rs, rc, rp in zip(resultsSVM[slot], resultsCNN[slot], resultsPattern[slot]):
        cmb = weightSVM * rs[2] + weightCNN * rc[2] + weightPAT * rp[2]
        self.logger.debug("classification: " + slot + ": " + rs[0] + ": " + rs[1] + " => " + str(rs[2]) + " ; " + str(rc[2]) + " ; " + str(rp[2]) + " => " + str(cmb))
        self.globalSlot2fillerCandidatesAndConfidence[slot].append([rs[0], rs[1], cmb, rs[3], rs[4], rs[5], rs[6]])

  def evaluatePatternMatcher(self):
    results = doPatternMatching.match(self.forPatternMatcher, self.patternsAllSlots, self.slot2inverse)
    for s in results:
      if not s in self.globalSlot2fillerCandidatesAndConfidence:
        self.globalSlot2fillerCandidatesAndConfidence[s] = []
      if not s in self.forClassifier:
        self.globalSlot2fillerCandidatesAndConfidence[s].extend(results[s])
    return results

  def resetGlobalConfidences(self):
    self.globalSlot2fillerCandidatesAndConfidence = {}

  def __init__(self, slot2proximity, slot2weightsSVM, slot2weightsCNN, slot2weightsPAT, patternsAllSlots, slot2inverse, svmVersion, cnnVersion, loggerMain):
    self.forRegex = {}
    self.forProximity = {}
    self.forClassifier = {}
    self.forPatternMatcher = {}

    self.slot2proximity = slot2proximity

    self.globalSlot2fillerCandidatesAndConfidence = {}

    self.slot2weightsSVM = slot2weightsSVM
    self.slot2weightsCNN = slot2weightsCNN
    self.slot2weightsPAT = slot2weightsPAT
    self.patternsAllSlots = patternsAllSlots

    self.slot2inverse = slot2inverse

    self.svmVersion = svmVersion
    self.cnnVersion = cnnVersion

    self.logger = loggerMain.getChild(__name__)
