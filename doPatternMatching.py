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
import re

import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
logger.setLevel(logging.INFO)

def getMatch(candidate, patterns):
  candidate = candidate.lower()
  maxMatch = 0.0
  for p in patterns:
    if p.match(candidate):
      logger.debug("found matching candidate: " + candidate + "; reg ex: " + p.pattern)
      maxMatch = 1.0
      return maxMatch
  return maxMatch


def match(slot2candidates, patternsPerSlot, slot2inverse):
  slot2candidatesAndFillersAndConfidence = {}
  for slot in slot2candidates:
    logger.info("Pattern matching: " + slot)
    patterns = []
    if slot in slot2inverse:
      slot_eval = slot2inverse[slot]
      logger.info("using patterns of slot: " + slot_eval)
    else:
      slot_eval = slot
    if slot_eval in patternsPerSlot:
      patterns = patternsPerSlot[slot_eval]
    candidateAndFillerAndOffsetList = slot2candidates[slot]
    if len(candidateAndFillerAndOffsetList) == 0:
      continue

    for cf in candidateAndFillerAndOffsetList:
      c = cf[1]
      if slot in slot2inverse:
        # reverse name and filler
        c_tmp = re.sub(ur' \<name\> ', ' NAME ', ' ' + c + ' ', re.UNICODE)
        c_tmp = re.sub(ur' \<filler\> ', ' <name> ', c_tmp, re.UNICODE)
        c_tmp = re.sub(ur' NAME ', ' <filler> ', c_tmp, re.UNICODE)
        c = c_tmp.strip()
      # match c against patterns for slot
      matchResult = getMatch(c, patterns)
      if not slot in slot2candidatesAndFillersAndConfidence:
        slot2candidatesAndFillersAndConfidence[slot] = []
      slot2candidatesAndFillersAndConfidence[slot].append([cf[0], cf[1], matchResult, cf[2], cf[3], cf[4], cf[5]])
  return slot2candidatesAndFillersAndConfidence
