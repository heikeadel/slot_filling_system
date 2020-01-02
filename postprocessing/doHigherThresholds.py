#!/usr/bin/python

#####################################
### CIS SLOT FILLING SYSTEM      ####
### 2014-2015                    ####
### Author: Heike Adel           ####
#####################################

import re
import sys
from doLoad import getNicknames, readSlotThresholdFile, getSlot2Inverse

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

if len(sys.argv) != 3:
  logger.error("please pass the results before higher thresholds and the thresholdfile as input parameter")
  exit()

def processLocation(slot):
  if "countr" in slot:
    slot = re.sub(r'countries', 'locations', slot)
    slot = re.sub(r'country', 'location', slot)
  elif "cit" in slot:
    slot = re.sub(r'cities', 'locations', slot)
    slot = re.sub(r'city', 'location', slot)
  elif "state" in slot:
    slot = re.sub(r'statesorprovinces', 'locations', slot)
    slot = re.sub(r'stateorprovince', 'location', slot)
  return slot

outputfile = sys.argv[1]
thresholdfile = sys.argv[2]

# read thresholds and make them higher
slot2higherThresholds = {} # store floats here!
slot2threshold = readSlotThresholdFile(thresholdfile)
slot2inverse = getSlot2Inverse()
for s in slot2threshold:
  slot2higherThresholds[s] = min(float(slot2threshold[s][0]) + 0.1, 0.9)

# read original output
f = open(outputfile, 'r')
out = open(outputfile + ".higherThresholds", 'w')
for line in f:
  line = line.strip()
  parts = line.split('\t')
  queryId = parts[0]
  slot = parts[1]
  if slot in slot2inverse:
    s_eval = slot2inverse[slot]
  else:
    s_eval = slot
  s_eval = processLocation(s_eval)
  if not s_eval in slot2higherThresholds:
    threshold = 0.8
  else:
    threshold = slot2higherThresholds[s_eval]
  if s_eval == "per:other_family":
    threshold = 0.4
  elif s_eval == "per:charges":
    threshold = 0.7
  confidence = float(parts[-1])
  if float(confidence) > float(threshold):
    out.write(line + "\n")
f.close()
out.close()
