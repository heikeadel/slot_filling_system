#!/usr/bin/python

#####################################
### CIS SLOT FILLING SYSTEM      ####
### 2014-2015                    ####
### Author: Heike Adel           ####
#####################################

import string
import re
import numpy
import logging
from sklearn.preprocessing import MultiLabelBinarizer

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

def readConfig(configfile):
  config = {}
  # read config file
  f = open(configfile, 'r')
  for line in f:
    if "#" == line[0]:
      continue # skip commentars
    line = line.strip()
    parts = line.split('=')
    name = parts[0]
    value = parts[1]
    config[name] = value
  f.close()

  if not "wordvectors" in config:
    logger.error("no word vector file specified in config")
    exit()

  if not "net" in config:
    logger.error("no file to save the network specified in config")
    exit()

  if not "hidden" in config:
    logger.warning("no hidden unit size specified in config. Setting to default value 200")
    config["hidden"] = 200

  if not "filtersize" in config:
    logger.warning("no filter size specified in config. Setting to default value 3")
    config["filtersize"] = 3

  if not "kmax" in config:
    logger.warning("no pool size specified for kmax pooling in config. Setting to default 3-max pooling")
    config["kmax"] = 3

  if not "nkerns" in config:
    logger.warning("no number of kernels specified in config. Setting to default value 100")
    config["nkerns"] = 100

  if not "contextsize" in config:
    logger.warning("no contextsize specified in config. Setting to default value 40")
    config["contextsize"] = 40

  return config

def readWordvectors(wordvectorfile):
  wordvectors = {}
  vectorsize = 0
  f = open(wordvectorfile, 'r')
  count = 0
  for line in f:
    if count == 0: # first line of word2vec output contains meta information
      count += 1
      continue
    parts = line.split()
    word = parts[0]
    parts.pop(0)
    wordvectors[word] = parts
    vectorsize = len(parts)
  f.close()
  return [wordvectors, vectorsize]

def getMatrixForContext(context, cap, fillerIndex, nameIndex, curLength, representationsize, contextsize, wordvectors, vectorsize):
    matrix = numpy.zeros(shape = (representationsize, contextsize))
    i = 0

    nextIndex = 0

    while i < len(context):
      word = context[i]
      nextIndex = 0
      # current word
      if word != "<empty>":
        if not word in wordvectors:
          word = "<unk>"
        curVector = wordvectors[word]
        for j in range(0, vectorsize):
          if j > len(curVector):
            logger.error("mismatch in word vector lengths: " + str(len(curVector)) + " vs " + vectorsize)
            exit()
          elem = float(curVector[j])
          matrix[j + nextIndex, i] = elem
      nextIndex += vectorsize

      # capitalization feature
      matrix[nextIndex, i] = float(cap[i])
      nextIndex += 1

      i += 1

    return matrix

def getThreeContextsAndLength(candidateAndFillerAndOffsetList, contextsize):
    inputList_a = []
    inputList_b = []
    inputList_c = []
    lengthList_a = []
    lengthList_b = []
    lengthList_c = []
    resultList = [] # only for training, otherwise empty
    nameBeforeFillerList = []
    logger.debug("using improved index computation for splitting")
    for line in candidateAndFillerAndOffsetList:
      if isinstance(line, list):
        wholeContext = line[1]
        posNeg = ""
      else:
        # comes from file, assuming the following format:
        # +/- : slot : query entity name : filler : proof sentence
        # alternative line separator: '::' instead of ':'
        line = line.strip()
        if re.search(r'^\S+ \: ', line):
          parts = line.split(' : ')
          wholeContext = " : ".join(parts[4:])
        else:
          parts = line.split(" :: ")
          wholeContext = " :: ".join(parts[4:])
        if parts[0] == '+':
          posNeg = 1
        elif parts[0] == '-':
          posNeg = 0
        else:
          posNeg = int(parts[0])
      contextWords = wholeContext.split()
      if not "<name>" in contextWords or not "<filler>" in contextWords:
        continue # skip example
      # improved index computation for splitting:
      # get all occurrences of <name> and <filler> and split where they are closest to each other
      # (idea: no <name> or <filler> tag in the middle context: keep middle context clean)
      fillerIndices = [i for i, x in enumerate(contextWords) if x == "<filler>"]
      nameIndices = [i for i, x in enumerate(contextWords) if x == "<name>"]
      fillerInd = -1
      nameInd = -1
      distanceNameFiller = len(contextWords)
      for fi in fillerIndices:
        for ni in nameIndices:
          distance = abs(ni - fi)
          if distance < distanceNameFiller:
            distanceNameFiller = distance
            nameInd = ni
            fillerInd = fi
      minIndex = 0
      maxIndex = 0
      if fillerInd < nameInd:
        nameBeforeFillerList.append(0)
        minIndex = fillerInd
        maxIndex = nameInd
      else:
        nameBeforeFillerList.append(1)
        maxIndex = fillerInd
        minIndex = nameInd

      contextWords_a = contextWords[0 : minIndex]
      contextWords_b = contextWords[minIndex + 1 : maxIndex]
      contextWords_c = contextWords[maxIndex + 1 :]
      myLength_a = min(contextsize, len(contextWords_a))
      myLength_a = max(1, myLength_a)
      myLength_b = min(contextsize, len(contextWords_b))
      myLength_b = max(1, myLength_b)
      myLength_c = min(contextsize, len(contextWords_c))
      myLength_c = max(1, myLength_c)
      # adjust context: left context
      while len(contextWords_a) < contextsize:
        ### pad only to the right
        contextWords_a.append("<empty>")
      while len(contextWords_a) > contextsize:
        ### remove items from the left
        contextWords_a.pop(0)

      # adjust context: middle context
      while len(contextWords_b) < contextsize:
        ### pad only to the right
        contextWords_b.append("<empty>")
      while len(contextWords_b) > contextsize:
        ### remove items from the middle
        indexToRemove = (len(contextWords_b) - 1) / 2
        contextWords_b.pop(indexToRemove)

      # adjust context: right context
      while len(contextWords_c) < contextsize:
        ### pad only to the right
        contextWords_c.append("<empty>")
      while len(contextWords_c) > contextsize:
        ### remove items from the right
        contextWords_c.pop(-1)

      inputList_a.append(contextWords_a)
      lengthList_a.append(myLength_a)
      inputList_b.append(contextWords_b)
      lengthList_b.append(myLength_b)
      inputList_c.append(contextWords_c)
      lengthList_c.append(myLength_c)
      resultList.append(posNeg)
    return [inputList_a, inputList_b, inputList_c, lengthList_a, lengthList_b, lengthList_c, nameBeforeFillerList, resultList]


def getThreeContextsAndLengthLc(candidateAndFillerAndOffsetList, contextsize):
    normal = getThreeContextsAndLength(candidateAndFillerAndOffsetList, contextsize)
    lc_a = []
    cap_a = []
    lc_b = []
    cap_b = []
    lc_c = []
    cap_c = []
    for a in normal[0]:
      tmp_a = []
      tmp_cap_a = []
      for part in a:
        if part[0].isupper():
          tmp_cap_a.append(1)
        else:
          tmp_cap_a.append(0)
        tmp_a.append(string.lower(part))
      lc_a.append(tmp_a)
      cap_a.append(tmp_cap_a)
    for b in normal[1]:
      tmp_b = []
      tmp_cap_b = []
      for part in b:
        if part[0].isupper():
          tmp_cap_b.append(1)
        else:
          tmp_cap_b.append(0)
        tmp_b.append(string.lower(part))
      lc_b.append(tmp_b)
      cap_b.append(tmp_cap_b)
    for c in normal[2]:
      tmp_c = []
      tmp_cap_c = []
      for part in c:
        if part[0].isupper():
          tmp_cap_c.append(1)
        else:
          tmp_cap_c.append(0)
        tmp_c.append(string.lower(part))
      lc_c.append(tmp_c)
      cap_c.append(tmp_cap_c)
    return [lc_a, lc_b, lc_c, cap_a, cap_b, cap_c, normal[3], normal[4], normal[5], normal[6], normal[7]]

def getInput(candidateAndFillerAndOffsetList, representationsize, contextsize, filtersize, wordvectors, vectorsize):
    inputListDev_a, inputListDev_b, inputListDev_c, capDev_a, capDev_b, capDev_c, lengthListDev_a, lengthListDev_b, lengthListDev_c, nameBeforeFillerListDev, resultListDev = getThreeContextsAndLengthLc(candidateAndFillerAndOffsetList, contextsize)

    numSamplesDev = len(inputListDev_a)
    if numSamplesDev == 0:
      logger.error("no dev examples for this slot: no testing possible")
      exit()

    inputMatrixDev_a = numpy.empty(shape = (numSamplesDev, representationsize * contextsize))
    inputMatrixDev_b = numpy.empty(shape = (numSamplesDev, representationsize * contextsize))
    inputMatrixDev_c = numpy.empty(shape = (numSamplesDev, representationsize * contextsize))
    inputFeaturesDev = numpy.empty(shape = (numSamplesDev, 1))
    for sample in range(0, numSamplesDev):
      contextA = inputListDev_a[sample]
      contextB = inputListDev_b[sample]
      contextC = inputListDev_c[sample]

      curLength_a = lengthListDev_a[sample]
      curLength_b = lengthListDev_b[sample]
      curLength_c = lengthListDev_c[sample]

      thisCapA = capDev_a[sample]
      thisCapB = capDev_b[sample]
      thisCapC = capDev_c[sample]

      nameBeforeFiller = nameBeforeFillerListDev[sample]

      lengthListDev_a[sample] = curLength_a + filtersize[1] / 2 * 2
      lengthListDev_b[sample] = curLength_b + filtersize[1] / 2 * 2
      lengthListDev_c[sample] = curLength_c + filtersize[1] / 2 * 2
      # pad with filtersize[1] / 2 values to the left and to the right with PADDING
      for dwin in range(filtersize[1] / 2):
        contextA.insert(0, "PADDING")
        contextA.insert(curLength_a + dwin + 1, "PADDING")
        contextB.insert(0, "PADDING")
        contextB.insert(curLength_b + dwin + 1, "PADDING")
        contextC.insert(0, "PADDING")
        contextC.insert(curLength_c + dwin + 1, "PADDING")
        thisCapA.insert(0, 0)
        thisCapA.insert(curLength_b + dwin + 1, 0)
        thisCapB.insert(0, 0)
        thisCapB.insert(curLength_b + dwin + 1, 0)
        thisCapC.insert(0, 0)
        thisCapC.insert(curLength_c + dwin + 1, 0)
      contextA = contextA[0:contextsize]
      contextB = contextB[0:contextsize]
      contextC = contextC[0:contextsize]
      thisCapA = thisCapA[0:contextsize]
      thisCapB = thisCapB[0:contextsize]
      thisCapC = thisCapC[0:contextsize]
      while not contextA[-filtersize[1]/2 + 1] in ["PADDING", "<empty>"]:
        contextA.pop(contextsize / 2)
        contextA.append("PADDING")
        thisCapA.pop(contextsize/2)
        thisCapA.append(0)
      while not contextB[-filtersize[1]/2 + 1] in ["PADDING", "<empty>"]:
        contextB.pop(contextsize / 2)
        contextB.append("PADDING")
        thisCapB.pop(contextsize / 2)
        thisCapB.append(0)
      while not contextC[-filtersize[1]/2 + 1] in ["PADDING", "<empty>"]:
        contextC.pop(contextsize / 2)
        contextC.append("PADDING")
        thisCapC.pop(contextsize / 2)
        thisCapC.append(0)

      nameIndex = curLength_a
      fillerIndex = curLength_a + curLength_b + 1
      if nameBeforeFiller == 0:
        nameIndex = curLength_a + curLength_b + 1
        fillerIndex = curLength_a

      matrixA = getMatrixForContext(contextA, thisCapA, fillerIndex, nameIndex, curLength_a, representationsize, contextsize, wordvectors, vectorsize)
      matrixA = numpy.reshape(matrixA, representationsize * contextsize)
      inputMatrixDev_a[sample,:] = matrixA

      nameIndex = -1
      fillerIndex = curLength_b
      if nameBeforeFiller == 0:
        nameIndex = curLength_b
        fillerIndex = -1

      matrixB = getMatrixForContext(contextB, thisCapB, fillerIndex, nameIndex, curLength_b, representationsize, contextsize, wordvectors, vectorsize)
      matrixB = numpy.reshape(matrixB, representationsize * contextsize)
      inputMatrixDev_b[sample,:] = matrixB

      nameIndex = - curLength_b - 2
      fillerIndex = -1
      if nameBeforeFiller == 0:
        nameIndex = -1
        fillerIndex = - curLength_b - 2

      matrixC = getMatrixForContext(contextC, thisCapC, fillerIndex, nameIndex, curLength_c, representationsize, contextsize, wordvectors, vectorsize)
      matrixC = numpy.reshape(matrixC, representationsize * contextsize)
      inputMatrixDev_c[sample,:] = matrixC

      matrixFeats = numpy.zeros(shape = (1, 1))
      matrixFeats[0][0] = nameBeforeFiller
      matrixFeats = numpy.reshape(matrixFeats, 1)
      inputFeaturesDev[sample,:] = matrixFeats

    return [inputMatrixDev_a, inputMatrixDev_b, inputMatrixDev_c, lengthListDev_a, lengthListDev_b, lengthListDev_c, inputFeaturesDev, resultListDev]


def getSlot2Types(filename="data/slots2types2015_41slots"):

    typeList = ["O", "PERSON", "LOCATION", "ORGANIZATION", "DATE", "NUMBER"]

    binarizer = MultiLabelBinarizer(classes=typeList)
    binarizer.fit([])

    slot2types = {}
    f = open(filename)
    for line in f:
      if re.search('^\#', line) or line.strip() == "":
        continue
      line = line.strip()
      parts = line.split()
      curSlot = parts[0]
      if "cities" in curSlot or "countries" in curSlot or "statesorprovinces" in curSlot:
        curSlot = re.sub('cities', 'locations', curSlot)
        curSlot = re.sub('countries', 'locations', curSlot)
        curSlot = re.sub('statesorprovinces', 'locations', curSlot)
      elif "city" in curSlot or "country" in curSlot or "stateorprovince" in curSlot:
        curSlot = re.sub('city', 'location', curSlot)
        curSlot = re.sub('country', 'location', curSlot)
        curSlot = re.sub('stateorprovince', 'location', curSlot)
      type1 = ""
      type2 = ""
      if "per:" in curSlot:
        type1 = ["PERSON"]
      elif "org:" in curSlot:
        type1 = ["ORGANIZATION"]
      type2 = []
      curTypes = parts[1].split(',')
      for t in curTypes:
        if t in typeList:
          type2.append(t)
      slot2types[curSlot] = (type1, type2)
    f.close()

    return slot2types, binarizer, len(typeList)

def getSlot2SingleType(filename="data/slots2types2015_41slots_mainNERtag"):

    typeList = ["O", "PERSON", "LOCATION", "ORGANIZATION", "DATE", "NUMBER"]

    binarizer = MultiLabelBinarizer(classes=typeList)
    binarizer.fit([])

    slot2types = {}
    f = open(filename)
    for line in f:
      if re.search('^\#', line) or line.strip() == "":
        continue
      line = line.strip()
      parts = line.split()
      curSlot = parts[0]
      if "cities" in curSlot or "countries" in curSlot or "statesorprovinces" in curSlot:
        curSlot = re.sub('cities', 'locations', curSlot)
        curSlot = re.sub('countries', 'locations', curSlot)
        curSlot = re.sub('statesorprovinces', 'locations', curSlot)
      elif "city" in curSlot or "country" in curSlot or "stateorprovince" in curSlot:
        curSlot = re.sub('city', 'location', curSlot)
        curSlot = re.sub('country', 'location', curSlot)
        curSlot = re.sub('stateorprovince', 'location', curSlot)
      type1 = ""
      type2 = ""
      if "per:" in curSlot:
        type1 = "PERSON"
      elif "org:" in curSlot:
        type1 = "ORGANIZATION"
      curTypes = parts[1].split(',')
      for t in curTypes:
        if t in typeList:
          type2 = t
      slot2types[curSlot] = (type1, type2)
    f.close()

    slot2typeVectors = {}
    for s in slot2types:
      type1bin = typeList.index(slot2types[s][0])
      type2bin = typeList.index(slot2types[s][1])
      slot2typeVectors[s] = (type1bin, type2bin)
    return slot2types, slot2typeVectors, len(typeList)

