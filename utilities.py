#!/usr/bin/python
# -*- coding: utf-8 -*-

#####################################
### CIS SLOT FILLING SYSTEM      ####
### 2014-2015                    ####
### Author: Heike Adel           ####
#####################################

import codecs, sys
reload(sys)
sys.setdefaultencoding('utf-8')
sys.stdout = codecs.getwriter('utf8')(sys.stdout)
sys.stderr = codecs.getwriter('utf8')(sys.stderr)
import string
import re
import os
import copy
import editdist
import unicodedata, string

import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
logger.setLevel(logging.DEBUG)

string2number = {'one' : '1', 'two' : '2', 'three' : '3', 'four' : '4', 'five' : '5', 'six' : '6',
                 'seven' : '7', 'eight' : '8', 'nine' : '9', 'ten' : '10', 'eleven' : '11', 'twelve' : '12'}

def remove_accents(data):
  dataU = data.decode('utf-8')
  return ''.join(x for x in unicodedata.normalize('NFKD', dataU) if x in string.ascii_letters).lower()

def cleanWord(word):
  word = word.encode('utf-8')
  word = re.sub(r'\`\`', '"', word, re.UNICODE)
  word = re.sub(r'\`', "'", word, re.UNICODE)
  word = re.sub(r'\xc2\x92\xc2\x94', "'''", word, re.UNICODE)
  word = re.sub(r"\'\'", '"', word, re.UNICODE)
  word = re.sub(r'”', '"', word, re.UNICODE)
  word = re.sub(r'\xe2\x80\x9c', '"', word, re.UNICODE)
  word = re.sub(r'\xe2\x80\x9d', '"', word, re.UNICODE)
  word = re.sub(r'\xe2\x80\x99', "'", word, re.UNICODE)
  word = re.sub(r'\xc2\x93', '"', word, re.UNICODE)
  word = re.sub(r'\xc2\x94', '"', word, re.UNICODE)
  word = re.sub(r'\xc2\x91', "'", word, re.UNICODE)
  word = re.sub(r'\xc2\xbb', '"', word, re.UNICODE)
  word = re.sub(r'&quot;', '"', word, re.UNICODE)
  word = re.sub(r'\&gt\;', '>', word, re.UNICODE)
  word = re.sub(r'\&lt\;', '<', word, re.UNICODE)
  word = re.sub(r'&amp;', '&', word, re.UNICODE)
  word = re.sub(r'\xc2\xa4', '$', word, re.UNICODE)
  word = re.sub(r'\xe2\x80\x94', '--', word, re.UNICODE)
  word = re.sub(r'\xc2\x97', "--", word, re.UNICODE)
  word = re.sub(r'\xc2\x96', "--", word, re.UNICODE)
  word = re.sub(r'\xe2\x80\x93', '--', word, re.UNICODE)
  word = re.sub(r'\xe2\x80\x95', '--', word, re.UNICODE)
  word = re.sub(r'\xc2\x92', "'", word, re.UNICODE)
  word = re.sub(r'\xc2\x85', '..', word, re.UNICODE)
  word = re.sub(r'\xe2\x80\x98', "'", word, re.UNICODE)
  word = re.sub(r'\-LRB\-', '(', word, re.UNICODE)
  word = re.sub(r'\-RRB\-', ')', word, re.UNICODE)
  word = re.sub(r'\-LCB\-', '{', word, re.UNICODE)
  word = re.sub(r'\-RCB\-', '}', word, re.UNICODE)
  word = re.sub(r'\-LSB\-', '[', word, re.UNICODE)
  word = re.sub(r'\-RSB\-', ']', word, re.UNICODE)
  word = re.sub(r'\xc2\x80', '$', word, re.UNICODE)
  word = re.sub(r'\xe2\x82\xac', '$', word, re.UNICODE)
  word = re.sub(r'\xe2\x80\xa6', '...', word, re.UNICODE)
  word = re.sub(r'\xc2\xa2', 'cents', word, re.UNICODE)
  word = re.sub(r'\xc2\xa3', '#', word, re.UNICODE) # Stanford CoreNLP does the same internally
  word = re.sub(r'\xc2\xbd', '1/2', word, re.UNICODE)
  word = re.sub(r'\xc2\xbe', '3/4', word, re.UNICODE)
  word = re.sub(r'\xc2\xbc', '1/4', word, re.UNICODE)
  word = re.sub(r'\xe2\x80\xba', "'", word, re.UNICODE)
  word = re.sub(r'\xe2\x80\xb9', "'", word, re.UNICODE)
  word = re.sub(r'\xc2\xab', '"', word, re.UNICODE)
  word = re.sub(r'\xe3\x80\x80', '', word, re.UNICODE)
  word = re.sub(r'\xe3\x80\x8c', '', word, re.UNICODE)
  word = re.sub(r'\xe3\x80\x8d', '', word, re.UNICODE)
  word = re.sub(r'\xe2\x80\xa6', '...', word, re.UNICODE)
  word = re.sub(r'\xc2\x8a', "'", word, re.UNICODE)
  word = re.sub(r'…', '...', word, re.UNICODE)
  word = re.sub(r'\-\_\-', '_', word, re.UNICODE)
  word = re.sub(r'\xe0\xb2[\x80-\xbf]', '-', word, re.UNICODE)
  word = re.sub(r'\xe0\xb3[\x80-\x8f]', '-', word, re.UNICODE)
  word = re.sub(r'\_+', '-_-', word, re.UNICODE)
  word = re.sub(r'a href', 'a-_-href', word, re.UNICODE)
  word = word.decode('utf-8')
  return word

def month2year(month):
  result = ""
  if month in string2number:
    month = string2number[month]
  if month.isdigit():
    result = str(int(month) / 12)
  return result

def tokenizeNames(nameListOrig, tmpfilename = "name"):
  # tokenize names to the same format as the texts will be tokenized
  curPwd = os.getcwd()
  nameList = []
  os.chdir("PATH/stanford-corenlp-full-2014-01-04") # replace with path to Stanford CoreNLP
  for n in nameListOrig:
    if re.search(ur'[^\w \.]', n, re.UNICODE) or "_" in n:
      logger.debug("found non-letter character in " + n)
      nameOut = open(tmpfilename + '.forTok', 'w')
      nameOut.write(n + "\n")
      nameOut.close()
      os.system("java -cp stanford-corenlp-3.3.1.jar:stanford-corenlp-3.3.1-models.jar:xom.jar:joda-time.jar:jollyday.jar:ejml-0.23.jar -Xmx3g edu.stanford.nlp.process.PTBTokenizer " + tmpfilename + ".forTok > " + tmpfilename + ".afterTok")
      nameIn = open(tmpfilename + '.afterTok', 'r')
      nameTok = ""
      for line in nameIn:
        line = line.strip()
        nameTok += line + " "
      nameTok = nameTok.strip()
      nameTok = re.sub(ur'( \.$)+', '', nameTok, re.UNICODE)
      nameList.append(nameTok)
      nameIn.close()
    else:
      nameList.append(n)
  os.chdir(curPwd)
  logger.debug("tokenized " + str(nameListOrig) + " to " + str(nameList))
  return nameList

def getsubidx(x, y):
    resultList = []
    l1, l2 = len(x), len(y)
    for i in range(l1):
        if x[i:i+l2] == y:
           resultList.append(i)
    return resultList

def levenshtein(a,b):
    "Calculates the Levenshtein distance between a and b."
    n, m = len(a), len(b)
    if n > m:
        # Make sure n <= m, to use O(min(n,m)) space
        a,b = b,a
        n,m = m,n

    current = range(n+1)
    for i in range(1,m+1):
        previous, current = current, [i]+[0]*n
        for j in range(1,n+1):
            add, delete = previous[j]+1, current[j-1]+1
            change = previous[j-1]
            if a[j-1] != b[i-1]:
                change = change + 1
            current[j] = min(add, delete, change)

    return current[n]

def getCapitalizationList(text):
  result = []
  for word in text.split():
    if word[0].isupper():
      result.append(1)
    else:
      result.append(0)
  return result

def compareNames(name, myText):
  myTextOrig = myText
  nameOrig = name
  nameIsAcronym = 0
  myTextOrigList = myText.split()
  capitalizationName = getCapitalizationList(name)
  capitalizationMyText = getCapitalizationList(myText)
  if name.isupper():
    nameIsAcronym = 1
  else:
    myText = string.lower(myText).strip()
    name = string.lower(name).strip()
  nameList = name.split()
  lengthName = len(nameList)
  myTextList = myText.split()
  lengthText = len(myTextList)
  resultHits = []
  for t in range(0, lengthText - lengthName + 1):
    testName = " ".join(myTextList[t:t+lengthName])
    if testName == name:
      # exact match
      if nameIsAcronym:
        testNameOrig = " ".join(myTextOrigList[t:t+lengthName])
        if testNameOrig.isupper(): # match only with acronyms!
          resultHits.append([t, t+lengthName])
      else:
        # compare uppercase letters only with upper case letters!
        noMatch = 0
        for ind, cn in enumerate(capitalizationName):
          if cn != capitalizationMyText[t + ind]:
            noMatch = 1
            break
        if noMatch == 1:
          logger.debug("no match: " + nameOrig + " <=> " + myTextOrig)
          continue
        resultHits.append([t, t+lengthName])
    else:
      # fuzzy match
      charactersName = list(name)
      charactersMyText = list(testName)
      maxDistance = max(1, len(charactersName) / 7) # just some heuristic
      if len(charactersName) < 2 or abs(len(charactersName) - len(charactersMyText)) > maxDistance:
        continue
      distanceNames = editdist.distance(name, testName)
      if distanceNames > maxDistance:
        continue
      else:
        if nameIsAcronym:
          testNameOrig = " ".join(myTextOrigList[t:t+lengthName])
          if testNameOrig.isupper(): # match only with acronyms!
            logger.info("found similar but not equal names: " + name + " - " + testNameOrig)
            resultHits.append([t, t+lengthName])
        else:
          if capitalizationName.count(1) == capitalizationMyText[t:t+lengthName].count(1):  
            # heuristic: in matched text must be the same number of uppercase letters as in name!
            logger.info("found similar but not equal names: " + name + " - " + testName)
            resultHits.append([t, t+lengthName])
          else:
            logger.debug("no match: " + nameOrig + " <=> " + myTextOrig)
            continue
  return resultHits

def compareNamesLc(name, myText):
  myTextOrig = myText
  nameOrig = name
  nameIsAcronym = 0
  myTextOrigList = myText.split()
  if name.isupper():
    nameIsAcronym = 1
  else:
    myText = string.lower(myText).strip()
    name = string.lower(name).strip()
  nameList = name.split()
  lengthName = len(nameList)
  myTextList = myText.split()
  lengthText = len(myTextList)
  resultHits = []
  for t in range(0, lengthText - lengthName + 1):
    testName = " ".join(myTextList[t:t+lengthName])
    if testName == name:
      # exact match
      if nameIsAcronym:
        testNameOrig = " ".join(myTextOrigList[t:t+lengthName])
        if testNameOrig.isupper(): # match only with acronyms!
          resultHits.append([t, t+lengthName])
      else:
        resultHits.append([t, t+lengthName])
    else:
      # fuzzy match
      charactersName = list(name)
      charactersMyText = list(testName)
      maxDistance = max(1, len(charactersName) / 7) # just some heuristic
      if len(charactersName) < 2 or abs(len(charactersName) - len(charactersMyText)) > maxDistance:
        continue
      distanceNames = editdist.distance(name, testName)
      if distanceNames > maxDistance:
        continue
      else:
        if nameIsAcronym: # only exact matches for acronyms
          testNameOrig = " ".join(myTextOrigList[t:t+lengthName])
          if testNameOrig.isupper(): # match only with acronyms!
            logger.info("found similar but not equal names: " + name + " - " + testNameOrig)
            resultHits.append([t, t+lengthName])
        else:
          logger.info("found similar but not equal names: " + name + " - " + testName)
          resultHits.append([t, t+lengthName])
  return resultHits

def compareNamesImprovedLc(name, myText):
  myTextOrig = myText
  nameOrig = name
  nameIsAcronym = 0
  myTextOrigList = myText.split()
  if name.isupper():
    nameIsAcronym = 1
  else:
    myText = string.lower(myText).strip()
    name = string.lower(name).strip()
  nameList = name.split()
  lengthName = len(nameList)
  myTextList = myText.split()
  lengthText = len(myTextList)
  resultHits = []
  for t in range(0, lengthText - lengthName + 1):
    testName = " ".join(myTextList[t:t+lengthName])
    if testName == name:
      # exact match
      if nameIsAcronym:
        testNameOrig = " ".join(myTextOrigList[t:t+lengthName])
        if testNameOrig.isupper(): # match only with acronyms!
          resultHits.append([t, t+lengthName])
      else:
        resultHits.append([t, t+lengthName])
    else:
      # fuzzy match
      charactersName = list(name)
      charactersMyText = list(testName)
      maxDistance = max(1, len(charactersName) / 7) # just some heuristic
      maxRelativeDistance = 0.4 # again some heuristic
      if len(charactersName) < 2 or abs(len(charactersName) - len(charactersMyText)) > maxDistance:
        continue
      distanceNames = editdist.distance(remove_accents(name), remove_accents(testName))
      relativeDistance = distanceNames * 1.0 / len(name)
      if distanceNames > maxDistance or relativeDistance > maxRelativeDistance:
        continue
      else:
        if hasDifferentAbbreviation(name, testName): # check if name and testName include abbreviations and don't match if those abbreviations are not equal!
          continue
        if nameIsAcronym:
          pass # no fuzzy match for acronyms!
        else:
            if name[0] != testName[0]:
              if (name[0] in ['k','K','c','C'] and testName[0] in ['k','K','c','C']) or (name[0] in ['i','I','y','Y'] and testName[0] in ['i','I','y','Y']) or (name[0] in ['f','F','p','P'] and testName[0] in ['f','F','p','P']): # could still be spelling variations
                logger.info("found similar but not equal names: " + name + " - " + testName)
                resultHits.append([t, t+lengthName])
              else:
                logger.debug("found similar name but will not append it because first characters do not match: " + name + " - " + testName)
            else:
              logger.info("found similar but not equal names: " + name + " - " + testName)
              resultHits.append([t, t+lengthName])
  return resultHits


def compareNamesImproved(name, myText):
  myTextOrig = myText
  nameOrig = name
  nameIsAcronym = 0
  myTextOrigList = myText.split()
  capitalizationName = getCapitalizationList(name)
  capitalizationMyText = getCapitalizationList(myText)
  if name.isupper():
    nameIsAcronym = 1
  else:
    myText = string.lower(myText).strip()
    name = string.lower(name).strip()
  nameList = name.split()
  lengthName = len(nameList)
  myTextList = myText.split()
  lengthText = len(myTextList)
  resultHits = []
  for t in range(0, lengthText - lengthName + 1):
    testName = " ".join(myTextList[t:t+lengthName])
    if testName == name:
      # exact match
      if nameIsAcronym:
        testNameOrig = " ".join(myTextOrigList[t:t+lengthName])
        if testNameOrig.isupper(): # match only with acronyms!
          resultHits.append([t, t+lengthName])
      else:
        # compare uppercase letters only with upper case letters!
        noMatch = 0
        for ind, cn in enumerate(capitalizationName):
          if cn != capitalizationMyText[t + ind]:
            noMatch = 1
            break
        if noMatch == 1:
          continue
        resultHits.append([t, t+lengthName])
    else:
      # fuzzy match
      # check if name and testName include abbreviations and don't match if those abbreviations are not equal!
      if hasDifferentAbbreviation(name, testName):
        continue
      charactersName = list(name)
      charactersMyText = list(testName)
      maxDistance = max(1, len(charactersName) / 7) # just some heuristic
      maxRelativeDistance = 0.4 # again some heuristic
      if len(charactersName) < 2 or abs(len(charactersName) - len(charactersMyText)) > maxDistance:
        continue
      distanceNames = editdist.distance(remove_accents(name), remove_accents(testName))
      relativeDistance = distanceNames * 1.0 / len(name)
      if distanceNames > maxDistance or relativeDistance > maxRelativeDistance:
        continue
      else:
        if nameIsAcronym:
          pass # no fuzzy match for acronyms!
        else:
          if capitalizationName.count(1) == capitalizationMyText[t:t+lengthName].count(1):
            # heuristic: in matched text must be the same number of uppercase letters as in name!
            if name[0] != testName[0]:
              if (name[0] in ['k','K','c','C'] and testName[0] in ['k','K','c','C']) or (name[0] in ['i','I','y','Y'] and testName[0] in ['i','I','y','Y']) or (name[0] in ['f','F','p','P'] and testName[0] in ['f','F','p','P']): # could still be spelling variations
                logger.info("found similar but not equal names: " + name + " - " + testName)
                resultHits.append([t, t+lengthName])
              else:
                logger.debug("found similar name but will not append it because first characters do not match: " + name + " - " + testName)
            else:
              logger.info("found similar but not equal names: " + name + " - " + testName)
              resultHits.append([t, t+lengthName])
          else:
            continue
  return resultHits

def hasDifferentAbbreviation(name, testName):
  if re.search(ur'\S+\.', name, re.UNICODE) and re.search(ur'\S+\.', testName, re.UNICODE):
    nameList = name.split()
    testNameList = testName.split()
    if len(nameList) != len(testNameList):
      logger.error("length of name list and test name list are different")
      return 0
    for i in range(len(nameList)):
      if "." in nameList[i] and "." in testNameList[i]:
        if nameList[i] != testNameList[i]:
          return 1
    return 0
  else: 
    return 0  

def updateBestIndices(bestIndices, hits):
  for hit in hits:
    start = hit[0]
    end = hit[1]
    foundRange = 0
    for hitInd, h in enumerate(bestIndices):
      # test whether there is an overlap at all
      if (start <= h[0] and end >= h[0]) or (start <= h[1] and end >= h[1]) or (start <= h[0] and end >= h[1]) or (start >= h[0] and end <= h[1]):
        foundRange = 1
        # test whether the current indices are better than the old ones
        if start <= h[0] and end >= h[1]:
          bestIndices[hitInd] = [start, end]
        elif start > h[0] and start < h[1] and end >= h[1]: # extend to the right
          bestIndices[hitInd] = [h[0], end]
        elif start <= h[0] and end < h[1] and end >= h[0]: # extend to the left
          bestIndices[hitInd] = [start, h[1]]

    if foundRange == 0:
      bestIndices.append([start, end]) # found additional occurrence of name

def compareNamesFuzzier(name, myText):
  myTextOrig = myText
  nameOrig = name
  nameIsAcronym = 0
  myTextOrigList = myText.split()
  capitalizationName = getCapitalizationList(name)
  capitalizationMyText = getCapitalizationList(myText)
  if name.isupper():
    nameIsAcronym = 1
  else:
    myText = string.lower(myText).strip()
    name = string.lower(name).strip()
  nameList = name.split()
  lengthName = len(nameList)
  myTextList = myText.split()
  lengthText = len(myTextList)
  for t in range(0, lengthText - lengthName + 1):
    testName = " ".join(myTextList[t:t+lengthName])
    if testName == name:
      # exact match
      if nameIsAcronym:
        testNameOrig = " ".join(myTextOrigList[t:t+lengthName])
        if testNameOrig.isupper(): # match only with acronyms!
          return [t, t+lengthName]
      else:
        # compare uppercase letters only with upper case letters!
        noMatch = 0
        for ind, cn in enumerate(capitalizationName):
          if cn != capitalizationMyText[t + ind]:
            noMatch = 1
            break
        if noMatch == 1:
          logger.debug("no match: " + nameOrig + " <=> " + myTextOrig)
          continue
        return [t, t+lengthName]
    else:
      # fuzzy match
      charactersName = list(name)
      charactersMyText = list(testName)
      maxDistance = max(1, len(charactersName) / 4) # just some heuristic
      if abs(len(charactersName) - len(charactersMyText)) > maxDistance:
        continue
      distanceNames = editdist.distance(name, testName)
      if distanceNames > maxDistance:
        continue
      else:
        if nameIsAcronym:
          testNameOrig = " ".join(myTextOrigList[t:t+lengthName])
          if testNameOrig.isupper(): # match only with acronyms!
            logger.info("found similar but not equal names: " + name + " - " + testNameOrig)
            return [t, t+lengthName]
        else:
          if capitalizationName.count(1) == capitalizationMyText[t:t+lengthName].count(1):
            # heuristic: in matched text must be the same number of uppercase letters as in name!
            logger.info("found similar but not equal names: " + name + " - " + testName)
            return [t, t+lengthName]
          else:
            logger.debug("no match: " + nameOrig + " <=> " + myTextOrig)
            continue
  return []


def matchPattern(candidate, patterns):
  candidate = string.lower(candidate)
  for p in patterns:
    if p.match(candidate):
      return 1
  return 0

def fillerEqualsName(nameOffsets, fillerOffset):
    fillerOffset = fillerOffset.strip(",")
    foParts = fillerOffset.split(',')
    for no in nameOffsets:
      no = no.strip()
      noParts = no.split(',')
      for fop in foParts:
        if fop in noParts:
          return 1
    return 0

def getOffsetsForRegex(thEscaped, curEx):
    thOffsets = [(match.start(), match.end()) for match in re.finditer(thEscaped, curEx, re.UNICODE)]
    resultList = []
    # map character index to word index
    for to in thOffsets:
      thStart = to[0]
      thEnd = to[1]
      thStartInd = -1
      thEndInd = -1
      curExList = curEx.split()
      chInd = 0
      foundStart = 0
      for wInd, w in enumerate(curExList):
        if foundStart == 1 and chInd >= thEnd:
          thEndInd = wInd
          break
        elif foundStart == 0 and (chInd >= thStart or chInd + len(w) >= thStart):
          thStartInd = wInd
          foundStart = 1
        chInd += len(w) + 1
      if foundStart == 1 and thEndInd == -1:
        # match until last character
        thEndInd = len(curEx)

      if thStartInd != -1:
        resultList.append([thStartInd, thEndInd])
    return resultList

def replaceOffsetListWithTag(curExIn, curOffsetListIn, offsetsToReplaceName, offsetsToReplaceFiller):
    curOffsetList = copy.deepcopy(curOffsetListIn)
    curExList = curExIn.split()
    # replace name with <name> tag
    for oneReplacement in offsetsToReplaceName: # there can be more than one <name> in sentence!
      if len(oneReplacement) > 1:
        noParts = oneReplacement
        if int(noParts[0]) >= int(curOffsetList[0]) and int(noParts[-1]) <= int(curOffsetList[-1]):
          # name is in curEx!
          if not noParts[0] in curOffsetList or not noParts[-1] in curOffsetList:
            return "", ""
          nameIndexFirst = curOffsetList.index(noParts[0])
          nameIndexLast = curOffsetList.index(noParts[-1])
          curExList[nameIndexFirst] = "<name>"
          toPop = nameIndexFirst + 1
          for nameInd in range(nameIndexFirst + 1, nameIndexLast + 1):
            curExList.pop(toPop)
            curOffsetList.pop(toPop)
      else:
        no = oneReplacement[0]
        if int(no) >= int(curOffsetList[0]) and int(no) <= int(curOffsetList[-1]):
          # name is in curEx!
          if not no in curOffsetList:
            return "", ""
          nameIndex = curOffsetList.index(no)
          curExList[nameIndex] = "<name>"

    # replace filler with <filler> tag
    if len(offsetsToReplaceFiller) > 1:
      noParts = offsetsToReplaceFiller
      if int(noParts[0]) >= int(curOffsetList[0]) and int(noParts[-1]) <= int(curOffsetList[-1]):
        # filler is in curEx!
        if not noParts[0] in curOffsetList or not noParts[-1] in curOffsetList:
          return "", ""
        fillerIndexFirst = curOffsetList.index(noParts[0])
        fillerIndexLast = curOffsetList.index(noParts[-1])
        curExList[fillerIndexFirst] = "<filler>"
        toPop = fillerIndexFirst + 1
        for fillerInd in range(fillerIndexFirst + 1, fillerIndexLast + 1):
          curExList.pop(toPop)
          curOffsetList.pop(toPop)
    else:
      fo = offsetsToReplaceFiller[0]
      if int(fo) >= int(curOffsetList[0]) and int(fo) <= int(curOffsetList[-1]):
        # filler is in curEx!
        if not fo in curOffsetList:
          return "", ""
        fillerIndex = curOffsetList.index(fo)
        curExList[fillerIndex] = "<filler>"
    curEx = " ".join(curExList)
    curOffsets = " ".join(curOffsetList)

    return curEx, curOffsets

def compareOffsets(slot, curFiller, curFillerOffset, referenceFillerList):
    curFillerOffsetList = curFillerOffset.split(",")
    curFillerList = curFiller.split()
    hypostart = int(curFillerOffsetList[0])
    hypoend = int(curFillerOffsetList[-1])
    lengthEnd = len(curFillerList[-1])
    endInd = len(curFillerOffsetList)-2
    while endInd >= 0:
      if int(curFillerOffsetList[endInd]) == hypoend:
        lengthEnd += len(curFillerList[endInd])
      else:
        break
      endInd -= 1
    hypoend += lengthEnd - 1 # first character is already counted!
    if hypostart > hypoend:
      logger.debug("filler: " + curFiller + " with length " + str(lengthEnd))
      logger.debug("ERROR: got hypostart = " + str(hypostart) + " and hypoend = " + str(hypoend))
    i = 0
    while i < len(referenceFillerList):
      referencestart = int(referenceFillerList[i][1])
      referenceend = int(referenceFillerList[i][2])
      referenceslot = referenceFillerList[i][0]
      if "cit" in referenceslot or "countr" in referenceslot or "province" in referenceslot:
          referenceslot = re.sub(ur'city', 'location', referenceslot, re.UNICODE)
          referenceslot = re.sub(ur'country', 'location', referenceslot, re.UNICODE)
          referenceslot = re.sub(ur'statesorprovinces', 'locations', referenceslot, re.UNICODE)
          referenceslot = re.sub(ur'cities', 'locations', referenceslot, re.UNICODE)
          referenceslot = re.sub(ur'countries', 'locations', referenceslot, re.UNICODE)
          referenceslot = re.sub(ur'stateorprovince', 'location', referenceslot, re.UNICODE)
      if slot == referenceslot:
        if referencestart > referenceend:
          logger.error("got referencestart = " + str(referencestart) + " and referenceend = " + str(referenceend))
        if hypostart == referencestart:
          if hypoend == referenceend or abs(hypoend - referenceend) == 1:
            # totally correct
            return [1, i]
          elif hypoend < referenceend:
            # too short
            return [2, i]
          else:
            # too long
            return [3, i]
        elif hypostart < referencestart:
          if hypoend < referencestart:
            i += 1
            continue
          if hypoend >= referenceend:
            # too long
            return [3, i]
          else:
            # partly overlapping
            return [5, i]
        else:
          if hypostart > referenceend:
            i += 1
            continue
          if hypoend <= referenceend:
            # too short
            return [2, i]
          else:
            # # partly overlapping
            return [5, i]
      i += 1
    return [-1]


def isNameInSentenceOffsets(nameOffsets, curOffsets):
    nameFound = 0
    for occ in nameOffsets:
      aliasOffs = int(occ.split(',')[0])
      if aliasOffs < int(curOffsets[0]) or aliasOffs > int(curOffsets[-1]):
        pass
      else:
        nameFound = 1
        break
    return nameFound

def isSentenceAdditional(additionalNameOffsetsFlattened, curOffsets):
    nameFound = 0
    for occ in additionalNameOffsetsFlattened:
      aliasOffsStart = int(occ.split(',')[0])
      aliasOffsEnd = int(occ.split(',')[-1])
      if (aliasOffsStart < int(curOffsets[0]) and aliasOffsEnd < int(curOffsets[0])) or (aliasOffsStart > int(curOffsets[-1]) and aliasOffsEnd > int(curOffsets[-1])):
        pass
      else:
        nameFound = 1
        break
    return nameFound

def normalizeDate(text):
  month = "XX"
  year = "XXXX"
  day = "XX"
  year = re.sub(ur'.*(\d{4}).*', '\\1', text, re.UNICODE)
  try:
    yearNumber = int(year)
  except ValueError:
    year = "XXXX"
  text = re.sub(ur'\d{4}', '', text, re.UNICODE)
  if re.search(ur'[Jj]anuary', text, re.UNICODE):
    month = "01"
  elif re.search(ur'[Ff]ebruary', text, re.UNICODE):
    month = "02"
  elif re.search(ur'[Mm]arch', text, re.UNICODE):
    month = "03"
  elif re.search(ur'[Aa]pril', text, re.UNICODE):
    month = "04"
  elif re.search(ur'[Mm]ay', text, re.UNICODE):
    month = "05"
  elif re.search(ur'[Jj]une', text, re.UNICODE):
    month = "06"
  elif re.search(ur'[Jj]uly', text, re.UNICODE):
    month = "07"
  elif re.search(ur'[Aa]ugust', text, re.UNICODE):
    month = "08"
  elif re.search(ur'[Ss]eptember', text, re.UNICODE):
    month = "09"
  elif re.search(ur'[Oo]ctober', text, re.UNICODE):
    month = "10"
  elif re.search(ur'[Nn]ovember', text, re.UNICODE):
    month = "11"
  elif re.search(ur'[Dd]ecember', text, re.UNICODE):
    month = "12"
  day = re.sub(ur'^.* (\d\d)[ \,].*$', '\\1', text, re.UNICODE)
  if day == "XX" or day == "":
    day = re.sub(ur'.* (\d)[ \,].*', '\\1', text, re.UNICODE)
  try:
    dayNumber = int(day)
  except ValueError:
    day = "XX"
  normed = year + "-" + month + "-" + day
  return normed

def getFillerType(slot):
    if slot in ["per:children", "per:other_family", "per:parents", "per:siblings", "per:spouse", "org:employees_or_members", "gpe:employees_or_members", "org:students", "gpe:births_in_city", "gpe:births_in_stateorprovince", "gpe:births_in_country", "gpe:residents_of_city", "gpe:residents_of_stateorprovince", "gpe:residents_of_country", "gpe:deaths_in_city", "gpe:deaths_in_stateorprovince", "gpe:deaths_in_country", "org:top_members_employees"]:
      return "PER"
    elif slot in ["per:schools_attended", "per:holds_shares_in", "org:holds_shares_in", "gpe:holds_shares_in", "per:organizations_founded", "org:organizations_founded", "gpe:organizations_founded", "per:top_member_employee_of", "org:member_of", "gpe:member_of", "org:subsidiaries", "gpe:subsidiaries", "gpe:headquarters_in_city", "gpe:headquarters_in_stateorprovince", "gpe:headquarters_in_country"]:
      return "ORG"
    elif slot in ["per:location_of_birth", "per:location_of_death", "per:locations_of_residence", "org:location_of_headquarters", "per:city_of_birth", "per:stateorprovince_of_birth", "per:country_of_birth", "per:cities_of_residence", "per:statesorprovinces_of_residence", "per:countries_of_residence", "per:city_of_death", "per:stateorprovince_of_death", "per:country_of_death", "org:city_of_headquarters", "org:stateorprovince_of_headquarters", "org:country_of_headquarters"]:
      return "GPE"
    elif slot in ["per:alternate_names", "per:date_of_birth", "per:age", "per:origin", "per:date_of_death", "per:cause_of_death", "per:title", "per:religion", "per:charges", "org:alternate_names", "org:political_religious_affiliation", "org:number_of_employees_members", "org:date_founded", "org:date_dissolved", "org:website"]:
      return "STRING"
    else:
      logger.error("could not find slot " + slot + " in slotlists for filler types")
      return "ORG"

def ner2fillerType(ner):
    if ner == "PERSON":
      return "PER"
    if ner == "LOCATION" or ner == "MISC":
      return "GPE"
    if ner == "ORGANIZATION":
      return "ORG"
    return "STRING"
