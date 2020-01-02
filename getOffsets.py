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
import re
from utilities import cleanWord
import os
import io
import gzip

import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
logger.setLevel(logging.DEBUG)

def using_split(line):
    words = line.split()
    offsets = []
    running_offset = 0
    for word in words:
      word_offset = line.index(word, running_offset)
      word_len = len(word)
      running_offset = word_offset + word_len
      offsets.append([word, word_offset, running_offset])
    return offsets

def getWords(docId, docPath):
  results = []
  if ".gz" in docPath:
    f = os.popen("zcat " + docPath, 'r')
  else:
    f = open(docPath)
  startOfDocument = ""
  readDoc = 0
  for line in f:
    startOfDocument += line
    if 'id="' + docId + '"' in line or 'DOCID> ' + docId in line: # assumption: docId appears only once in document!
      readDoc = 1
      results.extend(startOfDocument.split())
    if readDoc == 1:
      line = re.sub(r'\xc2\xa0', ' ', line)
      line = re.sub(r'\xe2\x80\x82', ' ', line)
      line = re.sub(r'\xe2\x80\x85', ' ', line)
      line = re.sub(r'\xe2\x80\x89', ' ', line)
      # the following characters are untokenizable by Stanford CoreNLP and are skipped in counting offsets
      line = re.sub(r'\xc2\x95', '', line)
      line = re.sub(r'\xe2\x80\x83', '', line)
      line = re.sub(r'\xc2\x99', '', line)
      line = re.sub(r'\xc2\x9e', '', line)
      line = re.sub(r'\xc2\x9c', '', line)
      line = re.sub(r'\xc2\x82', '', line)
      line = re.sub(r'\xc2\x98', '', line)
      line = re.sub(r'\xc2\xad', '', line)
      line = re.sub(r'\xef\x81\x82', '', line)
      line = re.sub(r'\xef\xbf\xbd', '', line)
      line = re.sub(r'\xe2\x80\xa8', '', line)
      results.extend(line.split())
    if "</DOC>" in line or "</doc>" in line:
      if readDoc == 1:
        results.extend(line.split())
        readDoc = 0
        continue
      else:
        startOfDocument = ""
  f.close()
  return results

def getOffsets(docId, docPath):
  if ".gz" in docPath:
    g = gzip.GzipFile(docPath)
    f = io.TextIOWrapper(io.BufferedReader(g), encoding='utf-8')
  else:
    f = io.open(docPath, encoding='utf-8')
  lastOffset = 0
  readDoc = 0
  results = []
  startOfDocument = ""
  for line in f:
    if ("<doc" in line or "<DOC" in line) and not "id" in line and not "ID" in line:
      startOfDocument += line
    if 'id="' + docId + '"' in line or 'DOCID> ' + docId in line:
      readDoc = 1
      # get offsets for start of document:
      if startOfDocument != "":
        result = using_split(startOfDocument)
        for r in range(len(result)):
          result[r][1] += lastOffset
          result[r][2] += lastOffset
        if len(result) > 0:
          lastOffset = result[-1][2] + 1
          results.extend(result)
    if readDoc == 1:
      line = line.encode('utf-8')
      line = re.sub(ur'\xc2\xa0', ' ', line, re.UNICODE)
      line = re.sub(ur'\xe2\x80\x82', ' ', line, re.UNICODE)
      line = re.sub(ur'\xe2\x80\x85', ' ', line, re.UNICODE)
      line = re.sub(ur'\xe2\x80\x89', ' ', line, re.UNICODE)
      # the following characters are untokenizable by Stanford CoreNLP and are skipped in counting offsets
      line = re.sub(ur'\xf4\x80\x80\x85', ' ', line, re.UNICODE)
      line = re.sub(ur'\xf4\x80\x80\x86', ' ', line, re.UNICODE)
      line = re.sub(ur'\xef\x9d\x8b', ' ', line, re.UNICODE)
      line = re.sub(ur'\xc2\x95', ' ', line, re.UNICODE)
      line = re.sub(ur'\xc2\x89', ' ', line, re.UNICODE)
      line = re.sub(ur'\xce\x80', ' ', line, re.UNICODE)
      line = re.sub(ur'\xc2\x8e', ' ', line, re.UNICODE)
      line = re.sub(ur'\xc2\x8b', ' ', line, re.UNICODE)
      line = re.sub(ur'\xc2\x81', ' ', line, re.UNICODE)
      line = re.sub(ur'\xc2\x9f', ' ', line, re.UNICODE)
      line = re.sub(ur'\xc2\x9d', ' ', line, re.UNICODE)
      line = re.sub(ur'\xc2\x88', ' ', line, re.UNICODE)
      line = re.sub(ur'\xc2\x91', "'", line, re.UNICODE)
      line = re.sub(ur'\xc2\x92', "'", line, re.UNICODE)
      line = re.sub(ur'\xc2\x9a', ' ', line, re.UNICODE)
      line = re.sub(ur'\xc2\x83', ' ', line, re.UNICODE)
      line = re.sub(ur'\xc2\x84', ' ', line, re.UNICODE)
      line = re.sub(ur'\xc2\x8c', ' ', line, re.UNICODE)
      line = re.sub(ur'\xc2\x86', ' ', line, re.UNICODE)
      line = re.sub(ur'\xc2\x8a', ' ', line, re.UNICODE)
      line = re.sub(ur'\xc2\x8f', ' ', line, re.UNICODE)
      line = re.sub(ur'\xc2\x8d', ' ', line, re.UNICODE)
      line = re.sub(ur'\xc2\x90', ' ', line, re.UNICODE)
      line = re.sub(ur'\xc2\x9b', ' ', line, re.UNICODE)
      line = re.sub(ur'\xc2\x87', ' ', line, re.UNICODE)
      line = re.sub(ur'\xe2\x80\x83', ' ', line, re.UNICODE)
      line = re.sub(ur'\xc2\x99', ' ', line, re.UNICODE)
      line = re.sub(ur'\xc2\x9e', ' ', line, re.UNICODE)
      line = re.sub(ur'\xc2\x9c', ' ', line, re.UNICODE)
      line = re.sub(ur'\xc2\x82', ' ', line, re.UNICODE)
      line = re.sub(ur'\xc2\x98', ' ', line, re.UNICODE)
      line = re.sub(ur'\xc2\x85', '…', line, re.UNICODE)
      line = re.sub(ur'\xc2\xad', '', line, re.UNICODE)
      line = re.sub(ur'\xe3\x80\x8a', '', line, re.UNICODE)
      line = re.sub(ur'\xe3\x80\x8b', '', line, re.UNICODE)
      line = re.sub(ur'\xe2\x80\xaa', '', line, re.UNICODE)
      line = re.sub(ur'\xe2\x80\x8e', '', line, re.UNICODE)
      line = re.sub(ur'\xe2\x80\xac', '', line, re.UNICODE)
      line = re.sub(ur'\xc3\x99\xc2\x8a\xc3\x99', '\xc3\x99 \xc3\x99', line, re.UNICODE)
      line = re.sub(ur'\xef\x81\x82', ' ', line, re.UNICODE)
      line = re.sub(ur'\xef\xbf\xbd', ' ', line, re.UNICODE)
      line = re.sub(ur'\xe2\x80\xa8', ' ', line, re.UNICODE)
      line = re.sub(ur'\xc3\x83\-re', '\xc3\x83re', line, re.UNICODE)
      line = re.sub(ur'C\-\+\+', 'C ++', line, re.UNICODE)
      line = line.decode('utf-8')
      result = using_split(line)
      for r in range(len(result)):
        result[r][1] += lastOffset
        result[r][2] += lastOffset
      if len(result) > 0:
        lastOffset = result[-1][2] + 1
        if line.strip() != '':
          results.extend(result)
      else:
        lastOffset += 1 # inconsistent offsets in queries!
    if "</DOC>" in line or "</doc>" in line:
      if readDoc == 1:
        readDoc = 0
        break
      else:
        startOfDocument = ""
  f.close()

  return results

def correctOffsets(docid, docpath, textPerLine, offsetsPerLine, offset2NormNer, additionalNameOffsets, nameOffsets, origNameOffsets, additionalCorefFillers = {}):
  usingOldOffsets = False
  pythonOffsetResults = getOffsets(docid, docpath)
  indOuterList = 0
  indCoreNLP = 0
  indPython = 0
  if len(offsetsPerLine) == 0:
    logger.error("length offsets per line is 0")
    return [[], {}, [], [], []]
  curOffsetList = offsetsPerLine[indOuterList].split()
  adjustTerm = 0
  newOffsetsPerLine = []
  newOffset2NormNer = {}
  newAdditionalNameOffsets = []
  newNameOffsets = []
  newOrigNameOffsets = []
  newAdditionalCorefFillers = {}
  curLineOffsets = ""
  prevPythonWord = ""
  prevPythonWordCleaned = ""
  mappingOld2New = {}
  while indPython < len(pythonOffsetResults):
    if indCoreNLP >= len(curOffsetList):
      # store updated offsets and get list from next line
      if len(curOffsetList) != len(curLineOffsets.strip().split()):
        logger.error("different offset numbers of line " + textPerLine[indOuterList])
      else:
        newOffsetsPerLine.append(curLineOffsets.strip())
      indOuterList += 1
      if indOuterList >= len(offsetsPerLine):
        curLineOffsets = ""
        break # everything should have been appended already
      curOffsetList = offsetsPerLine[indOuterList].split()
      indCoreNLP = 0
      curLineOffsets = ""
    pythonStartOffset = pythonOffsetResults[indPython][1]
    pythonWord = pythonOffsetResults[indPython][0]
    coreNLPStartOffset = int(curOffsetList[indCoreNLP]) + adjustTerm
    curWord = textPerLine[indOuterList].split()[indCoreNLP]
    if pythonStartOffset < coreNLPStartOffset:
      prevWord = ""
      if indCoreNLP > 0:
        prevWord = textPerLine[indOuterList].split()[indCoreNLP-1]
      prevWord = cleanWord(prevWord)
      curWord = cleanWord(curWord)
      if prevWord + curWord in prevPythonWordCleaned or (re.search(ur'^\-\_\-', prevPythonWordCleaned, re.UNICODE) and "-_-" in prevWord and prevWord.split('-_-')[1] + curWord in prevPythonWordCleaned):
        # adjust coreNLP offset backwards (decoding issue?)
        adjustTerm += pythonStartOffset - coreNLPStartOffset
        coreNLPStartOffset += pythonStartOffset - coreNLPStartOffset
        coreNLPStartOffset -= len(curWord)
        curLineOffsets += str(coreNLPStartOffset) + " "
        if curOffsetList[indCoreNLP] in offset2NormNer:
          newOffset2NormNer[str(coreNLPStartOffset)] = str(offset2NormNer[curOffsetList[indCoreNLP]])
        mappingOld2New[curOffsetList[indCoreNLP]] = str(coreNLPStartOffset)
        indCoreNLP += 1
        continue
      if (curWord in cleanWord(pythonWord) or cleanWord(curWord) in cleanWord(pythonWord)):
        # adjust coreNLP offset backwards (decoding issue?)
        adjustTerm += pythonStartOffset - coreNLPStartOffset
        coreNLPStartOffset += pythonStartOffset - coreNLPStartOffset
        curLineOffsets += str(coreNLPStartOffset) + " "
        if curOffsetList[indCoreNLP] in offset2NormNer:
          newOffset2NormNer[str(coreNLPStartOffset)] = str(offset2NormNer[curOffsetList[indCoreNLP]])
        mappingOld2New[curOffsetList[indCoreNLP]] = str(coreNLPStartOffset)
        indCoreNLP += 1
      indPython += 1
      prevPythonWord = pythonWord
      prevPythonWordCleaned = cleanWord(pythonWord)
      continue
    if pythonStartOffset == coreNLPStartOffset and (pythonWord[0] == curWord[0] or cleanWord(pythonWord)[0] == cleanWord(curWord)[0]):
      curLineOffsets += str(coreNLPStartOffset) + " "
      if curOffsetList[indCoreNLP] in offset2NormNer:
        newOffset2NormNer[str(coreNLPStartOffset)] = str(offset2NormNer[curOffsetList[indCoreNLP]])
      mappingOld2New[curOffsetList[indCoreNLP]] = str(coreNLPStartOffset)
      indCoreNLP += 1
      indPython += 1
      prevPythonWord = pythonWord
      prevPythonWordCleaned = cleanWord(pythonWord)
      continue
    if pythonStartOffset >= coreNLPStartOffset:
      prevWord = ""
      if indCoreNLP > 0:
        prevWord = textPerLine[indOuterList].split()[indCoreNLP-1]
      prevWord = cleanWord(prevWord)
      curWord = cleanWord(curWord)
      if indCoreNLP + 1 < len(textPerLine[indOuterList].split()):
        nextWord = textPerLine[indOuterList].split()[indCoreNLP+1]
        if curWord == '-' and ((prevWord == 'etc.' and nextWord == '(') or (prevWord == '=' and nextWord == '=' and not '-' in prevPythonWord) or (prevWord == 'Fe' and nextWord == 'ith') or (prevWord == '2008' and nextWord == 'ViewFinders') or (prevWord == '22' and nextWord == '26') or (prevWord == '10' and nextWord == '14') or (prevWord == '1pm' and nextWord == '4pm') or (re.search(r'\xc2\xa1', prevWord) and nextWord == '.') or (prevWord == '6:00' and nextWord == '8:00') or (prevWord == 'novice' and nextWord == 'or') or (prevWord == 'noon' and nextWord == '7:00') or (prevWord == '104' and nextWord == '105') or (prevWord == 'NULL-_-ST' and nextWord == 'RING') or (prevWord == "version" and nextWord == "libraries") or (prevWord == "Jews" and nextWord == "(")):
          # the word - has been inserted by CoreNLP
          # store new offset for coreNLP
          curLineOffsets += str(coreNLPStartOffset) + " "
          if curOffsetList[indCoreNLP] in offset2NormNer:
            newOffset2NormNer[str(coreNLPStartOffset)] = str(offset2NormNer[curOffsetList[indCoreNLP]])
          mappingOld2New[curOffsetList[indCoreNLP]] = str(coreNLPStartOffset)
          curWord = nextWord
          indCoreNLP += 1
          coreNLPStartOffset = int(curOffsetList[indCoreNLP]) + adjustTerm
      origPrevPythonWord = prevPythonWord 
      origPrevPythonWordCleaned = prevPythonWordCleaned
      if curWord == pythonWord and not curWord in prevPythonWord:
        adjustTerm += pythonStartOffset - coreNLPStartOffset
        # adjust coreNLPOffsets
        coreNLPStartOffset += pythonStartOffset - coreNLPStartOffset
        indPython += 1
        prevPythonWord = pythonWord
        prevPythonWordCleaned = cleanWord(pythonWord)
      elif (prevWord + curWord in prevPythonWord or
         prevWord + curWord in prevPythonWordCleaned or
         cleanWord(prevWord + curWord) in prevPythonWordCleaned or
         cleanWord(prevWord + curWord) in cleanWord(prevPythonWordCleaned) or
         re.sub(ur'\.\.', '.', prevWord + curWord, re.UNICODE) in prevPythonWord or
         re.sub(ur'\.\.', '.', prevWord + curWord, re.UNICODE) in prevPythonWordCleaned or
         re.sub(ur'\.\.\.', '.', prevWord + curWord, re.UNICODE) in prevPythonWord or
         re.sub(ur'\.\.\.', '.', prevWord + curWord, re.UNICODE) in prevPythonWordCleaned or
         re.sub(ur'\"', "''", prevWord + curWord, re.UNICODE) in re.sub(ur'\"', "''", prevPythonWordCleaned, re.UNICODE) or
         ((prevWord + curWord)[-1] == prevPythonWordCleaned[-1]) or
         (re.search(ur'^\-\_\-', prevPythonWord, re.UNICODE) and '-_-' in curWord and curWord.split('-_-')[1] in prevPythonWord) or 
         (re.search(ur'^\-\_\-', prevPythonWordCleaned, re.UNICODE) and '-_-' in curWord and curWord.split('-_-')[1] in prevPythonWordCleaned) or 
         (re.search(ur'^\-\_\-', prevPythonWord, re.UNICODE) and '-_-' in prevWord and prevWord.split('-_-')[1]+curWord in prevPythonWord) or 
         (re.search(ur'^\-\_\-', prevPythonWordCleaned, re.UNICODE) and '-_-' in prevWord and prevWord.split('-_-')[1]+curWord in prevPythonWordCleaned)): 
        pass
      else:
        adjustTerm += pythonStartOffset - coreNLPStartOffset
        # adjust coreNLPOffsets
        coreNLPStartOffset += pythonStartOffset - coreNLPStartOffset
        indPython += 1
        prevPythonWord = pythonWord
        prevPythonWordCleaned = cleanWord(pythonWord)
      # store new offset for coreNLP
      if "-_-" in curWord and (prevWord + curWord.split('-_-')[0] in origPrevPythonWord or prevWord + curWord.split('-_-')[0] in origPrevPythonWordCleaned or curWord.split('-_-')[0] in origPrevPythonWord or curWord.split('-_-')[0] in origPrevPythonWordCleaned) and not re.search(ur'^\-\_\-', prevPythonWord, re.UNICODE):
        tmpPrevCurWord = re.sub(ur'\-\_\-', '_', prevWord + curWord, re.UNICODE)
        prevPythonWord = "-_-" + prevPythonWord
        prevPythonWordCleaned = "-_-" + prevPythonWordCleaned
      else:
        curLineOffsets += str(coreNLPStartOffset) + " "
        if curOffsetList[indCoreNLP] in offset2NormNer:
          newOffset2NormNer[str(coreNLPStartOffset)] = str(offset2NormNer[curOffsetList[indCoreNLP]])
        mappingOld2New[curOffsetList[indCoreNLP]] = str(coreNLPStartOffset)
        indCoreNLP += 1
        continue
  if curLineOffsets != "":
    if len(curOffsetList) != len(curLineOffsets.strip().split()):
      logger.error("different offset numbers of line " + textPerLine[indOuterList])
      usingOldOffsets = True
    else:
      newOffsetsPerLine.append(curLineOffsets.strip()) # append everything which has not been appended so far

  if not usingOldOffsets and len(offsetsPerLine) != len(newOffsetsPerLine):
    logger.error("length offsets per line old and new are different")
    usingOldOffsets = True

  if not usingOldOffsets:
    for sublist in additionalNameOffsets:
      newOffsetListTotal = []
      for item in sublist:
        if item == "":
          continue
        itemList = item.split(',')
        newOffsetList = []
        for il in itemList:
          if not il in mappingOld2New:
            logger.error("(1) could not find offset " + il + " in mappingOld2New: " + str(mappingOld2New))
            continue
          newItem = mappingOld2New[il]
          newOffsetList.append(newItem)
        newOffsetListTotal.append(",".join(newOffsetList))
      newAdditionalNameOffsets.append(newOffsetListTotal)

    # same for name offsets and orig name offsets
    for item in nameOffsets:
      newOffsetList = []
      if item.strip() == "":
        continue
      allOffsets = item.split(',')
      for offs in allOffsets:
        if offs == "":
          continue
        if not offs in mappingOld2New:
          logger.error("(2) could not find offset " + offs + " in mappingOld2New: "  + str(mappingOld2New))
          continue
        newOffs = mappingOld2New[offs]
        newOffsetList.append(newOffs)
      newNameOffsets.append(",".join(newOffsetList))

    for item in origNameOffsets:
      newOffsetList = []
      idOrig = item[0]
      origOffset = item[1]
      if origOffset == "":
        continue
      allOffsets = origOffset.split(',')
      for offs in allOffsets:
        if offs == "":
          continue
        if not offs in mappingOld2New:
          logger.error("(3) could not find offset " + offs + " in mappingOld2New: "  + str(mappingOld2New))
          continue
        newOffs = mappingOld2New[offs]
        newOffsetList.append(newOffs)
      newOrigNameOffsets.append([idOrig, ",".join(newOffsetList)])

    for pronounOffset in additionalCorefFillers:
      if not pronounOffset in mappingOld2New:
        logger.error("(4) could not find offset " + pronounOffset + " in mappingOld2New: " + str(mappingOld2New))
        continue
      newPronounOffset = mappingOld2New[pronounOffset]
      person = additionalCorefFillers[pronounOffset][0]
      personOffsets = additionalCorefFillers[pronounOffset][1].split(',')
      newPersonOffsets = []
      for po in personOffsets:
        if not po in mappingOld2New:
          logger.error("(4) could not find offset " + po + " in mappingOld2New: " + str(mappingOld2New))
          continue
        newPersonOffsets.append(mappingOld2New[po])
      newAdditionalCorefFillers[newPronounOffset] = [person, ",".join(newPersonOffsets)]

    return [newOffsetsPerLine, newOffset2NormNer, newAdditionalNameOffsets, newNameOffsets, newOrigNameOffsets, newAdditionalCorefFillers]
  else:
    return [offsetsPerLine, offset2NormNer, additionalNameOffsets, nameOffsets, origNameOffsets, additionalCorefFillers]
