#!/usr/bin/python
# -*- coding: utf-8 -*-

#####################################
### CIS SLOT FILLING SYSTEM      ####
### 2014-2015                    ####
### Author: Heike Adel           ####
#####################################

########### this function performs NER and coref on the given sentences
## it calls the stanford tagger and parses its output
## it replaces markables of the name with the name
## it splits the resulting word- and ner-lists into one list for each sentence
## it returns these lists per sentences

from __future__ import unicode_literals
import codecs, sys
reload(sys)
sys.setdefaultencoding('utf-8')
sys.stdout = codecs.getwriter('utf8')(sys.stdout)
sys.stderr = codecs.getwriter('utf8')(sys.stderr)
import os
import re
import string
import os.path
from utilities import getsubidx, levenshtein, compareNamesImproved, updateBestIndices
from doNerAndCoref import tagDocument
import gzip
import copy
import io

import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
logger.setLevel(logging.DEBUG)

triggerWordsHyphenORG = ["-based", "-member"]
triggerWordsHyphenPER = ["-born", "year-old", "month-old", "-educated"]
triggerWordsHyphenLOC = ["-born", "-based"]
triggerWordsORG = ["company", "firm", "agency", "bureau", "department", "office", "group", "organization"]
triggerWordsPER = ["child"]

titles = ["Baroness", "Duke", "Sir", "Mr", "Mr.", "Mrs", "Mrs.", "Ms", "Miss", "Ms.", "Mister", "Lord", "Lady", "President", "Earl", "Count", "Imam", "Sheik"]

pronounListORG = ["it", "its", "itself", "we", "us", "our", "ours", "ourselves", "they", "them", "their", "theirs", "themselves", "themself"]
pronounListPER = ["i", "me", "my", "mine", "myself", "you", "your", "yours", "yourself", "he", "him", "his", "himself", "she", "her", "hers", "herself", "we", "us", "our", "ours", "ourselves", "yourselves", "they", "them", "their", "theirs", "themselves", "themself"]
pronounsForFiller = ["i", "I", "he", "He", "she", "She", "you", "You"]

def getIndexForMentionId(mention, resultingIds):
  result = []
  sentId, startId, endId = mention.split('-')
  for curIndex, curIds in enumerate(resultingIds):
    curIdList = curIds.split()
    if curIdList[0].split('-')[0] != sentId:
      continue
    # we are in the correct sentence
    result.append(curIndex)
    indexList = []
    for i in range(int(startId), int(endId)):
      searchTerm = sentId + "-" + str(i)
      if not searchTerm in curIdList:
        logger.warning("could not find " + searchTerm + " in resultingIds: " + str(resultingIds))
        break
      index = curIdList.index(searchTerm)
      indexList.append(index)
    if len(indexList) == 0:
      return []
    result.append(indexList)
    return result
  return result

def getLemmaStringForIds(mentionStart, mentionEnd, resultingIds, resultingLemmas, docId):
  myText = ""
  sentId = mentionStart.split('-')[0]
  sentStart = mentionStart.split('-')[1]
  sentEnd = mentionEnd.split('-')[1]
  for ind, ids in enumerate(resultingIds):
    if " " + mentionStart + " " in " " + ids + " ":
      idsList = ids.split()
      index = idsList.index(mentionStart)
      textList = resultingLemmas[ind].split()
      while index < len(idsList) and int(sentEnd) > int(idsList[index].split('-')[1]):
        myText += textList[index] + " "
        index += 1
      break
    else:
      # exact start id is not in ids: search for sentence and extract all words between start and end id
      if int(sentId) == int(ids.split()[0].split('-')[0]):
        # found sentence
        index = 0
        idsList = ids.split()
        textList = resultingLemmas[ind].split()
        while index < len(idsList) and int(sentStart) > int(idsList[index].split('-')[1]):
          index += 1
        if index >= len(idsList):
          # did not find mention
          logger.debug(docId + ": did not find mention " + str([mentionStart, mentionEnd]) + " in ids")
        else:
          while index < len(idsList) and int(sentEnd) > int(idsList[index].split('-')[1]):
            myText += textList[index] + " "
            index += 1
          break
  myText = myText.strip()
  return myText

def getIndexOfMention(mention, resultingIds):
  sentId = mention[0]
  sentStart = mention[1]
  sentEnd = mention[2]
  mentionStart = sentId + "-" + sentStart
  mentionEnd = sentId + "-" + sentEnd
  sentInd = -1
  startInd = -1
  endInd = -1
  for ind, ids in enumerate(resultingIds):
    if ids == "":
      continue
    if int(sentId) == int(ids.split()[0].split('-')[0]):
      # found sentence
      sentInd = ind
    else:
      continue
    index = 0
    idsList = ids.split()
    while index < len(idsList) and int(sentStart) > int(idsList[index].split('-')[1]):
      index += 1
    if index >= len(idsList):
      # did not find mention
      break
    else:
      startInd = index
      while index < len(idsList) and int(sentEnd) > int(idsList[index].split('-')[1]):
        index += 1
      endInd = index - 1
  if sentInd != -1 and startInd != -1 and endInd != -1:
    return [sentInd, startInd, endInd + 1] # endIndex points to the index AFTER the mention!
  else:
    return [-1, startInd, endInd] # error occurred

def cleanCorefChain(chain):
  curChain = chain
  redundantElements = []
  for ccInd in range(len(curChain)):
    cc = curChain[ccInd]
    for cc2Ind in range(ccInd + 1, len(curChain)):
      cc2 = curChain[cc2Ind]
      if int(cc[0]) == int(cc2[0]):
        if int(cc[1]) >= int(cc2[1]) and int(cc[2]) <= int(cc2[2]): # cc is included in cc2
          redundantElements.append(ccInd)
        elif int(cc2[1]) >= int(cc[1]) and int(cc2[2]) <= int(cc[2]): # cc2 is included in cc
          redundantElements.append(cc2Ind)
  redundantElementsSorted = sorted(list(set(redundantElements)), reverse=True)
  # delete redundant Elements
  for res in redundantElementsSorted:
    curChain.pop(res)
  return curChain

def getOffsetsForChain(chains, resultingIds, resultingOffsets):
  nameOffsets = []
  for curChain in chains:
    chainOffsets = []
    curNameOffset = ""
    for curMention in curChain:
      start = int(curMention[1])
      end = int(curMention[2])
      for i in range(start, end):
        searchTerm = curMention[0] + "-" + str(i)
        indices = [k for k, x in enumerate(resultingIds) if " " + searchTerm in " " + x]
        for sentInd in indices:
          curIdList = resultingIds[sentInd].split()
          indices2 = [k for k, x in enumerate(curIdList) if x == searchTerm]
          if len(indices2) == 0:
            logger.error("could not find end of mention of coref chain " + searchTerm + " in ids " + resultingIds[sentInd])
            continue
          if i == start:
            curNameOffset += " " + str(resultingOffsets[sentInd].split()[indices2[0]])
            for n in range(1, len(indices2)):
              curNameOffset += "," + str(resultingOffsets[sentInd].split()[indices2[n]])
          else:
            for n in indices2:
              curNameOffset += "," + str(resultingOffsets[sentInd].split()[n])
    curNameOffset = curNameOffset.strip()
    for cno in curNameOffset.split():
      if not cno in chainOffsets:
        chainOffsets.append(cno)
    nameOffsets.append(chainOffsets)
  return nameOffsets


def getCoreferredMentions(entity, chains, resultingIds, resultingLemmas, pronounList, docId):
  corefMentions = []
  entitySplitted = entity.split('-')
  entitySent = int(entitySplitted[0])
  entityStart = int(entitySplitted[1])
  entityEnd = int(entitySplitted[2])
  found = 0
  for wholeChain in chains:
    curMentions = []
    for mention in wholeChain:
      sentId = int(mention[0])
      start = int(mention[1])
      end = int(mention[2])
      if sentId == entitySent:
        if start <= entityStart and end >= entityEnd: # entity is contained in mention
          curMentions.append([entitySplitted[0], entitySplitted[1], entitySplitted[2]])
          for ment in wholeChain:
            if ment == mention: # already appended
              continue
            mentionStart = ment[0] + "-" + ment[1]
            mentionEnd = ment[0] + "-" + ment[2]
            myText = getLemmaStringForIds(mentionStart, mentionEnd, resultingIds, resultingLemmas, docId)
            if string.lower(myText) in pronounList:
              curMentions.append(ment)
    if len(curMentions) > 0:
      corefMentions.append(curMentions)
  if len(corefMentions) == 0:
    corefMentions.append([entitySplitted])
  return corefMentions

def leftIsDifferent(startInd, curNerList, curWordList):
  if startInd - 1 < 0:
    return 1
  if curNerList[startInd] != curNerList[startInd - 1]:
    return 1
  while curWordList[startInd - 1] in titles:
    startInd = startInd - 1
    if startInd - 1 < 0:
      return 1
    if curNerList[startInd] != curNerList[startInd - 1]:
      return 1
  return 0

def rightIsDifferent(endInd, curNerList):
  # endInd points to the word AFTER the name string
  if endInd >= len(curNerList):
    return 1
  if curNerList[endInd - 1] != curNerList[endInd]:
    return 1
  return 0


def nerAndCoref(nameList, docId, docPath, isPer, isOrg, isLoc, doCoref = 1, isBolt = 0, isPersonFiller = 1):

  name = nameList[0]

  global pronounListPER
  global pronounListORG
  if isPer == 1:
    pronounList = copy.deepcopy(pronounListPER)
  elif isOrg == 1:
    pronounList = copy.deepcopy(pronounListORG)
    if "Foundation" in name:
      triggerWordsORG.append("foundation")
  else:
    pronounList = []

  corefDir = TAGGINGPATH # replace with directory where CoreNLP output has been stored (from doNerAndCoref.py)

  if not os.path.isfile(corefDir + '/' + docId + ".prepared.gz"):
    tagDocument(docId, docPath, isBolt)
    if not os.path.isfile(corefDir +'/' + docId + ".prepared.gz"): # tagging went wrong
      logger.error("could not find tagging result for " + docId + " in " + corefDir)
      return []
  inTaggedDir = 0
  g = gzip.GzipFile(corefDir + '/' + docId + ".prepared.gz")
  f = io.TextIOWrapper(io.BufferedReader(g), encoding='utf-8')
  resultingWords = []
  resultingNer = []
  resultingLemmas = []
  resultingIds = []
  resultingOffsets = []
  resultingPos = []
  normDateMap = {}
  idOrigName = {}
  readingCoref = 0
  chains = []
  additionalChains = []
  longestNameLength = 0
  idOrigNameInSent = {}
  spellingVariations = {}
  additionalCorefFillers = {}
  allChains = []
  for n in nameList:
    nLen = n.split()
    longestNameLength = max(len(nLen), longestNameLength)
  foundNames = []
  foundDoc = 0
  finishedDoc = 0 # if document id exists more than once in opened file
  for line in f:
    if finishedDoc == 1: # do not read document twice
      break
    if inTaggedDir == 1:
      # search beginning of document
      if foundDoc == 0 and finishedDoc == 0:
        if re.search(ur"^\<DOC\>", line, re.UNICODE):
          if docId in line:
            foundDoc = 1
            continue
          else:
            continue
        else:
          continue
      if foundDoc == 1 and readingCoref == 1 and re.search(ur"^\<\/DOC\>", line, re.UNICODE):
        # found end of document
        foundDoc = 0
        finishedDoc = 1
        readingCoref = 0
        break
    else:
      if foundDoc == 0 and finishedDoc == 0:
        if re.search(ur"\<DOC", line, re.UNICODE) or re.search(ur'\<doc', line, re.UNICODE):
          if docId in line:
            foundDoc = 1
          else:
            continue
    line = line.strip()
    if "<coreference>" == line:
      # start reading coreference information
      if foundDoc == 1:
        readingCoref = 1
    elif readingCoref == 1:
      if doCoref == 0:
        continue
      # reading coreference information
      wholeChain = [] # read chain and see if the name appears in it
      curChain = []
      curChainAdditional = []
      parts = line.split('___')
      for p in parts:
        elements = p.split('-')
        if len(elements) < 3:
          logger.error("not enough mentions in coreference element: " + str(elements))
        else:
          wholeChain.append(elements)
      allChains.append(parts)
      for mention in wholeChain:
        mentionStart = mention[0] + "-" + mention[1]
        mentionEnd = mention[0] + "-" + mention[2]
        myText = getLemmaStringForIds(mentionStart, mentionEnd, resultingIds, resultingLemmas, docId)
        for foundName in idOrigNameInSent:
          origNameOccurrence = idOrigNameInSent[foundName]
          for occ in origNameOccurrence:
            # case 1: text of coreference mention is longer than name:
            if int(occ[0]) == int(mention[0]): # we are in the same sentence
              # case 1: text of coreference mention is longer than name:
              if int(mention[1]) <= int(occ[1]) and int(mention[2]) >= int(occ[2]):
                # name found: store name indices
                if not foundName in idOrigName:
                  idOrigName[foundName] = []
                tmpList = []
                tmpList.append(mention[0])
                tmpList.append(occ[1])
                tmpList.append(occ[2])
                if not tmpList in idOrigName[foundName]:
                  idOrigName[foundName].append(tmpList)
                curChain.append(tmpList)
                foundNames.append(foundName)
              # case 2: text of coreference mention is shorter than name:
              if int(mention[1]) >= int(occ[1]) and int(mention[2]) <= int(occ[2]):
                # name found: store name indices
                if not foundName in idOrigName:
                  idOrigName[foundName] = []
                tmpList = []
                tmpList.append(mention[0])
                tmpList.append(occ[1])
                tmpList.append(occ[2])
                if not tmpList in idOrigName[foundName]:
                  idOrigName[foundName].append(tmpList)
                curChain.append(tmpList)
                foundNames.append(foundName)
        curChainAdditional.append(mention)
        if string.lower(myText) in pronounList and myText != "US": # replace only pronouns
          curChain.append(mention)
        else:
          if re.search(ur'\'\s*s\s*$', myText, re.UNICODE):
            mention[2] = str(int(mention[2]) - 1)
          myTextTmp = re.sub(ur'\'\s*s\s*$', '', myText, re.UNICODE)
          myTextTmp = myTextTmp.strip()
          myTextTmpLc = string.lower(myTextTmp)
          nameLc = string.lower(name)
          if not myTextTmpLc.isspace() and not myTextTmpLc == "":
            for aliasInd, alias in enumerate(nameList): # original name is included in nameList, too!
              aliasLc = string.lower(alias)
              if " " + myTextTmpLc + " " in " " + aliasLc + " " and re.search(ur'[a-z]', myTextTmpLc):
                curChain.append(mention)
                # check whether same NER tag (!= O) continues to the left of to the right - then we don't have an occurrence of the name here but probably someone with the same first name / last name (exception: isOrg and the string matches exactly the name => NER tagging was probably wrong in this case!)
                curInd, startInd, endInd = getIndexOfMention(mention, resultingIds)
                if curInd == -1:
                  continue
                # try to extend mention to the left and right
                while startInd > 0 and (" ".join(resultingWords[curInd].split()[startInd - 1 : endInd]) in name or " ".join(resultingWords[curInd].split()[startInd - 1 : endInd]) in alias):
                  startInd -= 1
                while endInd < len(resultingWords[curInd].split()) - 1 and (" ".join(resultingWords[curInd].split()[startInd : endInd + 1]) in name or " ".join(resultingWords[curInd].split()[startInd : endInd + 1]) in alias):
                  endInd += 1
                curNerList = resultingNer[curInd].split()
                if curNerList[startInd] == 'O' or (leftIsDifferent(startInd, curNerList, resultingWords[curInd].split()) and rightIsDifferent(endInd, curNerList)) or (isOrg and " ".join(resultingWords[curInd].split()[startInd : endInd]) == name):
                  if not aliasInd in foundNames:
                    foundNames.append(aliasInd)

      # storing current chain: distinguish between chains containing the entity and between other chains
      if len(foundNames) > 0:
        curChain = cleanCorefChain(curChain)
        if not curChain in chains:
          chains.append(curChain)
        foundNames = []
      else:
        curChainAdditional = cleanCorefChain(curChainAdditional)
        if not curChainAdditional in additionalChains:
          additionalChains.append(curChainAdditional)

    else:
      # reading sentence
      parts = line.split()
      curWords = ""
      curNers = ""
      curPos = ""
      curLemmas = ""
      curIds = ""
      curOffsets = ""
      sentencesNormDate = ""
      for p in parts:
        if "___six____" in p:
          p = re.sub(ur'\_\_\_six\_\_\_\_', '____six___', p, re.UNICODE)
        if "___article.html/morton____" in p:
          p = re.sub(ur'\_\_\_article\.html\/morton\_\_\_\_', '____article.html/morton___', p, re.UNICODE)
        if "_______" in p:
          p = re.sub(ur'\_\_\_(\_+)\_\_\_', '___-_-___', p, re.UNICODE)
        if "____" in p:
          # is not a part seperator
          p = re.sub(ur'\_\_\_(\_)+', '-_-', p, re.UNICODE)
        elements = p.split('___')
        if len(elements) == 1: # word belongs to sentenceNormDate from token before
          if re.search(ur'[a-z]', elements[0], re.UNICODE):
            continue # sentenceNormDate consists of capital letters, numbers and symbols
          sentencesNormDate = sentencesNormDate.strip() + "_" + "_".join(elements[0].split()) + " "
          continue
        if len(elements) > 6:
          merged = 0
          mergedLemma = 0
          while len(elements) > 2 and not re.search(ur'^\d+\-\d+$ ', elements[2], re.UNICODE):
            # word has been splitted: merge it into one word again:
            elements[1] += '-_-' + elements[2]
            elements.pop(2)
            merged += 1
          if len(elements) <= 2:
            # did not work
            logger.error("error in parsing word sentence part of " + docId + ": skip sentence part " + p)
            continue
          # do the same for lemma:
          while len(elements) > 4 and not re.search(ur'^[A-Z]+\$*$', elements[4], re.UNICODE):
            elements[3] += '-_-' + elements[4]
            elements.pop(4)
            mergedLemma += 1
          if len(elements) <= 4:
            # did not work
            logger.error("error in parsing lemma sentence part of " + docId + ": skip sentence part " + p)
            continue
          if merged != mergedLemma:
            logger.warning("merged " + str(merged) + " word parts but " + str(mergedLemma) + " lemma parts for " + p)
        if len(elements) != 6:
          # something is wrong 
          logger.error("error in parsing of " + docId + ": skip sentence part " + " ".join(elements))
          continue
        ids = elements[0]
        word = elements[1]
        if "QUOTE-_-PREVIOUSPOST" in word:
          # need special treatment for this part: needs to be splitted etc
          continue
        begin = elements[2].split('-')[0]
        end = elements[2].split('-')[1]
        lemma = elements[3]
        pos = elements[4]
        ner = elements[5]
        normalizedNer = ""
        nerSplitted = ner.split('__')
        if word in [";", ","]:
          ner = "0" # allow enumerations of things
        if len(nerSplitted) > 1:
          ner = nerSplitted[0]
          normalizedNer = "__".join(nerSplitted[1:])
          myInfo = ids + "___" + normalizedNer.strip() 
          sentencesNormDate += myInfo + " " 
        if isOrg == 1 and re.search(ur'^[A-Z]+\/[A-Z]+$', word):
          word1, word2 = word.split('/')
          begin1 = begin
          begin2 = str(int(begin) + len(word1))
          begin3 = str(int(begin2) + 1)
          curWords += word1 + " / " + word2 + " "
          if word1 in nameList or word2 in nameList:
            curNers += "ORGANIZATION 0 ORGANIZATION " # often wrongly tagged as MISC
          else:
            curNers += ner + " 0 " + ner + " "
          curPos += pos + " SYM " + pos + " "
          curLemmas += lemma + " / " + lemma + " "
          curIds += ids + " " + ids + " " + ids + " "
          curOffsets += begin1 + " " + begin2 + " " + begin3 + " "
        elif isLoc == 1:
          foundTrHyphen = 0
          for trHyphen in triggerWordsHyphenLOC:
            if trHyphen + " " in word + " ":
              word1 = re.sub(ur'' + trHyphen + '$', '', word)
              word2 = trHyphen
              begin1 = begin
              begin2 = str(int(begin) + len(word1))
              curWords += word1 + " " + word2 + " "
              curNers += ner + " " + "0" + " "
              curPos += pos + " " + pos + " "
              curLemmas += lemma + " " + lemma + " "
              curIds += ids + " " + ids + " "
              curOffsets += begin1 + " " + begin2 + " "
              foundTrHyphen = 1
          if foundTrHyphen == 0:
            curWords += word + " "
            curNers += ner + " "
            curPos += pos + " "
            curLemmas += lemma + " "
            curIds += ids + " "
            curOffsets += begin + " "
        else:
          curWords += word + " "
          curNers += ner + " "
          curPos += pos + " "
          curLemmas += lemma + " "
          curIds += ids + " "
          curOffsets += begin + " "

      # search for name in sentence
      compareTrue = 0
      for myName in nameList:
        bestIndices = []
        occurrences = compareNamesImproved(myName, curWords)
        updateBestIndices(bestIndices, occurrences)
        for bestStart, bestEnd in bestIndices:
          compareTrue = 1
          foundNameInSent = nameList.index(myName)
          myId = curIds.split()[0]
          idSplitted = myId.split('-')
          curSentId = idSplitted[0]
          curStartId = idSplitted[1]
          nameStart = int(curStartId) + bestStart
          nameEnd = int(curStartId) + bestEnd
          nameStartIndex = nameStart - 1
          nameEndIndex = nameEnd - 1 # because ids start at 1 but list offsets at 0

          # try to extend mention to the left and right
          while nameStartIndex > 0 and (" ".join(curWords.split()[nameStartIndex - 1 : nameEndIndex]) in name or " ".join(curWords.split()[nameStartIndex - 1 : nameEndIndex]) in myName):
            nameStartIndex -= 1
          while nameEndIndex < len(curWords.split()) - 1 and (" ".join(curWords.split()[nameStartIndex : nameEndIndex + 1]) in name or " ".join(curWords.split()[nameStartIndex : nameEndIndex + 1]) in myName):
            nameEndIndex += 1

          # look whether NER left and right is different (if NER != O)
          # if not, we don't have an occurrence of the name here but probably someone with the same first name / last name
          # exception: isOrg and string matches exactly the full org name (=> probably NER error)
          curNerList = curNers.split()
          if nameStartIndex >= 0 and nameEndIndex > nameStartIndex and nameStartIndex < len(curNerList) and (curNerList[nameStartIndex] == 'O' or (leftIsDifferent(nameStartIndex, curNerList, curWords.split()) and rightIsDifferent(nameEndIndex, curNerList)) or (isOrg and " ".join(curWords.split()[nameStartIndex : nameEndIndex]) == name)):
            if not foundNameInSent in idOrigNameInSent:
              idOrigNameInSent[foundNameInSent] = []
            tmpList = []
            myId = curIds.split()[0]
            idSplitted = myId.split('-')
            curSentId = idSplitted[0]
            curStartId = idSplitted[1]
            tmpList.append(curSentId)
            tmpList.append(str(nameStartIndex + 1))
            tmpList.append(str(nameEndIndex + 1))
            nameString = " ".join(curWords.split()[nameStartIndex : nameEndIndex])
            if nameString[0] != '<' and nameString[-1] != '>':
              idOrigNameInSent[foundNameInSent].append(tmpList)

              # if there is a spelling variation, we have found an alias!
              nameInSentence = " ".join(curWords.split()[nameStartIndex : nameEndIndex])
              if nameInSentence != myName:
                if nameInSentence.isupper() and myName.isupper(): # for acronyms: do not append different acronyms
                  pass
                else:
                  spellingVariations[nameInSentence] = [docId, tmpList]
            else:
              logger.debug("found tag instead of name: " + nameString)

      if curWords.strip() != "":
        resultingWords.append(curWords.strip())
        resultingNer.append(curNers.strip())
        resultingPos.append(curPos.strip())
        resultingLemmas.append(curLemmas.strip())
        resultingIds.append(curIds.strip())
        resultingOffsets.append(curOffsets.strip())
      normDateList = sentencesNormDate.split()
      for nd in normDateList:
        if not "___" in nd:
          continue
        curNd = nd.split('___')
        date = curNd[1]
        date = re.sub(ur'^([\dX]{4}\-[\dX]{2}\-[\dW]{2}).*$', '\\1', date, re.UNICODE)
        if not re.search(ur'[A-WYZa-z\.\$]', date, re.UNICODE):
          normDateMap[curNd[0]] = date

  f.close()

  for myId in idOrigNameInSent:
    myList = idOrigNameInSent[myId]
    for ml in myList:
      if not myId in idOrigName:
        idOrigName[myId] = []
      if not ml in idOrigName[myId]:
        idOrigName[myId].append(ml)

  # clean idOrigName
  idOrigName1 = copy.deepcopy(idOrigName)
  keys1 = list(idOrigName.keys())
  keys2 = list(idOrigName.keys())
  for k1 in keys1:
    toDelete1 = []
    idList1 = idOrigName[k1]
    for k2 in keys2:
      if k2 <= k1:
        continue
      toDelete2 = []
      idList2 = idOrigName[k2]
      for idItem1 in idList1:
        for idItem2 in idList2:
          if idItem1[0] == idItem2[0]:
            if int(idItem2[1]) >= int(idItem1[1]) and int(idItem2[2]) <= int(idItem1[2]):
              # idItem2 is included in idItem1: delete idItem2!
              toDelete2.append(idItem2)
            elif int(idItem1[1]) >= int(idItem2[1]) and int(idItem1[2]) <= int(idItem2[2]):
              # idItem1 is included in idItem2: delete idItem1!
              toDelete1.append(idItem1)
      for d2 in toDelete2:
        if d2 in idOrigName[k2]:
          idOrigName[k2].remove(d2)
    for d1 in toDelete1:
      if d1 in idOrigName[k1]:
        idOrigName[k1].remove(d1)
  toDeleteKeys = []
  for k in idOrigName:
    if len(idOrigName[k]) == 0:
      toDeleteKeys.append(k)
  for dk in toDeleteKeys:
    del idOrigName[dk]

  spellingVariationsOffsets = []
  for spVar in spellingVariations:
    doc, ml = spellingVariations[spVar]
    sentIndex, startIndex, endIndex = getIndexOfMention(ml, resultingIds)
    spVarOffsets = ""
    if sentIndex == -1:
      logger.error("could not find mention of spelling variation  " + ml[0] + "-" + ml[1] + "-" + ml[2] + " in ids " + str(resultingIds))
      continue
    else:
      curOffsets = resultingOffsets[sentIndex].split()
      for j in range(startIndex, endIndex):
        spVarOffsets += curOffsets[j] + ","
      spVarOffsets = spVarOffsets.strip(',')
      spellingVariationsOffsets.append([spVar, doc, spVarOffsets])

  # process output from coref
  # store offset of all mentions which are coreferent to name
  nameOffsets = []
  sentOrig = "-1"
  startOrig = "-1"
  endOrig = "-1"
  origNameOffsets = []
  for idOrig in idOrigName:
    myList = idOrigName[idOrig]
    for ml in myList:
      origNameOffset = ""
      sentIndex, startIndex, endIndex = getIndexOfMention(ml, resultingIds)
      if sentIndex == -1:
        logger.error("could not find mention of original name  " + ml[0] + "-" + ml[1] + "-" + ml[2] + " in ids " + str(resultingIds))
        continue
      else:
        curOffsets = resultingOffsets[sentIndex].split()
        for j in range(startIndex, endIndex):
          origNameOffset += curOffsets[j] + ","
        origNameOffset = origNameOffset.strip(',')
        if not [idOrig, origNameOffset] in origNameOffsets:
          origNameOffsets.append([idOrig, origNameOffset])

  corefOffsets = getOffsetsForChain(chains, resultingIds, resultingOffsets)
  # we don't need the exact chain information any more
  corefOffsetsFlattened = list(set([val for sublist in corefOffsets for val in sublist]))
  nameOffsets.extend(corefOffsetsFlattened)

  for ono in origNameOffsets:
    curOffs = ono[1]
    if not curOffs in nameOffsets:
      nameOffsets.append(curOffs)

  # clean name offsets: delete offsets which are contained in other offsets:
  nameOffsets1 = copy.deepcopy(nameOffsets)
  redundant = []
  foundRed = 0
  for noInd in range(len(nameOffsets)):
    no = nameOffsets[noInd]
    for noInd2 in range(noInd + 1, len(nameOffsets)):
      no2 = nameOffsets[noInd2]
      if "," + no + "," in "," + no2 + ",":
        redundant.append(no)
        break
      elif "," + no2 + "," in "," + no + ",":
        redundant.append(no2)
  for r in redundant:
    if r in nameOffsets:
      nameOffsets.remove(r)

  additionalInformation = []

  if isLoc == 1:
    # see if sentence contains hyphen trigger and if yes, extract sentence before!
    idNames = [val for sublist in list(idOrigName.values()) for val in sublist]
    idChains = [val for sublist in chains for val in sublist]
    nameOccurrences = [nocc[0] for nocc in idNames] + [nocc[0] for nocc in idChains]
    nameOccurrences = list(set(nameOccurrences)) # delete multiple entries
    for sentenceString in nameOccurrences:
      sentenceId = int(sentenceString)
      prevSentence = sentenceId - 1
      if str(prevSentence) in nameOccurrences:
        continue # we only look for sentences which have not been found so far!
      sentenceIndex = -1
      for ind, ri in enumerate(resultingIds):
        if " " + str(sentenceId) + "-1" in " " + ri:
          sentenceIndex = ind
          break
      if sentenceIndex <= 0:
        continue # sentence does not exist or previous sentence does not exist
      # extract corresponding words and look for trigger words
      words = resultingWords[sentenceIndex]
      for trigger in triggerWordsHyphenLOC:
        if trigger + " " in words + " ":
          previousSentenceIds = [str(prevSentence), '1', str(len(resultingWords[sentenceIndex - 1].split()) + 1)]
          additionalInformation.append([previousSentenceIds])
  else:
    # see if sentence after name occurrence contains valuable information
    idNames = [val for sublist in list(idOrigName.values()) for val in sublist]
    idChains = [val for sublist in chains for val in sublist]
    nameOccurrences = [nocc[0] for nocc in idNames] + [nocc[0] for nocc in idChains]
    nameOccurrences = list(set(nameOccurrences)) # delete multiple entries
    for sentenceString in nameOccurrences:
      sentenceId = int(sentenceString)
      nextSentence = sentenceId + 1
      if str(nextSentence) in nameOccurrences:
        continue # we only look for sentences which have not been found so far!
      # extract index of sentence in resultingIds / resultingWords
      sentenceIndex = -1
      for ind, ri in enumerate(resultingIds):
        if " " + str(nextSentence) + "-1" in " " + ri:
          sentenceIndex = ind
          break
      if sentenceIndex == -1:
        continue # sentence does not exist
      # extract corresponding words and look for trigger words
      words = resultingWords[sentenceIndex]
      if isOrg == 1:
        for trigger in triggerWordsORG:
          if " " + trigger + " " in " " + words + " ":
            start = -1
            end = -1
            for index,w in enumerate(words.split()):
              if trigger in w:
                start = index + 1 # because ids start at 1 and not at 0
            end = start + 1
            if start == -1:
              logger.error("could not extract trigger index in sentence: " + words)
              continue
            triggerIndex = str(nextSentence) + "-" + str(start) + "-" + str(end)
            additionalInformation.extend(getCoreferredMentions(triggerIndex, additionalChains, resultingIds, resultingLemmas, pronounList, docId))
        for trigger in triggerWordsHyphenORG:
          if trigger + " " in words + " ":
            start = -1
            end = -1
            for index,w in enumerate(words.split()):
              if trigger in w:
                start = index + 1 # because ids start at 1 and not at 0
            end = start + 1
            if start == -1:
              logger.error("could not extract trigger index in sentence: " + words)
              continue
            triggerIndex = str(nextSentence) + "-" + str(start) + "-" + str(end)
            additionalInformation.extend(getCoreferredMentions(triggerIndex, additionalChains, resultingIds, resultingLemmas, pronounList, docId))
      elif isPer == 1:
        for trigger in triggerWordsPER:
          if " " + trigger + " " in " " + words + " ":
            start = -1
            end = -1
            for index,w in enumerate(words.split()):
              if trigger in w:
                start = index + 1 # because ids start at 1 and not at 0
            end = start + 1
            if start == -1:
              logger.error("could not extract trigger index in sentence: " + words)
              continue
            triggerIndex = str(nextSentence) + "-" + str(start) + "-" + str(end)
            additionalInformation.extend(getCoreferredMentions(triggerIndex, additionalChains, resultingIds, resultingLemmas, pronounList, docId))
        for trigger in triggerWordsHyphenPER:
          if trigger + " " in words + " ":
            start = -1
            end = -1
            for index,w in enumerate(words.split()):
              if trigger in w:
                start = index + 1 # because ids start at 1 and not at 0
            end = start + 1
            if start == -1:
              logger.error("could not extract trigger index in sentence: " + words)
              continue
            triggerIndex = str(nextSentence) + "-" + str(start) + "-" + str(end)
            newMention = getCoreferredMentions(triggerIndex, additionalChains, resultingIds, resultingLemmas, pronounList, docId)
            additionalInformation.extend(newMention)

  additionalNameOffsets = []
  additionalNameOffsets.extend(getOffsetsForChain(additionalInformation, resultingIds, resultingOffsets))

  offsets2normDate = {}
  for m in normDateMap:
    for indSent, idsPerSentence in enumerate(resultingIds):
      if ' ' + m + ' ' in ' ' + idsPerSentence + ' ':
        ind = idsPerSentence.split().index(m)
        offsets2normDate[resultingOffsets[indSent].split()[ind]] = normDateMap[m]

  if doCoref == 1 and isPersonFiller == 1:
    logger.info("getting additional coref fillers")
    # get additionalCorefFillers based on nameOffsets and allChains
    for i in range(0, len(resultingOffsets)):
      curOffsets = resultingOffsets[i]
      curWords = resultingWords[i]
      curIds = resultingIds[i]
      curOffsetList = curOffsets.split()
      curOffsetsComma = ",".join(curOffsetList)
      curWordList = curWords.split()
      curIdList = curIds.split()
      foundEntity = 0
      for no in nameOffsets:
        if foundEntity == 1:
          break
        if no in curOffsetsComma:
          foundEntity = 1
          # this sentence contains a mention of the name
          # see if it also contains a pronoun that might refer to a filler
          for p in pronounsForFiller:
            if p in curWordList:
              # it also contains a pronoun that might refer to a filler
              # test whether it refers to an entity that is a PERSON
              # get id for the pronoun:
              indexP = curWordList.index(p)
              idP = curIdList[indexP]
              pronounMention = idP + "-" + str(int(idP.split('-')[-1]) + 1)
              for chain in allChains:
                if pronounMention in chain:
                  # we have a chain for the pronoun: test whether there is also a person in this chain
                  for elem in chain:
                    if elem == pronounMention:
                      continue
                    indexOfCorefMention = getIndexForMentionId(elem, resultingIds)
                    if len(indexOfCorefMention) == 0:
                      continue
                    sentId = indexOfCorefMention[0]
                    interiorIds = indexOfCorefMention[1]
                    isPerson = 1
                    sentNerList = resultingNer[sentId].split()
                    for intId in interiorIds:
                      if sentNerList[intId] != "PERSON":
                        isPerson = 0
                        break
                    if isPerson == 1:
                      person = " ".join(resultingWords[sentId].split()[interiorIds[0] : interiorIds[-1] + 1])
                      logger.info("found pronoun " + p + " referring to person " + person)
                      offsetPronoun = curOffsetList[indexP]
                      offsetPerson = ",".join(resultingOffsets[sentId].split()[interiorIds[0] : interiorIds[-1] + 1])
                      if not " " + person + " " in " " + name + " " and not person in nameList and not offsetPerson in nameOffsets:
                        additionalCorefFillers[offsetPronoun] = [person, offsetPerson] # format: {offsetPronoun : offsetPerson}
                      break
                  break
              break

  return [resultingWords, resultingNer, resultingOffsets, nameOffsets, origNameOffsets, resultingPos, resultingLemmas, offsets2normDate, additionalNameOffsets, spellingVariationsOffsets, additionalCorefFillers]

def getSentenceAndOffsetsFromCoref(docId, docPath, isBolt = 0):
  logger.info("opening file " + docId)
  corefDir = TAGGINGPATH # replace with path where CoreNLP output has been stored (from doNerAndCoref.py)

  if not os.path.isfile(corefDir + '/' + docId + ".prepared.gz"): # not tagged
    tagDocument(docId, docPath, isBolt)
    if not os.path.isfile(corefDir +'/' + docId + ".prepared.gz"): # tagging went wrong
      logger.error("could not find tagging result for " + docId + " in " + corefDir)
      return []
  inTaggedDir = 0
  g = gzip.GzipFile(corefDir + '/' + docId + ".prepared.gz")
  f = io.TextIOWrapper(io.BufferedReader(g), encoding='utf-8')
  resultingWords = []
  resultingOffsets = []
  readingCoref = 0
  foundDoc = 0
  finishedDoc = 0 # if document id exists more than once in opened file
  corefInformation = []

  for line in f:
    if finishedDoc == 1: # do not read document twice
      continue
    if foundDoc == 1 and re.search(ur"^\<DOC\>", line, re.UNICODE): # start of new doc
      finishedDoc = 1
      continue
    if inTaggedDir == 1:
      # search beginning of document
      if foundDoc == 0 and finishedDoc == 0:
        if re.search(ur"^\<DOC\>", line, re.UNICODE):
          if docId in line:
            foundDoc = 1
            continue
          else:
            continue
        else:
          continue
      if foundDoc == 1 and readingCoref == 1 and re.search(ur"^\<\/DOC\>", line, re.UNICODE):
        # found end of document
        foundDoc = 0
        finishedDoc = 1
        readingCoref = 0
        break
    else:
      if foundDoc == 0 and finishedDoc == 0:
        if re.search(ur"\<DOC", line, re.UNICODE) or re.search(ur"\<doc", line, re.UNICODE):
          if docId in line:
            foundDoc = 1
          else:
            continue
    line = line.strip()
    if "<coreference>" == line:
      # start reading coreference information
      if foundDoc == 1:
        readingCoref = 1
    elif readingCoref == 1:
      finishedDoc = 1 # not interesting in coreference information
    else:
      # reading sentence
      parts = line.split()
      curWords = ""
      curOffsets = ""
      for p in parts:
        if "___six____" in p:
          p = re.sub(ur'\_\_\_six\_\_\_\_', '____six___', p, re.UNICODE)
        if "___article.html/morton____" in p:
          p = re.sub(ur'\_\_\_article\.html\/morton\_\_\_\_', '____article.html/morton___', p, re.UNICODE)
        if "_______" in p:
          p = re.sub(ur'\_\_\_(\_+)\_\_\_', '___-_-___', p, re.UNICODE)
        if "____" in p:
          # is not a part seperator
          p = re.sub(ur'\_\_\_(\_)+', '-_-', p, re.UNICODE)
        elements = p.split('___')
        if len(elements) == 1: # word belongs to sentenceNormDate from token before
          if re.search(ur'[a-z]', elements[0], re.UNICODE):
            continue # maybe rather not (sentenceNormDate consists of capital letters, numbers and symbols)
          continue
        if len(elements) > 6:
          merged = 0
          mergedLemma = 0
          while len(elements) > 2 and not re.search(ur'^\d+\-\d+$ ', elements[2], re.UNICODE):
            # word has been splitted: merge it into one word again:
            elements[1] += '-_-' + elements[2]
            elements.pop(2)
            merged += 1
          if len(elements) <= 2:
            # did not work
            logger.error("error in parsing word sentence part of " + docId + ": skip sentence part " + p)
            continue
          # do the same for lemma:
          while len(elements) > 4 and not re.search(ur'^[A-Z]+\$*$', elements[4], re.UNICODE):
            elements[3] += '-_-' + elements[4]
            elements.pop(4)
            mergedLemma += 1
          if len(elements) <= 4:
            # did not work
            logger.error("error in parsing lemma sentence part of " + docId + ": skip sentence part " + p)
            continue
          if merged != mergedLemma:
            logger.warning("merged " + str(merged) + " word parts but " + str(mergedLemma) + " lemma parts for " + p)
        if len(elements) != 6:
          # something is wrong
           continue
        ids = elements[0]
        word = elements[1]
        if "QUOTE-_-PREVIOUSPOST" in word:
          # need special treatment for this part: needs to be splitted etc
          continue
        begin = elements[2].split('-')[0]
        end = elements[2].split('-')[1]
        lemma = elements[3]
        pos = elements[4]
        ner = elements[5]
        normalizedNer = ""
        curWords += word + " "
        curOffsets += begin + " "

      resultingWords.append(curWords.strip())
      resultingOffsets.append(curOffsets.strip())
  f.close()
  return [resultingWords, resultingOffsets]

