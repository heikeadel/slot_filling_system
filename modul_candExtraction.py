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
import string
import re
import copy
from utilities import isSentenceAdditional, getOffsetsForRegex, month2year, replaceOffsetListWithTag, isNameInSentenceOffsets, fillerEqualsName, getFillerType, ner2fillerType
import logging

class CandidateExtraction:

  def cleanSentenceByLength(self, sentence):
    lengthThreshold = 650
    # asumption: sentence contains <name> and <filler> tags
    curWordList = sentence.split()
    if len(sentence) > lengthThreshold:
      fillerIndices = [i for i, x in enumerate(curWordList) if x == "<filler>"]
      nameIndices = [i for i, x in enumerate(curWordList) if x == "<name>"]
      fillerInd = -1
      nameInd = -1
      distanceNameFiller = len(curWordList)
      for fi in fillerIndices:
        for ni in nameIndices:
          distance = abs(ni - fi)
          if distance < distanceNameFiller:
            distanceNameFiller = distance
            nameInd = ni
            fillerInd = fi
      minInd = min(fillerInd, nameInd)
      maxInd = max(fillerInd, nameInd)
      newEnd = min(maxInd + 5, len(curWordList))
      newStart = max(0, minInd - 5)
      newSentence = " ".join(curWordList[newStart : newEnd])
      if len(newSentence) <= lengthThreshold:
        return [newStart, newEnd]
      else:
        return []
    else:
      return [0, len(curWordList)]

  def cleanSentenceByLengthBasedOnOffsets(self, sentence, offsets, fillerOffsets):
    lengthThreshold = 650
    curWordList = sentence.split()
    curOffsetList = offsets.split()
    if len(sentence) > lengthThreshold:
      fillerIndices = []
      for fo in fillerOffsets.split(","):
        if fo in curOffsetList:
          index = curOffsetList.index(fo)
          fillerIndices.append(index)
      minInd = fillerIndices[0]
      newEnd = min(minInd + 10, len(curWordList))
      newStart = max(0, minInd - 10)
      newSentence = " ".join(curWordList[newStart : newEnd])
      if len(newSentence) <= lengthThreshold:
        return [newStart, newEnd]
      else:
        return []
    else:
      return [0, len(curWordList)]

  def setDocumentInfos(self, sentences, nerInSentences, offsets, posInSentences, lemmasInSentence):
    self.sentences = copy.deepcopy(sentences)
    self.nerInSentences = copy.deepcopy(nerInSentences)
    self.offsets = copy.deepcopy(offsets)
    self.posInSentences = copy.deepcopy(posInSentences)
    self.lemmasInSentence = copy.deepcopy(lemmasInSentence)

  def getWebsiteResults(self, listOfAlias, docId):
    result = []
    foundWebsite = self.findWebsite(listOfAlias)
    if len(foundWebsite) > 0:
      sentenceIndex = foundWebsite[1]
      fillerOffset = self.offsets[sentenceIndex].split()[foundWebsite[2]]
      self.logger.info("found website " + foundWebsite[0] + " at " + fillerOffset + " with confidence " + str(foundWebsite[3]))
      result = [foundWebsite[0], self.sentences[sentenceIndex], foundWebsite[3], fillerOffset, self.offsets[sentenceIndex], docId, "STRING"]
    return result

  def isWebsiteInName(self):
    for index, words in enumerate(self.curName.split()):
      if re.search(ur'' + self.websiteRegex, words, re.UNICODE):
        # found website
        return index
    return -1

  def findWebsite(self, listOfAlias):
    for sInd, curSentence in enumerate(self.sentences):
      for wInd, words in enumerate(curSentence.split()):
        if re.search(ur'' + self.websiteRegex, words, re.UNICODE):
          # found website
          curFiller = words
          fillerInd = wInd
          # check whether it is website for curName
          toAppendMax = 0
          toAppendWebsite = 0
          numberWords = 0
          foundAcronym = 0
          for alias in listOfAlias:
            numberWords = 0
            toAppendWebsite = 0
            for part in alias.split():
              numberWords += 1
              if len(part) > 2 and string.lower(part) in string.lower(curFiller):
                toAppendWebsite += 1
            toAppendWebsite = toAppendWebsite * 1.0 / numberWords
            toAppendMax = max(toAppendMax, toAppendWebsite)
          acronym = ""
          for part in self.curName.split():
            if part[0].isupper():
              acronym += part[0]
          if len(self.curName.split()) > 1:
            if string.lower(acronym) in string.lower(curFiller):
              foundAcronym = 1
          confidence = toAppendMax
          if foundAcronym == 1:
            confidence = 1.0
          return [curFiller, sInd, fillerInd, confidence]
    # nothing found
    return []

  def getSentencesWithName(self, nameOffsets, additionalNameOffsetsFlattened):
    noName = []
    for index in range(len(self.offsets)):
      curOffsets = self.offsets[index].split()
      if isNameInSentenceOffsets(nameOffsets, curOffsets) == 0:
        if isSentenceAdditional(additionalNameOffsetsFlattened, curOffsets) > 0:
          pass # get also sentences with additional information
        else:
          noName.insert(0,index)
    for index in noName:
      self.offsets.pop(index)
      self.sentences.pop(index)
      self.nerInSentences.pop(index)
      self.posInSentences.pop(index)
      self.lemmasInSentence.pop(index)

  def cleanFromHTML(self):
    toClean = {}
    for indSent, sent in enumerate(self.sentences):
      toClean[indSent] = []
      for indWord, curWord in enumerate(sent.split()):
        if curWord[0] == '<' and curWord[-1] == '>' and curWord != "<filler>" and curWord != "<name>":
          toClean[indSent].append(indWord)
        elif re.search(ur'^\&(amp\;)?lt\;.*\&(amp\;)?gt\;$', curWord, re.UNICODE):
          toClean[indSent].append(indWord)
    toCleanKeySorted = sorted(toClean, key = toClean.get, reverse = True)
    for curKey in toCleanKeySorted:
      items = sorted(toClean[curKey], reverse = True)
      sent = self.sentences[curKey].split()
      offs = self.offsets[curKey].split()
      ners = self.nerInSentences[curKey].split()
      posS = self.posInSentences[curKey].split()
      lemmaS = self.lemmasInSentence[curKey].split()
      for i in items:
        sent.pop(i)
        offs.pop(i)
        ners.pop(i)
        posS.pop(i)
        lemmaS.pop(i)
      self.sentences[curKey] = " ".join(sent)
      self.offsets[curKey] = " ".join(offs)
      self.nerInSentences[curKey] = " ".join(ners)
      self.posInSentences[curKey] = " ".join(posS)
      self.lemmasInSentence[curKey] = " ".join(lemmaS)

  def getMappingOffset2Ner(self):
    for o, n in zip(self.offsets, self.nerInSentences):
      oList = o.split()
      nList = n.split()
      for oItem, nItem in zip(oList, nList):
        self.offset2ner[oItem] = nItem

  def getMappingAlias2Offsets(self, listOfAlias, numberOfFullAlias, d, origNameOffsets):
    curAliasOffsets = {}
    curNameOffsetsThisDoc = {}
    # extract which alias appears where
    smallestFullOffset = 1000000 # smallest offset of full name / alias
    for oo in origNameOffsets:
      aliasId = oo[0]
      myOffsets = oo[1].split()
      if not aliasId in curAliasOffsets:
        curAliasOffsets[aliasId] = []
      if aliasId < numberOfFullAlias:
        for offs in myOffsets:
          offsList = offs.split(',')
          offsListInt = list(map(int, offsList))
          minOffset = min(offsListInt)
          smallestFullOffset = min(smallestFullOffset, minOffset)
      curAliasOffsets[aliasId].append([d, myOffsets])
      if listOfAlias[aliasId] == self.curName:
        if not d in curNameOffsetsThisDoc:
          curNameOffsetsThisDoc[d] = []
        curNameOffsetsThisDoc[d].extend(myOffsets)
    return [curAliasOffsets, curNameOffsetsThisDoc, smallestFullOffset]

  def isValidAge(self, fillerCandidate):
    if fillerCandidate.isdigit() and all(ord(c) < 128 for c in fillerCandidate):
      fillerInt = int(fillerCandidate)
      if 0 <= fillerInt and fillerInt <= 150:
        return 1
    return 0

  def isValidFiller(self, slot, curFiller):
    if not re.search(r'[a-zA-Z0-9]', curFiller):
      return 0
    origins = {"europe":1, "european":1, "america":1, "american":1, "asia":1, "asian":1, "africa":1, "african":1, "australia":1, "australian":1}
    if slot == "per:age":
      return self.isValidAge(curFiller)
    if "location" in slot and not "gpe:" in slot:
      if not string.lower(curFiller) in self.locations:
        self.logger.info("no valid location: " + curFiller)
        return 0
    if "per:" in slot:
      if not "date" in slot and not slot == "per:cause_of_death" and not slot == "per:charges" and not slot == "per:title":
        if not curFiller[0].isupper():
          return 0
      if slot == "per:origin":
        if not string.lower(curFiller) in self.countries and not string.lower(curFiller) in origins:
          return 0
    elif "org:" in slot:
      if not slot == "org:website" and not "date" in slot and not "number" in slot:
        if not curFiller[0].isupper():
          return 0
      if slot == "org:number_of_employees_members":
        if curFiller in self.text2digit:
          curFiller = self.text2digit[curFiller]
        # allow commas but not dots
        if not re.search(ur'^\d*\,?\d*$', curFiller, re.UNICODE):
          return 0
      elif slot == "org:website":
        if not "." in curFiller:
          return 0
    return 1

  def extractCandidates(self, nameOffsets, origNameOffsets, additionalNameOffsets, d, additionalCorefFillers = {}):
    additionalCorefOffsets = []
    additionalNameOffsetsFlattened = [val for sublist in additionalNameOffsets for val in sublist]
    # for each sentence:
    # 1. find offset of entity mention in sentence
    # 1a. special treatment for additionalNameOffsets: also save them directly as potential fillers
    # 2. for all slots:
    #    extract possible slot fill candidates
    # 1.:

    searchSlots = self.slots # search all slots (for backward compatibility!)
    if self.curSlot != "":
      searchSlots = [self.curSlot]

    for index, curOffsets in enumerate(self.offsets):
      curOffsetList = curOffsets.split()
      if curOffsetList == []:
        continue
      startOffset = int(curOffsetList[0])
      endOffset = int(curOffsetList[-1])
      curEx = self.sentences[index]
      curExList = curEx.split()
      nameOffsetsToReplace = []
      for no in nameOffsets + additionalCorefOffsets:
        noSplitted = no.split(',')
        noStart = int(noSplitted[0])
        noEnd = int(noSplitted[-1])
        if noStart >= startOffset and noEnd <= endOffset: # we are in the correct sentence
          nameOffsetsToReplace.append(noSplitted)
      curNerList = self.nerInSentences[index].split()
      foundAdditionalLocFiller = 0
      if (self.isLoc == 1) and isSentenceAdditional(additionalNameOffsetsFlattened, curOffsetList):
        fillerCandidateString = ""
        fillerOffsets = ""
        fillerType = ""
        thSlot = ""
        if "gpe:births_in_location" in searchSlots and "-born" in self.sentences[index + 1]:
          fillerType = "PER"
          thSlot = "gpe:births_in_location"
          self.logger.info("extracting PERSON from additional sentence for born-slot")
          indexInSentence = 0
          while indexInSentence < len(curExList):
            curNer = curNerList[indexInSentence]
            if curNer == "PERSON":
              fillerCandidateString += curExList[indexInSentence]
              fillerOffsets += curOffsetList[indexInSentence]
              indexInSentence += 1
              while indexInSentence < len(curExList) and curNer == "PERSON":
                fillerCandidateString += " " + curExList[indexInSentence]
                fillerOffsets += "," + curOffsetList[indexInSentence]
                indexInSentence += 1
              break
            indexInSentence += 1
        elif "gpe:headquarters_in_location" in searchSlots and "-based" in self.sentences[index + 1]:
          fillerType = "ORG"
          thSlot = "gpe:headquarters_in_location"
          self.logger.info("extracting ORGANIZATION from additional sentence for based-slot")
          indexInSentence = 0
          while indexInSentence < len(curExList):
            curNer = curNerList[indexInSentence]
            if curNer == "ORGANIZATION":
              fillerCandidateString += curExList[indexInSentence]
              fillerOffsets += curOffsetList[indexInSentence]
              indexInSentence += 1
              curNer = curNerList[indexInSentence]
              while indexInSentence < len(curExList) and curNer == "ORGANIZATION":
                fillerCandidateString += " " + curExList[indexInSentence]
                fillerOffsets += "," + curOffsetList[indexInSentence]
                indexInSentence += 1
                curNer = curNerList[indexInSentence]
              break
            indexInSentence += 1
        if fillerCandidateString != "":
          foundAdditionalLocFiller = 1
          confidence = 0.6
          if len(curEx) > 1000:
            self.logger.debug("part0: found potential filler " + fillerCandidateString + " in a very long sentence with confidence " + str(confidence))
          else:
            self.logger.debug("part0: found potential filler " + fillerCandidateString + " in " + curEx + " with confidence " + str(confidence))
          curEx2 = curEx
          curOffsets2 = curOffsets
          if len(curEx) > 650:
            newStartEnd = self.cleanSentenceByLengthBasedOnOffsets(curEx, curOffsets, fillerOffsets)
            if len(newStartEnd) > 0:
              newStart, newEnd = newStartEnd
              curEx2 = " ".join(curEx.split()[newStart : newEnd])
              curOffsets2 = " ".join(curOffsetList[newStart : newEnd])
              if not "<name>" in curEx2 or not "<filler>" in curEx2:
                self.logger.warning("error in length cleaning: filler is no longer contained in sentence: " + curEx2)
              else:
                self.logger.debug("TEST ME: line 342: filler: " + fillerCandidateString + ", sentence after length cleaning: " + curEx2)
          candidate = [fillerCandidateString, curEx2, confidence, fillerOffsets, curOffsets2, d, fillerType]
          if self.isValidFiller(thSlot, fillerCandidateString) == 1 and thSlot in searchSlots:
            if not thSlot in self.globalSlot2fillerCandidatesAndConfidence:
              self.globalSlot2fillerCandidatesAndConfidence[thSlot] = []
            if not candidate in self.globalSlot2fillerCandidatesAndConfidence[thSlot]:
              self.globalSlot2fillerCandidatesAndConfidence[thSlot].append(candidate)
            if not thSlot in self.slot2fillerCandidates:
              self.slot2fillerCandidates[thSlot] = []
            if not [fillerCandidateString, curEx2, fillerOffsets, curOffsets2, d, fillerType] in self.slot2fillerCandidates[thSlot]:
              self.slot2fillerCandidates[thSlot].append([fillerCandidateString, curEx2, fillerOffsets, curOffsets2, d, fillerType])
          else: 
            self.logger.debug("no valid filler for slot " + thSlot + ": " + fillerCandidateString)

      if (self.isPerson == 1 or self.isOrg == 1) and len(nameOffsetsToReplace) == 0 and isSentenceAdditional(additionalNameOffsetsFlattened, curOffsetList):
        # 1a.:
        # extract potential filler directly
        for anoChain in additionalNameOffsets:
         found = 0
         foundTheLemma = 0
         foundNameOffsetList = []
         for ano in anoChain:
          anoSplitted = ano.split(',')
          anoStart = int(anoSplitted[0])
          anoEnd = int(anoSplitted[-1]) # use these variables when regarding fillers with spaces
          if anoStart >= startOffset and anoEnd <= endOffset: # we are in the correct sentence
            curLemmaList = self.lemmasInSentence[index].split()
            # (i) search for trigger words with hyphen (like '-year-old')
            for thSlot in self.triggerWordsHyphen:
              thList = self.triggerWordsHyphen[thSlot]
              for th in thList:
                thEscaped = re.escape(th)
                matchResult = getOffsetsForRegex(thEscaped, curEx)
                for match in matchResult:
                  thStartInd = match[0]
                  thEndInd = match[1]
                  w = " ".join(curExList[thStartInd:thEndInd])
                  if re.search(ur'^' + thEscaped, w, re.UNICODE):
                    thStartInd -= 1
                    w = " ".join(curExList[thStartInd:thEndInd])
                  if thEndInd < len(curNerList) and curNerList[thEndInd] in ["PERSON", "ORGANIZATION"] and curExList[thEndInd] not in self.curName and curExList[thEndInd] not in ["man", "woman", "girl", "boy"]:
                    self.logger.debug("found additional information " + w + " in " + curEx + " which most probably belongs to another entity")
                    continue
                  self.logger.debug("found additional information in " + w)
                  fillerOffset = curOffsetList[thStartInd]
                  curFiller = re.sub(ur'\s*' + thEscaped + '.*?$', '', w, re.UNICODE)
                  # try to extend filler to the left
                  fillerNer = curNerList[thStartInd]
                  if fillerNer != 'O' and thSlot != "per:age":
                    # else: don't extend!
                    thStartInd -= 1
                    while thStartInd >= 0 and curNerList[thStartInd] == fillerNer:
                      curFiller = curExList[thStartInd] + " " + curFiller
                      fillerOffset = curOffsetList[thStartInd] + "," + fillerOffset
                      thStartInd -= 1
                    thStartInd += 1 # it has been reduced once too often
                  if thSlot == "per:age" and curFiller in self.text2digit:
                    curFiller = self.text2digit[curFiller]
                    if "month-old" in th:
                      if curFiller.isdigit():
                        curFiller = month2year(curFiller)
                  confidence = 0.5
                  if thStartInd - 1 >= 0 and curLemmaList[thStartInd - 1] == "the":
                    confidence = 0.9
                    foundTheLemma = 1
                    tmpList = [curOffsetList[thStartInd - 1]]  # include "the" in the replacement
                    tmpList.extend(fillerOffset.split(","))
                    foundNameOffsetList.append(tmpList)
                    # extend name offsets also with coreffered mentions
                    for chainElement in anoChain:
                      if not "," + chainElement in "," + fillerOffset and not chainElement + "," in fillerOffset + "," and not "," + fillerOffset in "," + chainElement and not fillerOffset + "," in chainElement + ",":
                        additionalCorefOffsets.append(chainElement)
                  if len(curEx) > 1000:
                    self.logger.debug("part1: found potential filler " + curFiller + " in a very long sentence with confidence " + str(confidence))
                  else:
                    self.logger.debug("part1: found potential filler " + curFiller + " in " + curEx + " with confidence " + str(confidence))
                  curEx2 = curEx
                  curOffsets2 = curOffsets
                  if len(curEx) > 650:
                    newStartEnd = self.cleanSentenceByLengthBasedOnOffsets(curEx, curOffsets, fillerOffset)
                    if len(newStartEnd) > 0:
                      newStart, newEnd = newStartEnd
                      curEx2 = " ".join(curEx.split()[newStart : newEnd])
                      curOffsets2 = " ".join(curOffsetList[newStart : newEnd])
                      if not "<name>" in curEx2 or not "<filler>" in curEx2:
                        self.logger.warning("ERROR in length cleaning: filler is no longer contained in sentence: " + curEx2)
                      else:
                        self.logger.debug("TEST ME: line 383: filler: " + curFiller + ", sentence after length cleaning: " + curEx2)
                  fillerType = getFillerType(thSlot)
                  candidate = [curFiller, curEx2, confidence, fillerOffset, curOffsets2, d, fillerType]
                  if self.isValidFiller(thSlot, curFiller) == 1 and thSlot in searchSlots:
                    if not thSlot in self.globalSlot2fillerCandidatesAndConfidence:
                      self.globalSlot2fillerCandidatesAndConfidence[thSlot] = []
                    if not candidate in self.globalSlot2fillerCandidatesAndConfidence[thSlot]:
                      self.globalSlot2fillerCandidatesAndConfidence[thSlot].append(candidate)
                    if not thSlot in self.slot2fillerCandidates:
                      self.slot2fillerCandidates[thSlot] = []
                    if not [curFiller, curEx2, fillerOffset, curOffsets2, d, fillerType] in self.slot2fillerCandidates[thSlot]:
                      self.slot2fillerCandidates[thSlot].append([curFiller, curEx2, fillerOffset, curOffsets2, d, fillerType])
                  else:
                    self.logger.debug("no valid filler for slot " + thSlot + ": " + curFiller)
                  found = 1
            # (ii) search for possible predefined fills (like CEO for per:title)
            for slot in self.slot2possibleFills:
              if self.isOrg == 1 and ("per:" in slot or "gpe:" in slot):
                continue
              if self.isPerson == 1 and ("org:" in slot or "gpe:" in slot):
                continue
              if self.isLoc == 1 and ("org:" in slot or "per:" in slot):
                continue
              fillList = self.slot2possibleFills[slot]
              foundList = []
              for fill in fillList:
                fillLc = string.lower(fill)
                fillEscaped = re.escape(fillLc)
                curExLc = string.lower(curEx)
                curExLcCleaned = curExLc
                curExCleaned = curEx 
                matchResult = getOffsetsForRegex(fillEscaped, curExLcCleaned)
                curExCleanedList = curExCleaned.split()
                for match in matchResult:
                  thStartInd = match[0]
                  thEndInd = match[1]
                  # try to extend filler to the left:
                  if slot == "per:title":
                    while thStartInd > 0 and string.lower(curExCleanedList[thStartInd - 1]) in self.fillerPrefixes:
                      thStartInd -= 1
                  curFiller = " ".join(curExCleanedList[thStartInd:thEndInd])
                  if " " + fillLc + " " not in " " + string.lower(curFiller) + " ":
                    continue # did not match a full word!
                  if slot == "per:title" and thEndInd < len(curNerList) and curNerList[thEndInd] in ["PERSON", "ORGANIZATION"] and curExCleanedList[thEndInd] not in self.curName:
                    if len(curEx) > 1000:
                      self.logger.debug("found title " + curFiller + " which most probably belongs to another entity in a very long sentence - skipping...")
                    else:
                      self.logger.debug("found title " + curFiller + " which most probably belongs to another entity in sentence " + curEx + " - skipping...")
                    continue # belongs to another name most probably
                  fillerOffset = ",".join(curOffsetList[thStartInd:thEndInd])
                  confidence = 0.5
                  if thStartInd - 1 >= 0 and curLemmaList[thStartInd - 1] == "the":
                    confidence = 0.9
                    foundTheLemma = 1
                    tmpList = [curOffsetList[thStartInd - 1]]  # include "the" in the replacement
                    tmpList.extend(fillerOffset.split(","))
                    foundNameOffsetList.append(tmpList)
                    # extend name offsets also with coreffered mentions
                    for chainElement in anoChain:
                      if not "," + chainElement in "," + fillerOffset and not chainElement + "," in fillerOffset + "," and not "," + fillerOffset in "," + chainElement and not fillerOffset + "," in chainElement + ",":
                        additionalCorefOffsets.append(chainElement)
                  if len(curEx) > 1000:
                    self.logger.debug("partX: found potential string slot filler " + curFiller + " in a very long sentence with confidence " + str(confidence))
                  else:
                    self.logger.debug("partX: found potential string slot filler " + curFiller + " in " + curEx + " with confidence " + str(confidence))
                  curEx2 = curEx
                  curOffsets2 = curOffsets
                  if len(curEx) > 650:
                    newStartEnd = self.cleanSentenceByLengthBasedOnOffsets(curEx, curOffsets, fillerOffset)
                    if len(newStartEnd) > 0:
                      newStart, newEnd = newStartEnd
                      curEx2 = " ".join(curEx.split()[newStart : newEnd])
                      curOffsets2 = " ".join(curOffsetList[newStart : newEnd])
                      if not "<name>" in curEx2 or not "<filler>" in curEx2:
                        self.logger.warning("error in length cleaning: filler is no longer contained in sentence: " + curEx2)
                      else:
                        self.logger.info("filler: " + curFiller + ", sentence after length cleaning: " + curEx2)

                  found = 1
                  candidate = [curFiller, curEx2, confidence, fillerOffset, curOffsets2, d, "STRING"]

                  if slot in searchSlots and not slot in self.globalSlot2fillerCandidatesAndConfidence:
                    self.globalSlot2fillerCandidatesAndConfidence[slot] = []
                  if self.isValidFiller(slot, curFiller) == 1 and slot in searchSlots:
                    foundList.append(candidate)
                  else:
                    self.logger.debug("no valid filler for slot " + slot + ": " + curFiller)
              # only append longest filler per match (e.g. only "chief executive" and not both "chief executive" and "executive")
              toDelete = []
              for i1 in range(len(foundList)):
                item1 = foundList[i1]
                for i2 in range(i1 + 1, len(foundList)):
                  item2 = foundList[i2]
                  if " " + item2[3] + " " in " " + item1[3] + " ":
                    toDelete.append(item2)
                  elif " " + item1[3] + " " in " " + item2[3] + " ":
                    toDelete.append(item1)
              for td in toDelete:
                if td in foundList:
                  foundList.remove(td)
              for candidate in foundList:
                if not candidate in self.globalSlot2fillerCandidatesAndConfidence[slot]:
                  self.globalSlot2fillerCandidatesAndConfidence[slot].append(candidate)
                if not slot in self.slot2fillerCandidates:
                  self.slot2fillerCandidates[slot] = []
                if not [candidate[0], candidate[1], candidate[3], candidate[4], candidate[5], candidate[6]] in self.slot2fillerCandidates[slot]:
                  self.slot2fillerCandidates[slot].append([candidate[0], candidate[1], candidate[3], candidate[4], candidate[5], candidate[6]])
            # (iii): search for trigger words (e.g. "the child") and append them to foundNameOffsetList
            for trig in self.triggerWords:
              trigLc = string.lower(trig)
              fillEscaped = re.escape(trigLc)
              curExLc = string.lower(curEx)
              curExLcCleaned = curExLc
              curExCleaned = curEx
              matchResult = getOffsetsForRegex(fillEscaped, curExLcCleaned)
              curExCleanedList = curExCleaned.split()
              for match in matchResult:
                thStartInd = match[0]
                thEndInd = match[1]
                if thStartInd - 1 >= 0 and (curLemmaList[thStartInd - 1] in ["the", "that", "this"]):
                  foundTheLemma = 1
                  self.logger.info("found trigger word " + trig + " in sentence " + curEx)
                  fillerOffset = ",".join(curOffsetList[thStartInd : thEndInd])
                  tmpList = [curOffsetList[thStartInd - 1]]  # include "the" in the replacement
                  tmpList.extend(fillerOffset.split(","))
                  foundNameOffsetList.append(tmpList)
                  # extend name offsets also with coreffered mentions
                  for chainElement in anoChain:
                    if not "," + chainElement in "," + fillerOffset and not chainElement + "," in fillerOffset + "," and not "," + fillerOffset in "," + chainElement and not fillerOffset + "," in chainElement + ",":
                      additionalCorefOffsets.append(chainElement)

                if self.isOrg == 1 and "org:location_of_headquarters" in searchSlots and thStartInd -1 >= 0 and curNerList[thStartInd - 1] == "LOCATION":
                  possibleHeadquarter = curExCleanedList[thStartInd - 1]
                  possibleHqOffsets = curOffsetList[thStartInd - 1]
                  thStartInd = thStartInd - 1
                  while thStartInd -1 >= 0 and curNerList[thStartInd - 1] == "LOCATION":
                    possibleHeadquarter = curExCleanedList[thStartInd - 1] + " " + possibleHeadquarter
                    possibleHqOffsets = curOffsetList[thStartInd - 1] + "," + possibleHqOffsets
                    thStartInd = thStartInd - 1
                  if thStartInd - 1 >= 0 and (curLemmaList[thStartInd - 1] in ["the", "that", "this"]):
                    foundTheLemma = 1
                    self.logger.info("found trigger word " + trig + " in sentence " + curEx)
                    triggerOffset = ",".join(curOffsetList[thStartInd : thEndInd])
                    tmpList = [curOffsetList[thStartInd - 1]]  # include "the" in the replacement
                    tmpList.extend(triggerOffset.split(","))
                    foundNameOffsetList.append(tmpList)
                    # extend name offsets also with coreffered mentions
                    for chainElement in anoChain:
                      if not "," + chainElement in "," + triggerOffset and not chainElement + "," in triggerOffset + "," and not "," + triggerOffset in "," + chainElement and not triggerOffset + "," in chainElement + ",":
                        additionalCorefOffsets.append(chainElement)

                    # possible headquarter
                    curEx2 = curEx
                    curOffsets2 = curOffsets
                    if len(curEx) > 650:
                      newStartEnd = self.cleanSentenceByLengthBasedOnOffsets(curEx, curOffsets, possibleHqOffsets)
                      if len(newStartEnd) > 0:
                        newStart, newEnd = newStartEnd
                        curEx2 = " ".join(curEx.split()[newStart : newEnd])
                        curOffsets2 = " ".join(curOffsetList[newStart : newEnd])
                        if not "<name>" in curEx2 or not "<filler>" in curEx2:
                          self.logger.warning("error in length cleaning: filler is no longer contained in sentence: " + curEx2)
                        else:
                          self.logger.debug("TEST ME: line 464: filler: " + possibleHeadquarter + ", sentence after length cleaning: " + curEx2)
                    if self.isValidFiller("org:location_of_headquarters", possibleHeadquarter) == 1:
                      self.logger.info("found headquarter " + possibleHeadquarter + " in sentence " + curEx2)
                      candidate = [possibleHeadquarter, curEx2, 0.9, possibleHqOffsets, curOffsets2, d, "STRING"]
                      if not "org:location_of_headquarters" in self.globalSlot2fillerCandidatesAndConfidence:
                        self.globalSlot2fillerCandidatesAndConfidence["org:location_of_headquarters"] = []
                      self.globalSlot2fillerCandidatesAndConfidence["org:location_of_headquarters"].append(candidate)
                      if not "org:location_of_headquarters" in self.slot2fillerCandidates:
                        self.slot2fillerCandidates["org:location_of_headquarters"] = []
                      self.slot2fillerCandidates["org:location_of_headquarters"].append([possibleHeadquarter, curEx2, possibleHqOffsets, curOffsets2, d, "STRING"])
                    else:
                      self.logger.debug("no valid filler for slot org:location_of_headquarters: " + possibleHeadquarter)
                found = 1

            if found == 1:
              break # chain has already been appended
         if found * foundTheLemma == 1: # both are 1
           for tmpList in foundNameOffsetList:
             if not tmpList in nameOffsetsToReplace:
               nameOffsetsToReplace.append(tmpList)
      if len(nameOffsetsToReplace) == 0 and foundAdditionalLocFiller == 0:  # means: no <name> in curEx!
        self.logger.error("did not find name " + self.curName + " in sentence although it should be there: " + d + " " + curOffsets + " " + curEx)
        continue

      # 2.: we have a sentence with a name:
      # prepare datastructures
      for s in searchSlots:
        if not s in self.slot2fillerCandidates:
          self.slot2fillerCandidates[s] = []
        # decide how slot will be evaluated
        if s in self.slotsForPatternMatching:
          if not s in self.forPatternMatcher:
            self.forPatternMatcher[s] = []
        elif s in self.slot2proximity:
          if not s in self.forProximity:
            self.forProximity[s] = []
        else:
          if not s in self.forClassifier:
            self.forClassifier[s] = []


      # 2a: find candidates for slots with NER information
      candidateList = []
      for ner in self.type2slots:
        candidateList = []
        if ner == 'O':
          continue
        slotList = self.type2slots[ner]
        searchSlotList = []
        for s in slotList:
          if s in searchSlots:
            searchSlotList.append(s)
        if len(searchSlotList) == 0:
          continue

        # extract candidates
        curInd = 0
        while curInd < len(curNerList):
          prevTag = curNerList[curInd]
          fillerOffset = ""
          fillerCandidate = ""
          curTag = curNerList[curInd]
          if curTag == ner:
            fillerOffset += str(curOffsetList[curInd]) + ","
            fillerCandidate += curExList[curInd] + " "
            prevTag = curTag
            curInd += 1
            if curInd < len(curNerList):
              curTag = curNerList[curInd]
            while curTag == prevTag and curInd < len(curNerList):
              fillerOffset += str(curOffsetList[curInd]) + ","
              fillerCandidate += curExList[curInd] + " "
              curInd += 1
              prevTag = curTag
              if curInd < len(curNerList):
                curTag = curNerList[curInd]
            fillerCandidate = fillerCandidate.strip()
            fillerOffset = fillerOffset.strip(",")
            if fillerEqualsName(nameOffsets, fillerOffset) == 0 and fillerEqualsName(additionalNameOffsetsFlattened, fillerOffset) == 0:
              curCandidate, newOffsets = replaceOffsetListWithTag(curEx, curOffsetList, nameOffsetsToReplace, fillerOffset.split(","))
              if not "<name>" in curCandidate or not "<filler>" in curCandidate:
                self.logger.error("error in tag replacement: did not find both tags: " + curCandidate)
              else:
                curOffsetListNew = newOffsets.split()
                if len(curOffsetListNew) != len(curCandidate.split()):
                  self.logger.error("error after tag replacement: length of sentence and offsets are different: " + curCandidate + " --- " + newOffsets)
                if curCandidate != "":
                  # clean by length:
                  newStartEnd = self.cleanSentenceByLength(curCandidate)
                  if len(newStartEnd) > 0:
                    newStart, newEnd = newStartEnd
                    curCandidate2 = " ".join(curCandidate.split()[newStart : newEnd])
                    curOffsets2 = " ".join(curOffsetListNew[newStart : newEnd]) 
                    if not "<name>" in curCandidate2 or not "<filler>" in curCandidate2:
                      self.logger.error("error in length cleaning: filler is no longer contained in sentence: " + curCandidate2)
                    else:
                      fillerType = ner2fillerType(ner)
                      if not [fillerCandidate, curCandidate2, fillerOffset, curOffsets2, d, fillerType] in candidateList:
                        self.logger.info("filler candidate: " + fillerCandidate)
                        candidateList.append([fillerCandidate, curCandidate2, fillerOffset, curOffsets2, d, fillerType])
                        foundWithNer = 1
          else:
            curInd += 1

        # assign them to all slots in slotList with fitting type (PER, ORG)
        for s in searchSlotList:
          # clean candidateList:
          cleanedCandidateList = []
          for candidate in candidateList:
            if s == "per:employee_or_member_of" and ner == "LOCATION":
              # avoid FPs
              proof = candidate[1]
              if not "member" in proof and not "employee" in proof and not "employer" in proof and not "fellow" in proof and not "leader" in proof and not "of " + candidate[0] in proof:
                self.logger.info("per:employee_or_member_of: skipping filler candidate " + candidate[0] + " with proof " + proof)
                continue
            if self.isValidFiller(s, candidate[0]) == 1:
              cleanedCandidateList.append(candidate)
            else:
              self.logger.info("no valid filler for slot " + s + ": " + candidate[0])
          if self.isOrg == 1 and ("per:" in s or "gpe:" in s):
            continue
          if self.isPerson == 1 and ("org:" in s or "gpe:" in s):
            continue
          if self.isLoc == 1 and ("org:" in s or "per:" in s):
            continue
          for candidate in cleanedCandidateList:
            if not candidate in self.slot2fillerCandidates[s]:
              self.slot2fillerCandidates[s].append(candidate)
              if s in self.forPatternMatcher:
                self.forPatternMatcher[s].append(candidate)
              elif s in self.forProximity:
                self.forProximity[s].append(candidate)
              elif s in self.forClassifier:
                self.forClassifier[s].append(candidate)
              else:
                self.logger.error("did not find evaluation method for slot " + s)

      candidateList = []

      # 2b: find candidates for slots with trigger
      for s in searchSlots:
        # decide whether slot was for NER
        slotForNer = 0
        nerList = self.slot2types[s]
        if 'O' in nerList:
          nerList.remove('O')
        if len(nerList) > 0:
          slotForNer = 1
          continue # change this to apply both NER and something else (e.g. triggers)
        if slotForNer == 0 and not s in self.slot2possibleFills and s != "org:website" and index == 0: # all are 0, but output only once
          self.logger.warning("cannot find examples for slot " + s)
          continue

        for candidate in candidateList:
          if not candidate in self.slot2fillerCandidates[s]:
            self.slot2fillerCandidates[s].append(candidate)
            if s in self.forPatternMatcher:
              self.forPatternMatcher[s].append(candidate)
            elif s in self.forClassifier:
              self.forClassifier[s].append(candidate)
            elif s in self.forProximity:
              self.forProximity[s].append(candidate)

      # search for trigger-hyphen words
      if self.isPerson == 1 or self.isOrg == 1:
        for thSlot in self.triggerWordsHyphen:
          triggerHyphenCandidates = []
          if not thSlot in searchSlots:
            continue
          thList = self.triggerWordsHyphen[thSlot]
          for th in thList:
            thEscaped = re.escape(th)
            matchResult = getOffsetsForRegex(thEscaped, curEx)
            for match in matchResult:
              thStartInd = match[0]
              thEndInd = match[1]

              curLemmaList = self.lemmasInSentence[index].split()
              w = " ".join(curExList[thStartInd:thEndInd])
              if thEndInd < len(curNerList) and curNerList[thEndInd] in ["PERSON", "ORGANIZATION"] and curExList[thEndInd] not in self.curName:
                self.logger.debug("found additional filler candidate " + w + " in " + curEx + " which most probably belongs to another entity")
                continue
              self.logger.info("found additional filler candidate in " + w)
              fillerOffset = curOffsetList[thStartInd]
              curFiller = re.sub(ur'' + thEscaped + '.*?$', '', w, re.UNICODE)
              curEx_tmp = re.sub(ur'' + thEscaped, ' ' + th, curEx, re.UNICODE)
              curOffsetList_tmp = copy.deepcopy(curOffsetList)
              thOffset = str(int(fillerOffset) + len(curFiller) + 1)
              curOffsetList_tmp.insert(thStartInd + 1, thOffset)
              # try to extend filler to the left
              fillerNer = curNerList[thStartInd]
              if fillerNer != 'O':
                # else: don't extend!
                thStartInd -= 1
                while thStartInd >= 0 and curNerList[thStartInd] == fillerNer:
                  curFiller = curExList[thStartInd] + " " + curFiller
                  fillerOffset = curOffsetList[thStartInd] + "," + fillerOffset
                  thStartInd -= 1
                thStartInd += 1 # it has been reduced once too often
              if thSlot == "per:age" and curFiller in self.text2digit:
                curFiller = self.text2digit[curFiller]
                if "month-old" in th:
                  self.logger.info("found age in months: " + str(curFiller))
                  if curFiller.isdigit():
                    curFiller = month2year(curFiller)
              confidence = 0.5
              if thStartInd - 1 >= 0 and curLemmaList[thStartInd - 1] == "the":
                confidence = 0.9

              # replace filler with <filler>!
              curEx2, newOffsets2 = replaceOffsetListWithTag(curEx_tmp, curOffsetList_tmp, nameOffsetsToReplace, fillerOffset.split(","))
              if not "<name>" in curEx2 or not "<filler>" in curEx2:
                self.logger.error("error in tag replacement: did not find both tags: " + curEx2)
              else:
                curOffsetListNew = newOffsets2.split()
                if len(curOffsetListNew) != len(curEx2.split()):
                  self.logger.error("error after tag replacement: length of sentence and offsets are different: " + curEx2 + " --- " + newOffsets2)
                if curEx2 != "":
                  # clean by length:
                  newStartEnd = self.cleanSentenceByLength(curEx2)
                  if len(newStartEnd) > 0:
                    newStart, newEnd = newStartEnd
                    curEx22 = " ".join(curEx2.split()[newStart : newEnd])
                    curOffsets22 = " ".join(curOffsetListNew[newStart : newEnd])
                    if not "<name>" in curEx22 or not "<filler>" in curEx22:
                      self.logger.error("error in length cleaning: filler is no longer contained in sentence: " + curEx22)
                    else:
                      fillerType = getFillerType(thSlot)
                      candidate = [curFiller, curEx22, fillerOffset, curOffsets22, d, fillerType]
                      if fillerEqualsName(nameOffsets, fillerOffset) == 0:
                        if len(curEx22) > 1000:
                          self.logger.debug("part2: found potential filler " + curFiller + " in a very long sentence")
                        else:
                          self.logger.debug("part2: found potential filler " + curFiller + " in " + curEx22)
                        if self.isValidFiller(thSlot, curFiller) == 1:
                          if th == "-based" and "<filler> -based <name>" in curEx22:
                            candidateConf = [candidate[0], candidate[1], 1.0, candidate[2], candidate[3], candidate[4], candidate[5]]
                            if not thSlot in self.globalSlot2fillerCandidatesAndConfidence:
                              self.globalSlot2fillerCandidatesAndConfidence[thSlot] = []
                            if not candidateConf in self.globalSlot2fillerCandidatesAndConfidence[thSlot]:
                              self.globalSlot2fillerCandidatesAndConfidence[thSlot].append(candidateConf)
                              self.logger.debug("part2: appending to global candidates with conf=1.0: " + str(candidateConf))
                          else:
                            if not candidate in triggerHyphenCandidates:
                              self.logger.debug("part2: appending " + str(candidate))
                              triggerHyphenCandidates.append(candidate)
                          if not thSlot in self.slot2fillerCandidates:
                            self.slot2fillerCandidates[thSlot] = []
                          if not candidate in self.slot2fillerCandidates[thSlot]:
                            self.slot2fillerCandidates[thSlot].append(candidate)
                        else:
                          self.logger.info("no valid filler for slot " + thSlot + ": " + curFiller)

          for candidate in triggerHyphenCandidates:
            if thSlot in self.forPatternMatcher:
              self.forPatternMatcher[thSlot].append(candidate)
            elif thSlot in self.forClassifier:
              self.forClassifier[thSlot].append(candidate)
            elif thSlot in self.forProximity:
              self.forProximity[thSlot].append(candidate)


      # search for predefined slot fills (like CEO for per:title):
      for slot in self.slot2possibleFills:
        if not slot in searchSlots:
          continue
        if self.isOrg == 1 and ("per:" in slot or "gpe:" in slot):
          continue
        if self.isPerson == 1 and ("org:" in slot or "gpe:" in slot):
          continue
        if self.isLoc == 1 and ("org:" in slot or "per:" in slot):
          continue
        fillList = self.slot2possibleFills[slot]
        foundList = []
        for fill in fillList:
          fillLc = string.lower(fill)
          fillLcEscaped = re.escape(fillLc)
          curExLc = string.lower(curEx)
          curExLcCleaned = curExLc
          curExCleaned = curEx
          curExCleanedList = curExCleaned.split()
          matchResult = getOffsetsForRegex(fillLcEscaped, curExLcCleaned)
          for match in matchResult:
            thStartInd = match[0]
            thEndInd = match[1]
            # try to extend filler to the left:
            if slot == "per:title":
              while thStartInd > 0 and string.lower(curExCleanedList[thStartInd - 1]) in self.fillerPrefixes:
                thStartInd -= 1
            curFiller = " ".join(curExCleanedList[thStartInd:thEndInd])
            if " " + fillLc + " " not in " " + string.lower(curFiller) + " ":
              continue # did not match a full word!
            if slot == "per:title" and thEndInd < len(curNerList) and curNerList[thEndInd] in ["PERSON", "ORGANIZATION"] and curExCleanedList[thEndInd] not in self.curName:
              if len(curEx) > 1000:
                self.logger.debug("found title " + curFiller + " which most probably belongs to another entity in a very long sentence -- skipping...")
              else:
                self.logger.debug("found title " + curFiller + " which most probably belongs to another entity in sentence "   + curEx + " - skipping...")
              continue # belongs to another name most probably
            fillerOffset = ",".join(curOffsetList[thStartInd:thEndInd])
            if len(curEx) > 1000:
              self.logger.debug("STRINGSLOT2 " + slot + ": found possible fill -- " + curFiller + " -- in a very long sentence at -- " + fillerOffset)
            else:
              self.logger.debug("STRINGSLOT2 " + slot + ": found possible fill -- " + curFiller + " -- in -- " + curEx + " -- at -- " + fillerOffset)
            # replace filler with <filler>!
            curEx2, newOffsets2 = replaceOffsetListWithTag(curExCleaned, curOffsetList, nameOffsetsToReplace, fillerOffset.split(","))
            if not "<name>" in curEx2 or not "<filler>" in curEx2:
              self.logger.error("error in tag replacement: did not find both tags: " + curEx2)
            else:
              curOffsetListNew = newOffsets2.split()
              if len(curOffsetListNew) != len(curEx2.split()):
                self.logger.error("error after tag replacement: length of sentence and offsets are different: " + curEx2 + " --- " + newOffsets2)
              if curEx2 != "":
                # clean by length:
                newStartEnd = self.cleanSentenceByLength(curEx2)
                if len(newStartEnd) > 0:
                  newStart, newEnd = newStartEnd
                  curEx22 = " ".join(curEx2.split()[newStart : newEnd])
                  curOffsets22 = " ".join(curOffsetListNew[newStart : newEnd]) 
                  if not "<name>" in curEx22 or not "<filler>" in curEx22:
                    self.logger.error("error in length cleaning: filler is no longer contained in sentence: " + curEx22)
                  else:
                    candidate = [curFiller, curEx22, fillerOffset, curOffsets22, d, "STRING"]
                    if self.isValidFiller(slot, curFiller) == 1:
                      if not candidate in foundList:
                        foundList.append(candidate)
                    else:
                      self.logger.info("no valid filler for slot " + slot + ": " + curFiller)
        # only append longest filler per match (e.g. only "chief executive" and not both "chief executive" and "executive")
        toDelete = []
        for i1 in range(len(foundList)):
          item1 = foundList[i1]
          for i2 in range(i1 + 1, len(foundList)):
            item2 = foundList[i2]
            if " " + item2[2] + " " in " " + item1[2] + " ":
              toDelete.append(item2)
            elif " " + item1[2] + " " in " " + item2[2] + " ":
              toDelete.append(item1)
        for td in toDelete:
          if td in foundList:
            foundList.remove(td)
        for candidate in foundList:
          if not candidate in self.slot2fillerCandidates[slot]:
            self.slot2fillerCandidates[slot].append(candidate)
            if slot in self.forPatternMatcher:
              self.forPatternMatcher[slot].append(candidate)
            elif slot in self.forProximity:
              self.forProximity[slot].append(candidate)
            elif slot in self.forClassifier:
              self.forClassifier[slot].append(candidate)
            else:
              self.logger.error("did not find evaluation method for slot " + slot)

      # search for additional fillers which appear as pronoun mentions in the current sentence
      for slotPer in self.slotsWithPerFillers:
        if not slotPer in searchSlots:
          continue
        for pronounOffset in additionalCorefFillers:
          if pronounOffset in curOffsetList:
            self.logger.info("found pronoun which could be a filler in " + curEx)
            filler = additionalCorefFillers[pronounOffset][0]
            fillerOffset1 = pronounOffset
            fillerOffset2 = additionalCorefFillers[pronounOffset][1]
            # replace filler with <filler>!
            curEx2, newOffsets2 = replaceOffsetListWithTag(curEx, curOffsetList, nameOffsetsToReplace, fillerOffset1.split(","))
            if not "<name>" in curEx2 or not "<filler>" in curEx2:
              self.logger.error("error in tag replacement: did not find both tags: " + curEx2)
            else:
              curOffsetListNew = newOffsets2.split()
              if len(curOffsetListNew) != len(curEx2.split()):
                self.logger.error("error after tag replacement: length of sentence and offsets are different: " + curEx2 + " --- " + newOffsets2)
              if curEx2 != "":
                # clean by length:
                newStartEnd = self.cleanSentenceByLength(curEx2)
                if len(newStartEnd) > 0:
                  newStart, newEnd = newStartEnd
                  curEx22 = " ".join(curEx2.split()[newStart : newEnd])
                  curOffsets22 = " ".join(curOffsetListNew[newStart : newEnd])
                  if not "<name>" in curEx22 or not "<filler>" in curEx22:
                    self.logger.error("error in length cleaning: filler is no longer contained in sentence: " + curEx22)
                  else:
                    if not slotPer in self.slot2fillerCandidates:
                      self.slot2fillerCandidates[slotPer] = []
                    candidate = [filler, curEx22, fillerOffset1, curOffsets22, d, "PER"]
                    if self.isValidFiller(slotPer, filler) == 1:
                      self.slot2fillerCandidates[slotPer].append(candidate)
                      if slotPer in self.forPatternMatcher:
                        self.forPatternMatcher[slotPer].append(candidate)
                      elif slotPer in self.forClassifier:
                        self.forClassifier[slotPer].append(candidate)
                      elif slotPer in self.forProximity:
                        self.forProximity[slotPer].append(candidate)
                    else:
                      self.logger.info("no valid filler for slot " + slotPer + ": " + filler)

      # special case: pattern "employee of" or "member of" + acronym or ORGANIZATION
      if self.isPerson == 1 and "per:employee_or_member_of" in searchSlots:
        if " employee of " in curEx or " member of " in curEx:
          slotEmpl = "per:employee_of_member_of"
          for ind, word in enumerate(curExList):
            if word in ["employee", "member"] and ind < len(curExList) - 1:
              nextWord = curExList[ind + 1]
              if nextWord == "of" and ind + 1 < len(curExList) - 1:
                possibleOrg = curExList[ind + 2]
                fillerIndex = [ind + 2]
                if possibleOrg.isupper() or curNerList[ind + 2] == "ORGANIZATION": # we have an acronym or organization here! => pattern matches!
                  if curNerList[ind + 2] == "ORGANIZATION":
                    nextInd = ind + 2
                    while nextInd + 1 < len(curNerList) and curNerList[nextInd + 1] == "ORGANIZATION":
                      nextInd += 1
                      possibleOrg += " " + curExList[nextInd]
                      fillerIndex.append(nextInd)
                  alreadyExtracted = 0
                  # check whether filler has been extracted before
                  if not slotEmpl in self.slot2fillerCandidates:
                    pass
                  else:
                    for fillerCand in self.slot2fillerCandidates[slotEmpl]:
                      if fillerCand[0] == possibleOrg:
                        alreadyExtracted = 1
                  if alreadyExtracted == 0:
                    fillerOffset = curOffsetList[fillerIndex[0]]
                    for furtherIndex in range(1, len(fillerIndex)):
                      fillerOffset += "," + curOffsetList[fillerIndex[furtherIndex]]
                    # replace filler with <filler>!
                    curEx2, newOffsets2 = replaceOffsetListWithTag(curEx, curOffsetList, nameOffsetsToReplace, fillerOffset.split(","))
                    if not "<name>" in curEx2 or not "<filler>" in curEx2:
                      self.logger.error("error in tag replacement: did not find both tags: " + curEx2)
                    else:
                      curOffsetListNew = newOffsets2.split()
                      if len(curOffsetListNew) != len(curEx2.split()):
                        self.logger.error("error: after tag replacement: length of sentence and offsets are different: " + curEx2 + " --- " + newOffsets2)
                      if curEx2 != "":
                        # clean by length:
                        newStartEnd = self.cleanSentenceByLength(curEx2)
                        if len(newStartEnd) > 0:
                          newStart, newEnd = newStartEnd
                          curEx22 = " ".join(curEx2.split()[newStart : newEnd])
                          curOffsets22 = " ".join(curOffsetListNew[newStart : newEnd])
                          if not "<name>" in curEx22 or not "<filler>" in curEx22:
                            self.logger.error("error in length cleaning: filler is no longer contained in sentence: " + curEx22)
                          else:
                            if not slotEmpl in self.slot2fillerCandidates:
                              self.slot2fillerCandidates[slotEmpl] = []
                            candidate = [possibleOrg, curEx22, fillerOffset, curOffsets22, d, "ORG"]
                            self.logger.info("found candidate " + possibleOrg + " with per:empl pattern + acronym in context " + curEx22)
                            if self.isValidFiller(slotEmpl, possibleOrg):
                              self.slot2fillerCandidates[slotEmpl].append(candidate)
                              if slotEmpl in self.forPatternMatcher:
                                self.forPatternMatcher[slotEmpl].append(candidate)
                              elif slotEmpl in self.forClassifier:
                                self.forClassifier[slotEmpl].append(candidate)
                              elif slotEmpl in self.forProximity:
                                self.forProximity[slotEmpl].append(candidate)
                            else:
                              self.logger.info("no valid filler for slot " + slotEmpl + ": " + possibleOrg)


  def resetCandidatesForDoc(self):
    self.slot2fillerCandidates = {}

  def resetCandidates(self):
    self.forPatternMatcher = {}
    self.forClassifier = {}
    self.forProximity = {}
    self.globalSlot2fillerCandidatesAndConfidence = {}

  def setCurName(self, curName):
    self.curName = curName

  def setType(self, curType):
    triggerWordsORG = ["company", "firm", "agency", "bureau", "department", "office", "group", "organization"]
    triggerWordsPER = ["child"]
    if curType == 'PER':
      self.isOrg = 0
      self.isLoc = 0
      self.isPerson = 1
      self.triggerWords = triggerWordsPER
    elif curType == 'ORG':
      self.isOrg = 1
      self.isLoc = 0
      self.isPerson = 0
      self.triggerWords = triggerWordsORG
    else:
      self.isOrg = 0
      self.isLoc = 1
      self.isPerson = 0
      self.triggerWords = []


  def setTriggerWordsHyphen(self, triggerWordsHyphen):
    self.triggerWordsHyphen = triggerWordsHyphen

  def setCurSlot(self, curSlot):
    slot = curSlot
    if "cit" in slot or "countr" in slot or "province" in slot:
      slot = re.sub(ur'city', 'location', slot, re.UNICODE)
      slot = re.sub(ur'country', 'location', slot, re.UNICODE)
      slot = re.sub(ur'statesorprovinces', 'locations', slot, re.UNICODE)
      slot = re.sub(ur'cities', 'locations', slot, re.UNICODE)
      slot = re.sub(ur'countries', 'locations', slot, re.UNICODE)
      slot = re.sub(ur'stateorprovince', 'location', slot, re.UNICODE)
    self.curSlot = slot

  def __init__(self, slots, slot2possibleFills, slot2types, type2slots, slotsForPatternMatching, slot2proximity, websiteRegex, countries, states, cities, loggerMain):
    self.forPatternMatcher = {}
    self.forClassifier = {}
    self.forProximity = {}
    self.slot2possibleFills = slot2possibleFills
    self.slots = slots
    self.slot2types = slot2types
    self.type2slots = type2slots
    self.slotsForPatternMatching = slotsForPatternMatching
    self.slot2proximity = slot2proximity
    self.slot2fillerCandidates = {} # for analysis
    self.websiteRegex = websiteRegex
    self.curName = ""
    self.isPerson = -1
    self.isOrg = -1
    self.isLoc = -1
    self.triggerWordsHyphen = {}
    self.globalSlot2fillerCandidatesAndConfidence = {}
    self.fillerPrefixes = {}
    for fp in ["senior", "sr.", "chief", "co-", "vice", "executive", "junior", "jr", "jr." "sr", "acting", "associate", "assistant", "deputy", "general", "principal", "regional", "adjunct", "administrative", "advertising", "advisory", "business", "development", "certified", "corporate", "deputy", "distinguished", "division", "editorial", "faculty", "founding", "global", "group", "independent", "interim", "laboratory", "lead", "managing", "military", "political", "professional", "research", "supervising", "technical", "teaching", "visiting"]:
      self.fillerPrefixes[fp] = 1
    self.text2digit = {'one' : '1', 'two' : '2', 'three' : '3', 'four' : '4', 'five' : '5', 'six' : '6', 'seven' : '7', 'eight' : '8', 'nine' : '9', 'ten' : '10', 'eleven' : '11', 'twelve' : '12'}
    self.countries = countries
    self.locations = copy.deepcopy(countries)
    self.locations.update(states)
    self.locations.update(cities)
    self.curSlot = ""
    self.offset2ner = {}
    self.slotsWithPerFillers = ["per:children", "per:parents", "per:other_family", "per:siblings", "per:spouse", "org:employees_or_members", "gpe:employees_or_members", "org:students", "gpe:births_in_location", "gpe:deaths_in_location", "gpe:residents_of_location", "org:shareholders", "org:founded_by", "org:top_members_employees"]

    self.logger = loggerMain.getChild(__name__)
