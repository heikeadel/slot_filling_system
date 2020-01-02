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
import re
import string
from utilities import matchPattern, levenshtein, normalizeDate, getsubidx, getFillerType
from doQuery import queryTwoLocations
from readNerAndCoref import getSentenceAndOffsetsFromCoref

class Postprocessing:

  def cleanFillerType(self, slot, fillerType):
    # get possible filler type for slot
    if slot in ["per:employee_or_member_of", "org:members", "org:parents"]:
      referenceFillerType = ["ORG", "GPE"]
    elif slot in ["org:shareholders", "org:founded_by"]:
      referenceFillerType = ["PER", "ORG", "GPE"]
    else:
      referenceFillerType = [getFillerType(slot)]
    # clean hypo filler
    if fillerType in referenceFillerType:
      return fillerType
    else:
      return referenceFillerType[0] # guess filler type

  def getInferenceResults(self, candidate, slot):
    locationFiller = candidate[0]
    fillerConfidence = candidate[2]
    newCandidates = []
    if "cit" in slot:
      city = locationFiller.strip()
      if city in self.city2stateAndCountry:
        state, country = self.city2stateAndCountry[city]
        # search for city+state in corpus
        foundNewSlotState = 0
        foundNewSlotCountry = 0
        if state.strip() != "":
          queryResults = queryTwoLocations(city, state, self.terrierDir)
          newSlot = re.sub(ur'cities', 'statesorprovinces', slot, re.UNICODE)
          newSlot = re.sub(ur'city', 'stateorprovince', newSlot, re.UNICODE)
          for docId in queryResults:
            if foundNewSlotState == 1:
              break
            docPath = self.docId2path[docId]
            sentenceSplittingResults = getSentenceAndOffsetsFromCoref(docId, docPath)
            if len(sentenceSplittingResults) == 0:
              # something went wrong - skip document
              continue
            sentences, offsets = sentenceSplittingResults
            for s,o in zip(sentences, offsets):
              if city in s and state in s: # could be improved with pattern matching
                sentenceList = s.split()
                offsetList = o.split()
                newFillerStart = getsubidx(sentenceList, country.split())
                oldFillerStart = getsubidx(sentenceList, state.split())
                if len(newFillerStart) == 0 or len(oldFillerStart) == 0:
                  # something went wrong
                  continue
                closestNewFiller = -1
                closestOldFiller = -1
                closestDistance = len(sentenceList)
                for i in range(len(newFillerStart)):
                  for j in range(len(oldFillerStart)):
                    dist = abs(newFillerStart[i] - oldFillerStart[j])
                    if dist < closestDistance:
                      closestDistance = dist
                      closestNewFiller = i
                      closestOldFiller = j
                newOffsetStart = offsetList[newFillerStart[closestNewFiller]]
                newOffsetEnd = str(int(newOffsetStart) + len(state) - 1)
                newFillerOffset = docId + ":" + newOffsetStart + "-" + newOffsetEnd  # take the filler most next to the locationFiller?
                proofStart = offsetList[0]
                proofEnd = str(int(offsetList[-1]) - 1)
                if int(proofEnd) - int(proofStart) > 150:
                  minIndex = min(newFillerStart[closestNewFiller], oldFillerStart[closestOldFiller])
                  maxIndex = max(newFillerStart[closestNewFiller], oldFillerStart[closestOldFiller])
                  if int(offsetList[maxIndex]) - int(offsetList[minIndex]) <= 150:
                    while minIndex > 0 and maxIndex < len(offsetList) and int(offsetList[maxIndex]) - int(offsetList[minIndex]) <= 150:
                      minIndex -= 1
                      maxIndex += 1
                    newProofOffset = docId + ":" + offsetList[minIndex + 1] + "-" + str(int(offsetList[maxIndex -1]) - 1)
                    newCandidates.append([newSlot, state, newFillerOffset, newProofOffset, s])
                    foundNewSlotState = 1
                    self.logger.info("found inference in sentence: " + s)
                    break
                else:
                  newProofOffset = docId + ":" + proofStart + "-" + proofStart
                  newCandidates.append([newSlot, state, newFillerOffset, newProofOffset, s])
                  foundNewSlotState = 1
                  self.logger.info("found inference in sentence: " + s)
                  break
        if country.strip() != "":
          # search for city+country in corpus
          queryResults2 = queryTwoLocations(city, country, self.terrierDir)
          newSlot = re.sub(ur'cities', 'countries', slot, re.UNICODE)
          newSlot = re.sub(ur'city', 'country', newSlot, re.UNICODE)
          for docId in queryResults2:
            if foundNewSlotCountry == 1: # only one inference per filler!
              break
            docPath = self.docId2path[docId]
            sentenceSplittingResults = getSentenceAndOffsetsFromCoref(docId, docPath)
            if len(sentenceSplittingResults) == 0:
              # something went wrong - skip document
              continue
            sentences, offsets = sentenceSplittingResults
            for s,o in zip(sentences, offsets):
              if city in s and country in s: # could be improved with pattern matching
                sentenceList = s.split()
                offsetList = o.split()
                newFillerStart = getsubidx(sentenceList, country.split())
                oldFillerStart = getsubidx(sentenceList, state.split())
                if len(newFillerStart) == 0 or len(oldFillerStart) == 0:
                  # something went wrong
                  continue
                closestNewFiller = -1
                closestOldFiller = -1
                closestDistance = len(sentenceList)
                for i in range(len(newFillerStart)):
                  for j in range(len(oldFillerStart)):
                    dist = abs(newFillerStart[i] - oldFillerStart[j])
                    if dist < closestDistance:
                      closestDistance = dist
                      closestNewFiller = i
                      closestOldFiller = j
                newOffsetStart = offsetList[newFillerStart[closestNewFiller]]
                newOffsetEnd = str(int(newOffsetStart) + len(country) - 1)
                newFillerOffset = docId + ":" + newOffsetStart + "-" + newOffsetEnd  # take the filler most next to the locationFiller?
                proofStart = offsetList[0]
                proofEnd = str(int(offsetList[-1]) - 1)
                if int(proofEnd) - int(proofStart) > 150:
                  minIndex = min(newFillerStart[closestNewFiller], oldFillerStart[closestOldFiller])
                  maxIndex = max(newFillerStart[closestNewFiller], oldFillerStart[closestOldFiller])
                  if int(offsetList[maxIndex]) - int(offsetList[minIndex]) <= 150:
                    while minIndex > 0 and maxIndex < len(offsetList) and int(offsetList[maxIndex]) - int(offsetList[minIndex]) <= 150:
                      minIndex -= 1
                      maxIndex += 1
                    newProofOffset = docId + ":" + offsetList[minIndex + 1] + "-" + str(int(offsetList[maxIndex -1]) - 1)
                    newCandidates.append([newSlot, country, newFillerOffset, newProofOffset, s])
                    foundNewSlotCountry = 1
                    self.logger.info("found inference in sentence: " + s)
                    break
                else:
                  newProofOffset = docId + ":" + proofStart + "-" + proofStart
                  newCandidates.append([newSlot, country, newFillerOffset, newProofOffset, s])
                  foundNewSlotCountry = 1
                  self.logger.info("found inference in sentence: " + s)
                  break

          if foundNewSlotCountry == 0: # prefer city+country sentences over state+country sentences but use state+country if there is no result for city+country
            if foundNewSlotState == 1:
              # search for state+country in corpus
              if state.strip() != "":
                queryResults3 = queryTwoLocations(state, country, self.terrierDir)
                newSlot = re.sub(ur'cities', 'countries', slot, re.UNICODE)
                newSlot = re.sub(ur'city', 'country', newSlot, re.UNICODE)
                foundNewSlotCountry = 0
                for docId in queryResults3:
                  if foundNewSlotCountry == 1: # only one inference per filler!
                    break
                  docPath = self.docId2path[docId]
                  sentenceSplittingResults = getSentenceAndOffsetsFromCoref(docId, docPath)
                  if len(sentenceSplittingResults) == 0:
                    # something went wrong - skip document
                    continue
                  sentences, offsets = sentenceSplittingResults
                  for s,o in zip(sentences, offsets):
                    if state in s and country in s:
                      sentenceList = s.split()
                      offsetList = o.split()
                      newFillerStart = getsubidx(sentenceList, country.split())
                      oldFillerStart = getsubidx(sentenceList, state.split())
                      if len(newFillerStart) == 0 or len(oldFillerStart) == 0:
                        # something went wrong
                        continue
                      closestNewFiller = -1
                      closestOldFiller = -1
                      closestDistance = len(sentenceList)
                      for i in range(len(newFillerStart)):
                        for j in range(len(oldFillerStart)):
                          dist = abs(newFillerStart[i] - oldFillerStart[j])
                          if dist < closestDistance:
                            closestDistance = dist
                            closestNewFiller = i
                            closestOldFiller = j
                      newOffsetStart = offsetList[newFillerStart[closestNewFiller]]
                      newOffsetEnd = str(int(newOffsetStart) + len(country) - 1)
                      newFillerOffset = docId + ":" + newOffsetStart + "-" + newOffsetEnd  # take the filler most next to the locationFiller?
                      proofStart = offsetList[0]
                      proofEnd = str(int(offsetList[-1]) - 1)
                      if int(proofEnd) - int(proofStart) > 150:
                        minIndex = min(newFillerStart[closestNewFiller], oldFillerStart[closestOldFiller])
                        maxIndex = max(newFillerStart[closestNewFiller], oldFillerStart[closestOldFiller])
                        if int(offsetList[maxIndex]) - int(offsetList[minIndex]) <= 150:
                          while minIndex > 0 and maxIndex < len(offsetList) and int(offsetList[maxIndex]) - int(offsetList[minIndex]) <= 150:
                            minIndex -= 1
                            maxIndex += 1
                          newProofOffset = docId + ":" + offsetList[minIndex + 1] + "-" + str(int(offsetList[maxIndex -1]) - 1)
                          newCandidates.append([newSlot, country, newFillerOffset, newProofOffset, s])
                          foundNewSlotCountry = 1
                          self.logger.info("found inference in sentence: " + s)
                          break
                      else:
                        newProofOffset = docId + ":" + proofStart + "-" + proofStart
                        newCandidates.append([newSlot, country, newFillerOffset, newProofOffset, s])
                        foundNewSlotCountry = 1
                        self.logger.info("found inference in sentence: " + s)
                        break

    elif "state" in slot:
      state = locationFiller.strip()
      if state in self.state2country:
        country = self.state2country[state]
        if country.strip() != "":
          # search for state+country in corpus
          queryResults = queryTwoLocations(state, country, self.terrierDir)
          newSlot = re.sub(ur'statesorprovinces', 'countries', slot, re.UNICODE)
          newSlot = re.sub(ur'stateorprovince', 'country', newSlot, re.UNICODE)
          foundNewSlot = 0
          for docId in queryResults:
            if foundNewSlot == 1:
             break
            docPath = self.docId2path[docId]
            sentenceSplittingResults = getSentenceAndOffsetsFromCoref(docId, docPath)
            if len(sentenceSplittingResults) == 0:
              # something went wrong - skip document
              continue
            sentences, offsets = sentenceSplittingResults
            for s,o in zip(sentences, offsets):
              if state in s and country in s: # could be improved with pattern matching
                sentenceList = s.split()
                offsetList = o.split()
                newFillerStart = getsubidx(sentenceList, country.split())
                oldFillerStart = getsubidx(sentenceList, state.split())
                if len(newFillerStart) == 0 or len(oldFillerStart) == 0:
                  # something went wrong
                  continue
                closestNewFiller = -1
                closestOldFiller = -1
                closestDistance = len(sentenceList)
                for i in range(len(newFillerStart)):
                  for j in range(len(oldFillerStart)):
                    dist = abs(newFillerStart[i] - oldFillerStart[j])
                    if dist < closestDistance:
                      closestDistance = dist
                      closestNewFiller = i
                      closestOldFiller = j
                newOffsetStart = offsetList[newFillerStart[closestNewFiller]]
                newOffsetEnd = str(int(newOffsetStart) + len(country) - 1)
                newFillerOffset = docId + ":" + newOffsetStart + "-" + newOffsetEnd  # take the filler most next to the locationFiller?
                proofStart = offsetList[0]
                proofEnd = str(int(offsetList[-1]) - 1)
                if int(proofEnd) - int(proofStart) > 150:
                  minIndex = min(newFillerStart[closestNewFiller], oldFillerStart[closestOldFiller])
                  maxIndex = max(newFillerStart[closestNewFiller], oldFillerStart[closestOldFiller])
                  if int(offsetList[maxIndex]) - int(offsetList[minIndex]) <= 150:
                    while minIndex > 0 and maxIndex < len(offsetList) and int(offsetList[maxIndex]) - int(offsetList[minIndex]) <= 150:
                      minIndex -= 1
                      maxIndex += 1
                    newProofOffset = docId + ":" + offsetList[minIndex + 1] + "-" + str(int(offsetList[maxIndex -1]) - 1)
                    newCandidates.append([newSlot, country, newFillerOffset, newProofOffset, s])
                    foundNewSlot = 1
                    self.logger.info("found inference in sentence: " + s)
                    break
                else:
                  newProofOffset = docId + ":" + proofStart + "-" + proofStart
                  newCandidates.append([newSlot, country, newFillerOffset, newProofOffset, s])
                  foundNewSlot = 1
                  self.logger.info("found inference in sentence: " + s)
                  break

    return newCandidates

  def isEquivalentToFiller(self, name1, name2, type1, type2):
    if name1 == name2:
      return 1
    if type1 == 'PER':
      if len(name1.split()) != len(name2.split()):
        return 0
      if re.search(ur'\s\w(\s)?(\.)?\s', name1, re.UNICODE):
        # found middle initial
        # middle initial with and without dots are equivalent
        name1Tmp = re.sub(ur'\.+', '', name1, re.UNICODE)
        name2Tmp = re.sub(ur'\.+', '', name2, re.UNICODE)
        if name1Tmp == name2Tmp:
          self.logger.debug("equivalent fillers: " + name1 + " -- " + name2)
          return 1
      if re.search(ur'\s\w(\s)?(\.)?\s', name2, re.UNICODE):
        # found middle initial
        # middle initial with and without dots are equivalent
        name1Tmp = re.sub(ur'\.+', '', name1, re.UNICODE)
        name2Tmp = re.sub(ur'\.+', '', name2, re.UNICODE)
        if name1Tmp == name2Tmp:
          self.logger.debug("equivalent fillers: " + name1 + " -- " + name2)
          return 1
      # middle name and middle initials (with and without dots) are equivalent
      splittedName1 = name1.split()
      splittedName2 = name2.split()
      if len(splittedName1) == 3:
        name1Tmp = splittedName1[0] + " " + splittedName1[1][0] + " " + splittedName1[2]
        name2Tmp = splittedName2[0] + " " + splittedName2[1][0] + " " + splittedName2[2]
        if name1Tmp == name2Tmp:
          self.logger.debug("equivalent fillers: " + name1 + " -- " + name2)
          return 1
    elif type1 == 'ORG':
      # if only punctuation is different, the names are equivalent
      name1Tmp = re.sub(ur'\.+', '', name1, re.UNICODE)
      name2Tmp = re.sub(ur'\.+', '', name2, re.UNICODE)
      if name1Tmp == name2Tmp:
        self.logger.debug("equivalent fillers: " + name1 + " -- " + name2)
        return 1
      splittedName1 = name1.split()
      splittedName2 = name2.split()
      sn1 = len(splittedName1) - 1
      while sn1 >= 0:
        if splittedName1[sn1] in ["Corporation", "Corp", "Corp.", "Corps", "Corps.", "Co", "Co.", "corporation", "corp", "corp.", "corps", "corps.", "co", "co."]:
          splittedName1.pop(sn1)
        sn1 -= 1
      sn2 = len(splittedName2) - 1
      while sn2 >= 0:
        if splittedName2[sn2] in ["Corporation", "Corp", "Corp.", "Corps", "Corps.", "Co", "Co.", "corporation", "corp", "corp.", "corps", "co  rps.", "co", "co."]:
          splittedName2.pop(sn2)
        sn2 -= 1
      joinedName1 = " ".join(splittedName1)
      joinedName2 = " ".join(splittedName2)
      if joinedName1 == joinedName2:
        self.logger.debug("equivalent fillers: " + name1 + " -- " + name2)
        return 1
      if "the " + string.lower(joinedName1) == string.lower(joinedName2) or "the " + string.lower(joinedName2) == string.lower(joinedName1):
        self.logger.debug("equivalent fillers: " + name1 + " -- " + name2)
        return 1
    else: # type1 == 'GPE'
      pass
    return 0

  def isEquivalentToName(self, refName, name2):
    if self.curType == 'PER':
      # first name or last name alone are no valid fillers
      tmpRefName = " " + refName + " "
      if " " + name2 + " " in tmpRefName:
        return 1
      # the same which applied for isEqulivalentToFiller also applies or isEquivalentToName
      return self.isEquivalentToFiller(refName, name2, self.curType, self.curType)
    else:
      return self.isEquivalentToFiller(refName, name2, self.curType, self.curType)

  
  def getLocationSlot(self, s, filler, example):
    myFiller = string.lower(filler)
    myFiller2 = myFiller
    if re.search(ur'\.$', myFiller, re.UNICODE):
      myFiller2 = re.sub(ur'\s+\.$', '.',  myFiller, re.UNICODE)
    else:
      myFiller2 = myFiller + "."
    if myFiller in self.states or myFiller2 in self.states:
      if "locations" in s:
        s = re.sub(ur'locations', 'statesorprovinces', s, re.UNICODE)
      else:
        s = re.sub(ur'location', 'stateorprovince', s, re.UNICODE)
    elif myFiller in self.cities or myFiller2 in self.cities:
      if "locations" in s:
        s = re.sub(ur'locations', 'cities', s, re.UNICODE)
      else:
        s = re.sub(ur'location', 'city', s, re.UNICODE)
    elif myFiller in self.countries or myFiller2 in self.countries:
      if "locations" in s:
        s = re.sub(ur'locations', 'countries', s, re.UNICODE)
      else:
        s = re.sub(ur'location', 'country', s, re.UNICODE)
      return s
    else:
      if "province" in example or "district" in example:
        if "locations" in s:
          s = re.sub(ur'locations', 'statesorprovinces', s, re.UNICODE)
        else:
          s = re.sub(ur'location', 'stateorprovince', s, re.UNICODE)
      elif "city" in myFiller or "st " in myFiller or "st." in myFiller or "san " in myFiller or "santa " in myFiller:
        if "locations" in s:
          s = re.sub(ur'locations', 'cities', s, re.UNICODE)
        else:
          s = re.sub(ur'location', 'city', s, re.UNICODE)
      else:
        s = ""
    return s

  def setCandidates(self, globalSlot2fillerCandidatesAndConfidence):
    self.globalSlot2fillerCandidatesAndConfidence = globalSlot2fillerCandidatesAndConfidence

  def applyThreshold(self):
    slotsToDelete = []
    for slot in self.globalSlot2fillerCandidatesAndConfidence:
      newCandidateList = []
      for candidate in self.globalSlot2fillerCandidatesAndConfidence[slot]:
        example = candidate[1]
        if not slot in self.slot2thresholdPF:
          threshold = 0.6
          if slot == "per:other_family": # slot for proximity evaluation: gets small confidence values
            threshold = 0.2
          elif slot == "per:charges":
            threshold = 0.5
        else:
          threshold = float(self.slot2thresholdPF[slot][0])
        confidence = float(candidate[2])
        if confidence > threshold:
          newCandidateList.append(candidate)
      if len(newCandidateList) == 0:
        self.logger.debug("confidence threshold: no candidates left over for slot: " + slot)
        slotsToDelete.append(slot)
      else:
        self.globalSlot2fillerCandidatesAndConfidence[slot] = newCandidateList
    for slot in slotsToDelete:
      del self.globalSlot2fillerCandidatesAndConfidence[slot]

  def postprocessLocation(self):
    # postprocess location:
    newGlobalResults = {}
    newInferenceResults = {}
    for s in self.globalSlot2fillerCandidatesAndConfidence:
      for cands in self.globalSlot2fillerCandidatesAndConfidence[s]:
        if len(cands) == 0:
          continue
        filler = cands[0]
        originalFiller = filler.strip()
        example = cands[1]
        s_new = s
        confidence = cands[2]
        myDocId = cands[5]
        if "location" in s:
          if self.curType == "GPE": # take given slot!
            s_new = self.origSlot
            if s_new == "":
              continue # location is not known
          else: # do postprocessing based on filler
            s_new = self.getLocationSlot(s, originalFiller, example)
            self.logger.debug("LOC postprocessing: " + originalFiller + ": result: " + str(s_new))
            if s_new == "":
              continue # location is not known
        # do inference here
        if self.doInference == 1 and ("countr" in self.searchSlots or "state" in self.searchSlots or "location" in self.searchSlots) and not "countr" in s_new: # else: no inference possible
          inferenceResults = self.getInferenceResults(cands, s_new)
          for ir in inferenceResults:
            inferredSlot = ir[0]
            self.logger.debug("INFERENCE: inferred slot: "  + str(inferredSlot))
            self.logger.debug("search slots: " + str(self.searchSlots))
            if "state" in inferredSlot:
              inferredSlotGeneral = re.sub(r'stateorprovince', 'location', inferredSlot)
              inferredSlotGeneral = re.sub(r'statesorprovinces', 'locations', inferredSlotGeneral)
            elif "countr" in inferredSlot:
              inferredSlotGeneral = re.sub(r'country', 'location', inferredSlot)
              inferredSlotGeneral = re.sub(r'countries', 'locations', inferredSlotGeneral)
            if inferredSlot in self.searchSlots or inferredSlotGeneral in self.searchSlots:
              newSlot, newFiller, newOffsetFiller, newOffsetProof, newProof = ir
              self.logger.info("inferred filler: " + newFiller)
              if not newSlot in newInferenceResults:
                newInferenceResults[newSlot] = []
              newInferenceResults[newSlot].append([newFiller, example, float(confidence), newOffsetFiller, [cands[4], newOffsetProof  ], myDocId, "GPE", "INFERENCE"])
                    
        if "cit" in s_new:
          s_new_general = re.sub(r'city', 'location', s_new)
          s_new_general = re.sub(r'cities', 'locations', s_new_general)
        elif "state" in s_new:
          s_new_general = re.sub(r'stateorprovince', 'location', s_new)
          s_new_general = re.sub(r'statesorprovinces', 'locations', s_new_general)
        elif "countr" in s_new:
          s_new_general = re.sub(r'country', 'location', s_new)
          s_new_general = re.sub(r'countries', 'locations', s_new_general)
        if s_new in self.searchSlots or s_new_general in self.searchSlots:
          if not s_new in newGlobalResults:
            newGlobalResults[s_new] = []
          newGlobalResults[s_new].append(cands)
    self.globalSlot2fillerCandidatesAndConfidence = newGlobalResults
    for newSlot in newInferenceResults:
      if "residence" in newSlot: # list-valued slot
        if not newSlot in self.globalSlot2fillerCandidatesAndConfidence:
          self.globalSlot2fillerCandidatesAndConfidence[newSlot] = []
        for candidate in newInferenceResults[newSlot]:
          self.logger.debug("appending " + str(candidate) + " to " + newSlot)
          self.globalSlot2fillerCandidatesAndConfidence[newSlot].append(candidate)
      else:
        if newSlot in self.globalSlot2fillerCandidatesAndConfidence and len(self.globalSlot2fillerCandidatesAndConfidence[newSlot]) > 0:
          self.logger.debug("found slot " + newSlot + " with inference but ignored it since slot already exists")
        else:
          if not newSlot in self.globalSlot2fillerCandidatesAndConfidence:
            self.globalSlot2fillerCandidatesAndConfidence[newSlot] = []
          for candidate in newInferenceResults[newSlot]:
            self.logger.debug("appending " + str(candidate) + " to " + newSlot)
            self.globalSlot2fillerCandidatesAndConfidence[newSlot].append(candidate)

  def setCurType(self, curType):
    self.curType = curType

  def setCurName(self, curName):
    self.curName = curName

  def setListOfAlias(self, listOfAlias):
    self.listOfAlias = copy.deepcopy(listOfAlias)

  def setCurQueryId(self, curQueryId):
    self.curQueryId = curQueryId

  def alternateNamesFillers(self, aliasOffsets, curNameOffsets, spellingVariationsOffsets, doc2wordsAndOffsets = {}):
    # store aliases as fillers for alternate names if they appear in the documents
    slot = string.lower(self.curType) + ":alternate_names"

    foundAlias = []

    listOfAliasLengths = [len(i.split()) for i in self.listOfAlias]
    listOfAlias_ind_sorted = sorted(range(len(listOfAliasLengths)), key=listOfAliasLengths.__getitem__, reverse = True)
    lengthLongestAlias = len(self.listOfAlias[listOfAlias_ind_sorted[0]].split())
    for i in listOfAlias_ind_sorted:
      self.logger.info("processing " + self.listOfAlias[i])
      if string.lower(self.listOfAlias[i]) == string.lower(self.curName):
        self.logger.debug("alias " + str(i) + " is curName")
        continue
      if i in aliasOffsets:
        aliasOccurrences = aliasOffsets[i] # here we have one entry per document which consists of a list with each element containing the docId and the occurrence of the alias
        newAliasOccurrences = []
        for aliasOccurrencesInCertainDoc in aliasOccurrences:
          newAliasOccInDoc = []
          for j in range(0, len(aliasOccurrencesInCertainDoc)):
            doc = aliasOccurrencesInCertainDoc[j][0]
            offsetList = aliasOccurrencesInCertainDoc[j][1]
            newOffsetList = []
            for offs in offsetList:
              offsL = offs.split(',')
              if doc in doc2wordsAndOffsets:
                wordsAndOffsets = doc2wordsAndOffsets[doc]
                indexStart = -1
                indexEnd = -1
                aliasOcc = ""
                for ind, wo in enumerate(wordsAndOffsets):
                  w, o = wo
                  if int(o) == int(offsL[0]):
                    indexStart = ind
                  if indexStart > -1:
                    aliasOcc += w + " "
                  if int(o) == int(offsL[-1]):
                    indexEnd = ind + 1
                    break
                  elif indexStart > -1 and int(o) > int(offsL[-1]):
                    indexEnd = ind
                    break
                if indexStart > -1:
                  aliasOccList = []
                  aliasOcc = aliasOcc.strip()
                  aliasOcc2 = re.sub(r'\W$', '', aliasOcc)
                  aliasOccList.append(string.lower(aliasOcc))
                  aliasOccList.append(string.lower(aliasOcc2))
                  aliasOcc2 = re.sub(r'^\W', '', aliasOcc)
                  aliasOccList.append(string.lower(aliasOcc2))
                  aliasOcc3 = re.sub(r'\W', ' TOSPLIT ', aliasOcc)
                  aliasOcc3 = re.sub(r' +', ' ', aliasOcc3)
                  aliasOccParts = aliasOcc3.split(' TOSPLIT ')
                  aliasOccPartsLc = [string.lower(p) for p in aliasOccParts]
                  aliasOccList.extend(aliasOccPartsLc)
                  aliasFromList = string.lower(self.listOfAlias[i])
                  if not aliasFromList in aliasOccList:
                    self.logger.debug("alias in document " + str(aliasOccList) + " does not equal alias from list: " + aliasFromList)
                  else:
                    newOffsetList.append(",".join(offsL))
            if len(newOffsetList) > 0:
              newAliasOccInDoc.append([doc, newOffsetList])
          if len(newAliasOccInDoc) > 0:
            newAliasOccurrences.append(newAliasOccInDoc)
        aliasOffsets[i] = newAliasOccurrences
      if string.lower(" ".join(self.listOfAlias[i].split('-'))) == string.lower(self.curName):
        continue
      if self.listOfAlias[i] in self.curName:
        # check whether alias or curName occurs in document:
        if i in aliasOffsets: # i is the index of an alias
          aliasOccurrences = aliasOffsets[i] # here we have one entry per document which consists of a list with each element containing the docId and the occurrence of the alias
          appended = 0
          for aliasOccurrencesInCertainDoc in aliasOccurrences:
           for j in range(0, len(aliasOccurrencesInCertainDoc)):
            if appended == 1:
              break
            doc = aliasOccurrencesInCertainDoc[j][0]
            if not doc in curNameOffsets:
              # alias occurs in document: append to results
              offsetList = aliasOccurrencesInCertainDoc[j][1]
              for ol in offsetList:
                offsets = ol.split(',')
                start = int(offsets[0])
                end = int(offsets[-1])
                if [doc, start] in foundAlias:
                  continue
                isLongerAlias = 0
                for k in listOfAlias_ind_sorted:
                  if k == i:
                    break
                  if isLongerAlias == 1:
                    break
                  # test whether it is an appearance of a longer alias
                  testAlias = self.listOfAlias[k]
                  if not k in aliasOffsets:
                    continue
                  testAliasOcc = aliasOffsets[k]
                  for o in testAliasOcc:
                    if len(o[1]) == 0:
                      continue
                    testAliasStart = int(o[1][0].split(',')[0])
                    if start == testAliasStart:
                      isLongerAlias = 1
                      break
                if isLongerAlias == 1:
                  continue
                if not slot in self.globalSlot2fillerCandidatesAndConfidence:
                  self.globalSlot2fillerCandidatesAndConfidence[slot] = []
                self.globalSlot2fillerCandidatesAndConfidence[slot].append([self.listOfAlias[i], self.listOfAlias[i], 1.0, str(start) + "," + str(end), str(start) + " " + str(end), doc, self.curType])
                foundAlias.append([doc, start])
                appended = 1
                break
            else:
              # check whether occurrence is actually an occurrence of curName:
              offsetsCurName = curNameOffsets[doc]
              offsetList = aliasOccurrencesInCertainDoc[j][1]
              appended = 0
              for ol in offsetList:
                if appended == 1:
                  break
                offsets = ol.split(',')
                startAlias = int(offsets[0])
                foundCurName = 0
                for ocn in offsetsCurName:
                  startCurName = ocn.split(',')[0]
                  if startCurName == startAlias:
                    foundCurName = 1
                    break
                if foundCurName == 0:
                    if [doc, startAlias] in foundAlias:
                      continue
                    isLongerAlias = 0
                    for k in listOfAlias_ind_sorted:
                      if k == i:
                        break
                      if isLongerAlias == 1:
                        break
                      # test whether it is an appearance of a longer alias
                      testAlias = self.listOfAlias[k]
                      if not k in aliasOffsets:
                        continue
                      aliasOffsetsOfAliasK = aliasOffsets[k]
                      for testAliasOcc in aliasOffsetsOfAliasK:
                        for o in testAliasOcc:
                          if len(o[1]) == 0:
                            continue
                          testAliasStart = int(o[1][0].split(',')[0])
                          if startAlias == testAliasStart:
                            isLongerAlias = 1
                            break
                    if isLongerAlias == 1:
                      continue
                    # alias occurs at this position: append to results
                    endAlias = int(offsets[-1])
                    if not slot in self.globalSlot2fillerCandidatesAndConfidence:
                      self.globalSlot2fillerCandidatesAndConfidence[slot] = []
                    self.globalSlot2fillerCandidatesAndConfidence[slot].append([self.listOfAlias[i], self.listOfAlias[i], 1.0, str(startAlias)+","+str(endAlias), str(startAlias)+" "+str(endAlias), doc, self.curType])
                    foundAlias.append([doc, startAlias])
                    appended = 1
                    break
      elif i in aliasOffsets:
       for aliasOccurrences in aliasOffsets[i]:
        doc = aliasOccurrences[0][0]
        offsetList = aliasOccurrences[0][1]
        if len(offsetList) == 0:
          continue
        ol = offsetList[0]
        offsets = ol.split(',')
        start = int(offsets[0])
        end = int(offsets[-1])
        if [doc, start] in foundAlias:
          continue
        isLongerAlias = 0
        for k in listOfAlias_ind_sorted:
          if k == i:
            break
          if isLongerAlias == 1:
            break
          # test whether it is an appearance of a longer alias
          testAlias = self.listOfAlias[k]
          if not k in aliasOffsets:
            continue
          for testAliasOcc in aliasOffsets[k]:
            for o in testAliasOcc:
              if len(o[1]) == 0:
                continue
              testAliasStart = int(o[1][0].split(',')[0])
              if start == testAliasStart:
                isLongerAlias = 1
                break
        if isLongerAlias == 1:
          continue
        if not slot in self.globalSlot2fillerCandidatesAndConfidence:
          self.globalSlot2fillerCandidatesAndConfidence[slot] = []
        self.globalSlot2fillerCandidatesAndConfidence[slot].append([self.listOfAlias[i], self.listOfAlias[i], 1.0, str(start) + "," + str(end), str(start) + " " + str(end), doc, self.curType])
        foundAlias.append([doc, start])

  def postprocessDates(self, slots_orig, doc2offsets2normDate):
    for s in slots_orig:
      if "date" in s:
        if not s in self.globalSlot2fillerCandidatesAndConfidence:
          continue
        candList = self.globalSlot2fillerCandidatesAndConfidence[s]
        newCandList = []
        for myIndex, myCands in enumerate(candList):
          originalFiller = myCands[0]
          example = myCands[1]
          confidenceString = str(myCands[2])
          curOffsetFillerList = myCands[3].strip().split(",")
          fillerStart = curOffsetFillerList[0]
          fillerEnd = int(fillerStart) + len(originalFiller) - 1
          curCandidateOffset = myCands[4]
          doc = myCands[5]
          if not doc in doc2offsets2normDate:
            continue
          offsets2normDate = doc2offsets2normDate[doc]
          self.logger.debug("original date: " + originalFiller)
          foundNormDate = 0
          for fInd in range(int(fillerStart), fillerEnd + 1):
            if str(fInd) in offsets2normDate:
              originalFillerTmp = offsets2normDate[str(fInd)]
              if re.search(ur'^\w{4}$', originalFillerTmp, re.UNICODE):
                # append XX for month and day
                originalFillerTmp += "-XX-XX"
              elif re.search(ur'^\w{4}\-\w{2}$', originalFillerTmp, re.UNICODE):
                # append XX for day
                originalFillerTmp += "-XX"
              if re.search(ur'^\w{4}\-\w{2}\-\w{2}$', originalFillerTmp, re.UNICODE):
                foundNormDate = 1
                originalFiller = originalFillerTmp
                break
          if foundNormDate == 0: # try to normalize by myself
            originalFiller = normalizeDate(originalFiller)
          self.logger.debug("normalized date: " + originalFiller)
          if "XXXX-XX-XX" != originalFiller:
            year = originalFiller.split('-')[0]
            if year[0] == "X" or int(year[0]) > 2:
              # don't store filler: makes no sense
              pass
            else: # store normalized date:
              curCandidateOffsetList = curCandidateOffset.split()
              proofStart = curCandidateOffsetList[0]
              proofEnd = int(curCandidateOffsetList[-1]) + len(example.split()[-1]) - 1
              docAndOffsetProof = doc + ":" + str(proofStart) + "-" + str(proofEnd)
              docAndOffsetFiller = doc + ":" + str(fillerStart) + "-" + str(fillerEnd)
              newCandList.append(copy.deepcopy(myCands))
              newCandList[-1][0] = originalFiller
              newCandList[-1][3] = str(fillerStart) + "," + str(fillerEnd)
            
        self.globalSlot2fillerCandidatesAndConfidence[s] = newCandList


  def rankingPerSlot(self, queryResult):
    listSlots = {'per:alternate_names', 'per:children', 'per:cities_of_residence', 'per:countries_of_residence' , 'per:employee_or_member_of', 'per:origin', 'per:other_family', 'per:parents', 'per:schools_attended', 'per:siblings', 'per:spouse', 'per:statesorprovinces_of_residence', 'per:charges', 'per:title', 'org:alternate_names', 'org:founded_by', 'org:member_of', 'org:members', 'org:parents', 'org:political_religious_affiliation', 'org:shareholders', 'org:subsidiaries', 'org:top_members_employees', 'org:employees_or_members', 'gpe:employees_or_members', 'org:students', 'gpe:births_in_city', 'gpe:births_in_stateorprovince', 'gpe:births_in_country', 'gpe:residents_of_city', 'gpe:residents_of_stateorprovince', 'gpe:residents_of_country', 'gpe:deaths_in_city', 'gpe:deaths_in_stateorprovince', 'gpe:deaths_in_country', 'per:holds_shares_in', 'org:holds_shares_in', 'gpe:holds_shares_in', 'per:organizations_founded', 'org:organizations_founded', 'gpe:organizations_founded', 'per:top_member_employee_of', 'gpe:member_of', 'gpe:subsidiaries', 'gpe:headquarters_in_city', 'gpe:headquarters_in_stateorprovince', 'gpe:headquarters_in_country'}
    singleSlots = {'per:city_of_birth', 'per:stateorprovince_of_birth', 'per:country_of_birth', 'per:country_of_death', 'per:stateorprovince_of_death', 'per:city_of_death', 'per:age', 'per:date_of_birth', 'per:date_of_death', 'per:cause_of_death', 'per:religion', 'org:city_of_headquarters', 'org:country_of_headquarters', 'org:stateorprovince_of_headquarters', 'org:date_dissolved', 'org:date_founded', 'org:number_of_employees_members', 'org:website'}
    for s in self.globalSlot2fillerCandidatesAndConfidence:
      candList = self.globalSlot2fillerCandidatesAndConfidence[s]
      candListSorted = sorted(candList, key=lambda x:x[2], reverse=True)
      count = 0
      highestConf = -1
      bestDocIndex = 10000
      resultCandList = []

      for index, cands in enumerate(candListSorted):
        candidate = cands[1]
        curFiller = cands[0]
        curConfidence = cands[2]
        curFillerOffset = cands[3]
        curCandidateOffset = cands[4]
        docId = cands[5]
        fillerType = cands[6]
        if len(curFiller) == 0:
          continue
        if s in listSlots:
          # remove redundancies from list-valued slots
          curFillerLc = string.lower(curFiller)
          fillerAlias = self.aliasModule.getAlias(curFiller)
          fillerAlias.extend(self.aliasModule.getAlias(curFillerLc))
          fillerAliasLc = [string.lower(fa) for fa in fillerAlias]
          if not curFillerLc in fillerAliasLc:
            fillerAliasLc.append(curFillerLc)
          isRedundant = 0 # is curFiller redundant to an already saved filler?
          overwrite = -1
          curCandidates = []
          for indexCandidate, resultingCandidate in enumerate(resultCandList):
            if isRedundant == 1 or overwrite > -1:
              break
            for fa in fillerAliasLc:
              if self.isEquivalentToFiller(fa, string.lower(resultingCandidate[0]), resultingCandidate[6], resultingCandidate[6]) == 1:
                if resultingCandidate[2] > curConfidence:
                  isRedundant = 1
                  break
                elif resultingCandidate[2] == curConfidence:
                  # break tie by keeping the results with the document with more precedence in IR
                  if queryResult.index(docId) < queryResult.index(resultingCandidate[5]):
                    overwrite = indexCandidate # curFiller should overwrite old filler
                    break
                  else:
                    isRedundant = 1
              else:
                distance = levenshtein(fa, string.lower(resultingCandidate[0])) # don't allow different spelling variations of fillers as different fillers => is that correct?
                if distance < len(curFiller) / 5 + 1: # just some arbitrary heuristic
                  if resultingCandidate[2] > curConfidence:
                    isRedundant = 1
                    break
                  elif resultingCandidate[2] == curConfidence:
                    # break tie by keeping the results with the document with more precedence in IR
                    if queryResult.index(docId) < queryResult.index(resultingCandidate[5]):
                      overwrite = indexCandidate # curFiller should overwrite old filler
                      break
                    else:
                      isRedundant = 1
                  else:
                    overwrite = 1
                    break
          if overwrite > -1:
            resultCandList[overwrite] = [curFiller, candidate, curConfidence, curFillerOffset, curCandidateOffset, docId, fillerType]
          elif isRedundant == 0:
            if self.isEquivalentToName(self.curName, curFiller) == 0:
              if len(cands) > 7:
                resultCandList.append([curFiller, candidate, curConfidence, curFillerOffset, curCandidateOffset, docId, fillerType, cands[7]])
              else:
                resultCandList.append([curFiller, candidate, curConfidence, curFillerOffset, curCandidateOffset, docId, fillerType])
              count += 1
        else:
          if not s in singleSlots:
            self.logger.error("could not find slot " + s + " in single valued slots and list slots. Assuming single valued slot")
          # extract name with highest confidence for single-valued slots
          if highestConf == -1: # first result
            highestConf = curConfidence
            bestDocIndex = queryResult.index(docId)
            if len(cands) > 7:
              resultCandList.append([curFiller, candidate, curConfidence, curFillerOffset, curCandidateOffset, docId, fillerType, cands[7]])
            else:
              resultCandList.append([curFiller, candidate, curConfidence, curFillerOffset, curCandidateOffset, docId, fillerType])
          else:
            if curConfidence == highestConf: # break tie according to document precedence in IR
              curDocIndex = queryResult.index(docId)
              if curDocIndex < bestDocIndex:
                bestDocIndex = curDocIndex
                if len(cands) > 7:
                  resultCandList[0] = [curFiller, candidate, curConfidence, curFillerOffset, curCandidateOffset, docId, fillerType, cands[7]]
                else:
                  resultCandList[0] = [curFiller, candidate, curConfidence, curFillerOffset, curCandidateOffset, docId, fillerType]
            else:
              break

      self.globalSlot2fillerCandidatesAndConfidence[s] = resultCandList

  def postprocessForOutput(self, slots_orig, doc2wordsAndOffsets, doc2additionalCorefFillers = {}):

    for s in slots_orig:
      if self.curType == "PER" and not "per" in s:
        continue
      if self.curType == "ORG" and not "org" in s:
        continue
      if self.curType == "GPE" and not "gpe" in s:
        continue
      if not s in self.globalSlot2fillerCandidatesAndConfidence:
        continue
      bestDocIndex = 10000
      fillerList = []
      proofList = []
      confidenceList = []
      offsetFillerList = []
      offsetProofList = []
      docIdList = []
      candList = self.globalSlot2fillerCandidatesAndConfidence[s]

      # prepare results for printing
      gotOutputForSlot = 0
      for cands in candList:
        confidenceString = str(cands[2])
        myFiller = cands[0]
        myDocId = cands[5]
        fillerTypeOrig = cands[6]
        fillerType = self.cleanFillerType(s, fillerTypeOrig)
        if fillerTypeOrig != fillerType:
          self.logger.debug("cleaned filler type for " + s + " to " + fillerType)

        myOffsetFillerList = [cands[3]]
        if len(cands) > 7:
          myOffsetProofList = cands[4][0].split()
        else:
          myOffsetProofList = cands[4].split()

        if len(cands) > 7:
          docAndOffsetFiller = cands[3]
          fillerStart = docAndOffsetFiller.split(':')[1].split('-')[0]
          fillerEnd = docAndOffsetFiller.split(':')[1].split('-')[1]
        else:
          if " " in cands[3]: 
            myOffsetFillerList = cands[3].split(" ")
          elif "," in cands[3]:
            myOffsetFillerList = cands[3].split(",")
          myFillerList = myFiller.split()
          # adjust filler end offset with length of filler:
          fillerStart = myOffsetFillerList[0]
          if "date_" in s:
            fillerEnd = int(myOffsetFillerList[-1]) # already includes length because of normalization
          else:
            fillerEnd = int(fillerStart) + len(myFiller) - 1
          docAndOffsetFiller = myDocId + ":" + fillerStart + "-" + str(fillerEnd)
          if fillerEnd - int(fillerStart) + 1 > 150:
            self.logger.debug("filler " + myFiller + " is longer than 150 characters")
            continue # filler is too long!

        myProof = cands[1]
        proofStart = myOffsetProofList[0]
        proofEnd = str(int(myOffsetProofList[-1]) + len(myProof.split()[-1]) - 1)
        if int(proofEnd) - int(proofStart) + 1 > 150:
          # too long: split it into two parts
          newProofStart = max(int(fillerEnd) - 75, int(proofStart))
          if int(fillerEnd) - int(fillerStart) + 1 > 75:
            newProofStart = max(int(fillerEnd) - int(fillerStart) + 1, int(proofStart))
          self.logger.debug("fillerStart " + str(fillerStart))
          self.logger.debug("proofStart " + str(proofStart))
          self.logger.debug("newProofStart " + str(newProofStart))
          self.logger.debug("fillerEnd " + str(fillerEnd))
          self.logger.debug("proofEnd " + str(proofEnd))
          newProofEnd = min(int(newProofStart) + 150, int(proofEnd))
          self.logger.debug("newProofEnd " + str(newProofEnd))
          if newProofEnd - newProofStart + 1 < 145:
            newProofStart = newProofEnd - 145
          self.logger.debug("too long proof: old start-end: " + proofStart + "-" + proofEnd + ", what would be better: " + str(newProofStart) + "-" + str(newProofEnd))

          # find nearest offsets in myOffsetProofList:
          newStart = -1
          newEnd = -1
          for index, offP in enumerate(myOffsetProofList):
            if newStart == -1 and int(offP) >= newProofStart:
              proofStart = offP
              newStart = index
            if newEnd == -1 and int(offP) >= newProofEnd:
              proofEnd = offP
              newEnd = index
              break
          proofStart = myOffsetProofList[newStart]
          proofEnd = str(int(myOffsetProofList[newEnd]) - 1) # -1 because offset span includes end offset
          self.logger.info("adapted too long proof to: " + myDocId + ":" + proofStart + "-" + proofEnd + ": " + " ".join(cands[1].split()[int(newStart) : int(newEnd) + 1]))
        docAndOffsetProof = myDocId + ":" + proofStart + "-" + proofEnd
        if len(cands) > 7:
          docAndOffsetProof += "," + cands[4][1]

        originalFiller = myFiller

        originalFiller = re.sub(ur'\W+$', '', originalFiller, re.UNICODE)
        originalFiller = re.sub(ur'^\W+', '', originalFiller, re.UNICODE)

        fillerStart = int(fillerStart)
        fillerEnd = int(fillerEnd)

        if "alternate_names" in s and originalFiller == self.curName:
          continue

        if myDocId in doc2additionalCorefFillers:
          additionalCorefFillers = doc2additionalCorefFillers[myDocId]
          pronounOffset = cands[3]
          if pronounOffset in additionalCorefFillers:
            self.logger.debug("filler is a corefered mention: expanding filler and proof offsets accordingly")
            personOffsetList = additionalCorefFillers[pronounOffset][1].split(',')
            personStart = personOffsetList[0]
            personEnd = str(int(personOffsetList[-1]) + len(myFiller) - 1)
            docAndOffsetProof = docAndOffsetProof + "," + myDocId + ":" + personStart + "-" + personEnd
            docAndOffsetFiller = myDocId + ":" + personStart + "-" + personEnd

        resultString = self.curQueryId + "\t" + s + "\t" + "CIS" + "\t" + docAndOffsetProof + "\t" + originalFiller + "\t" + fillerType + "\t" + docAndOffsetFiller + "\t" + confidenceString
        self.myResults.append(resultString)


  def reduceFPs(self):
    # (1)
    slot2resultList = {}
    slot2fillerStringList = {}
    for line in self.myResults:
      parts = line.split('\t')
      queryId, slot, runId, proof, filler, fillerType, fillerOffsets, confidence = parts
      if slot in slot2fillerStringList:
        fillerStringList = slot2fillerStringList[slot]
      else:
        fillerStringList = []
      isRedundant = 0
      overwrite = -1
      for index,fi in enumerate(fillerStringList):
        if "alternate_names" in slot:
          break
        if isRedundant == 1 or overwrite > -1:
          break
        if re.search(r'\W' + re.escape(filler) + '\W', " " + fi + " ") or re.search(r'\W' + re.escape(fi) + '\W', " " + filler + " "):
          # favor filler with higher confidence or longer filler
          confidenceFi = float(slot2resultList[slot][index][7])
          confidenceFiller = float(confidence)
          if confidenceFiller > confidenceFi:
            overwrite = index
            break
          elif confidenceFiller < confidenceFi:
            isRedundant = 1
            break
          else:
            if len(filler) > len(fi):
              overwrite = index
              break
            else:
              isRedundant = 1
              break
        namePartsFiller = filler.split()
        namePartsFi = fi.split()
        for np1 in namePartsFiller:
          if isRedundant == 1 or overwrite > -1:
            break
          if np1 in self.name2nicknames:
            nicknames = self.name2nicknames[np1]
            for nick1 in nicknames:
              if nick1 in namePartsFi:
                test = re.sub(r' '+re.escape(nick1)+' ', ' ' + np1 + ' ', ' ' + fi + ' ').strip()
                if filler in test or test in filler:
                  # favor filler with higher confidence or longer filler
                  confidenceFi = float(slot2resultList[slot][index][7])
                  confidenceFiller = float(confidence)
                  if confidenceFiller > confidenceFi:
                    overwrite = index
                    break
                  elif confidenceFiller < confidenceFi:
                    isRedundant = 1
                    break
                  else:
                    if len(np1) > len(nick1):
                      overwrite = index
                      break
                    else:
                      isRedundant = 1
                      break
      if isRedundant == 1:
        continue
      if overwrite > -1:
        slot2resultList[slot][overwrite] = parts
        slot2fillerStringList[slot][overwrite] = filler
        continue
      if not slot in slot2resultList:
        slot2resultList[slot] = []
      slot2resultList[slot].append(parts)
      if not slot in slot2fillerStringList:
        slot2fillerStringList[slot] = []
      slot2fillerStringList[slot].append(filler)

    # (2)
    # threshold for per:title, per:religion: 0.6; per:other_family: 0.2
    if "per:title" in slot2resultList:
      tmpList = slot2resultList["per:title"]
      newList = []
      for t in tmpList:
        if float(t[7]) >= 0.6:
          newList.append(t)
      slot2resultList["per:title"] = newList
    if "per:religion" in slot2resultList:
      tmpList = slot2resultList["per:religion"]
      newList = []
      for t in tmpList:
        if float(t[7]) >= 0.6:
          newList.append(t)
      slot2resultList["per:religion"] = newList
    if "per:other_family" in slot2resultList:
      tmpList = slot2resultList["per:other_family"]
      newList = []
      for t in tmpList:
        if float(t[7]) >= 0.2:
          newList.append(t)
      slot2resultList["per:other_family"] = newList
    if "per:charges" in slot2resultList:
      tmpList = slot2resultList["per:charges"]
      newList = []
      for t in tmpList:
        if float(t[7]) >= 0.5:
          newList.append(t)
      slot2resultList["per:charges"] = newList
    if "org:website" in slot2resultList:
      tmpList = slot2resultList["org:website"]
      newList = []
      for t in tmpList:
        if float(t[7]) >= 0.6:
          newList.append(t)
      slot2resultList["org:website"] = newList
    if "org:political_religious_affiliation" in slot2resultList:
      tmpList = slot2resultList["org:political_religious_affiliation"]
      newList = []
      for t in tmpList:
        if float(t[7]) >= 0.6:
          newList.append(t)
      slot2resultList["org:political_religious_affiliation"] = newList

    # (3)
    # maximum number of fillers for some slots:
    slot2maxNumber = {"per:spouse": 5, "per:parents": 3, "per:origin": 3, "per:children": 8, "per:siblings": 8, "org:founded_by": 7, "org:subsidiaries": 10, "org:parents": 10, "org:members": 10, "org:top_members_employees": 10}
    for s in slot2maxNumber:
      if s in slot2resultList:
        number = slot2maxNumber[s]
        tmpList = slot2resultList[s]
        if len(tmpList) > number:
          tmpListSorted = sorted(tmpList, key = lambda x:float(x[7]), reverse = True)
          slot2resultList[s] = tmpListSorted[:number]

    # done
    resultList = []
    for key in slot2resultList.keys():
      for tmpList in slot2resultList[key]:
        resultList.append("\t".join(tmpList))
    self.myResults = resultList


  def resetResults(self):
    self.myResults = []

  def setOrigSlot(self, slot):
    self.origSlot = slot
    self.searchSlots = self.slots # search all slots (for backward compatibility to evaluations before 2015!)
    if self.origSlot != "":
      self.searchSlots = [self.origSlot]

  def __init__(self, slots, countries, states, cities, slot2thresholdPF, city2stateAndCountry, state2country, docId2path, name2nicknames, doInference, terrierDir, aliasModule, loggerMain):
    self.slots = slots
    self.countries = countries
    self.states = states
    self.cities = cities
    self.globalSlot2fillerCandidatesAndConfidence = {}
    self.curType = ""
    self.curName = ""
    self.myResults = [] # will be output later
    self.slot2thresholdPF = slot2thresholdPF
    self.city2stateAndCountry = city2stateAndCountry
    self.state2country = state2country
    self.docId2path = docId2path
    self.doInference = doInference
    self.terrierDir = terrierDir
    self.origSlot = ""
    self.searchSlots = self.slots
    self.name2nicknames = name2nicknames
    self.aliasModule = aliasModule

    self.logger = loggerMain.getChild(__name__)
