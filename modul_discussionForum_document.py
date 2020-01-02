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
import getOffsets
import readNerAndCoref
from utilities import compareNamesImprovedLc
import re

class DiscussionForumDocument:

  def deleteOffsetListFromNameOffsets(self, offsetList):
    for offset in offsetList:
      # delete from nameOffsets
      toDelete = []
      for index, extractedOffsets in enumerate(self.nameOffsets):
        if extractedOffsets == offset or ',' + offset + ',' in ',' + extractedOffsets + ',':
          self.logger.debug("deleting " + extractedOffsets + " from self.nameOffsets")
          toDelete.append(index)
      for td in sorted(toDelete, reverse=True):
        self.nameOffsets.pop(td)
      # delete from origNameOffsets:
      toDelete = []
      for index, extractedOffsetsTuple in enumerate(self.origNameOffsets):
         extractedOffsets = extractedOffsetsTuple[1]
         if extractedOffsets == offset or ',' + offset + ',' in ',' + extractedOffsets + ',':
           toDelete.append(index)
           self.logger.debug("deleting " + extractedOffsets + " from self.origNameOffset")
      for td in sorted(toDelete, reverse=True):
        self.origNameOffsets.pop(td)
      # delete from offsets2normDate:
      if offset in self.offsets2normDate:
        self.logger.debug("deleting " + offset + " from self.offsets2normDate")
        del self.offsets2normDate[offset]
    # delete from additionalNameOffsets
    newAdditionalNameOffsets = []
    for index, extractedOffsetsList in enumerate(self.additionalNameOffsets):
      newInnerList = []
      for extractedOffset in extractedOffsetsList:
        if not extractedOffset in offsetList:
          newInnerList.append(extractedOffset)
        else:
          self.logger.debug("deleting " + extractedOffset + " from self.additionalNameOffsets")
      newAdditionalNameOffsets.append(newInnerList)
    self.additionalNameOffsets = newAdditionalNameOffsets

  def ignoreQuoteSections(self):
    newSentences = []
    newNerInSentences = []
    newOffsets = []
    newPosInSentences = []
    newLemmasInSentences = []
    inQuote = 0
    for sentInd, sentence in enumerate(self.sentences):
      sentenceList = sentence.split()
      if inQuote == 1:
        if "</quote>" in sentenceList:
          endQuote = sentenceList.index("</quote>")
          if endQuote != len(sentenceList) - 1:
            # store non-quotation part
            newSentence = " ".join(sentenceList[endQuote + 1:])
            newSentences.append(newSentence)
            newNers = " ".join(self.nerInSentences[sentInd].split()[endQuote + 1:])
            newNerInSentences.append(newNers)
            newOffs = " ".join(self.offsets[sentInd].split()[endQuote + 1:])
            newOffsets.append(newOffs)
            newPos = " ".join(self.posInSentences[sentInd].split()[endQuote + 1:])
            newPosInSentences.append(newPos)
            newLemmas = " ".join(self.lemmasInSentence[sentInd].split()[endQuote + 1:])
            newLemmasInSentences.append(newLemmas)
          # delete quotation part from name/date offsets
          self.deleteOffsetListFromNameOffsets(self.offsets[sentInd].split()[:endQuote + 1])
          inQuote = 0
      elif "<quote>" in sentenceList:
        beginQuote = sentenceList.index("<quote>")
        if beginQuote != 0:
          # store non-quotation part
          newSentence = " ".join(sentenceList[:beginQuote])
          newSentences.append(newSentence)
          newNers = " ".join(self.nerInSentences[sentInd].split()[:beginQuote])
          newNerInSentences.append(newNers)
          newOffs = " ".join(self.offsets[sentInd].split()[:beginQuote])
          newOffsets.append(newOffs)
          newPos = " ".join(self.posInSentences[sentInd].split()[:beginQuote])
          newPosInSentences.append(newPos)
          newLemmas = " ".join(self.lemmasInSentence[sentInd].split()[:beginQuote])
          newLemmasInSentences.append(newLemmas)
        # delete quotation part from name/date offsets
        self.deleteOffsetListFromNameOffsets(self.offsets[sentInd].split()[beginQuote:])
        if not "</quote>" in sentenceList:
          inQuote = 1
        else:
          endQuote = sentenceList.index("</quote>")
      else:
        beginQuote = -1
        for wInd, w in enumerate(sentenceList):
          if "<quote" in w:
            beginQuote = wInd
            break
        if beginQuote != -1:
          if beginQuote != 0:
            # store non-quotation part
            newSentence = " ".join(sentenceList[:beginQuote])
            newSentences.append(newSentence)
            newNers = " ".join(self.nerInSentences[sentInd].split()[:beginQuote])
            newNerInSentences.append(newNers)
            newOffs = " ".join(self.offsets[sentInd].split()[:beginQuote])
            newOffsets.append(newOffs)
            newPos = " ".join(self.posInSentences[sentInd].split()[:beginQuote])
            newPosInSentences.append(newPos)
            newLemmas = " ".join(self.lemmasInSentence[sentInd].split()[:beginQuote])
            newLemmasInSentences.append(newLemmas)
          # delete quotation part from name/date offsets
          self.deleteOffsetListFromNameOffsets(self.offsets[sentInd].split()[beginQuote:])
          if not "</quote>" in sentenceList:
            inQuote = 1
          else:
            endQuote = sentenceList.index("</quote>")
        else:
          # store non-quotation part
          newSentences.append(sentence)
          newNerInSentences.append(self.nerInSentences[sentInd])
          newOffsets.append(self.offsets[sentInd])
          newPosInSentences.append(self.posInSentences[sentInd])
          newLemmasInSentences.append(self.lemmasInSentence[sentInd])
          # nothing to delete here: whole sentence is non-quotation
    self.sentences = newSentences
    self.nerInSentences = newNerInSentences
    self.offsets = newOffsets
    self.posInSentences = newPosInSentences
    self.lemmasInSentence = newLemmasInSentences

  def normalizeCasing(self):
    # normalize stuff like "sErVice"
    normalizedSentences = []
    for sentence in self.sentences:
      normalizedSentence = ""
      for w in sentence.split():
        if re.search(ur'[a-z]+[A-Z]', w, re.UNICODE):
          w = w.lower()
        normalizedSentence += w + " "
      normalizedSentences.append(normalizedSentence.strip())
    self.sentences = normalizedSentences

  def preprocessDoc(self, listOfTokenizedAlias, isPer, isOrg, isLoc, isPersonFiller = 1):
    # do coreference resolution and named entity tagging
    wordTags = readNerAndCoref.nerAndCoref(listOfTokenizedAlias, self.docId, self.docPath, isPer, isOrg, isLoc, self.doCoref, isBolt = 1, isPersonFiller = isPersonFiller)
    
    if len(wordTags) == 0:
      self.logger.error("error in ner- and coref- alignment")
      return -1

    # results:
    self.sentences = wordTags[0]
    self.nerInSentences = wordTags[1]
    self.offsets = wordTags[2]
    self.nameOffsets = list(set(wordTags[3]))
    self.origNameOffsets = wordTags[4]
    self.posInSentences = wordTags[5]
    self.lemmasInSentence = wordTags[6]
    self.offsets2normDate = wordTags[7]
    self.additionalNameOffsets = wordTags[8]
    self.spellingVariationsOffsets = wordTags[9]
    self.additionalCorefFillers = wordTags[10]

    # adjust offsets: CoreNLP does not count spaces at the beginning of lines, but TAC does
    newOffsets, newOffset2NormDate, newAdditionalNameOffsets, newNameOffsets, newOrigNameOffsets, newAdditionalCorefFillers = getOffsets.correctOffsets(self.docId, self.docPath, self.sentences, self.offsets, self.offsets2normDate, self.additionalNameOffsets, self.nameOffsets, self.origNameOffsets, self.additionalCorefFillers)

    if len(newOffsets) == 0: # should not happen
      self.logger.error("length offsets per line old: " + str(len(self.offsets)) + ", and new: " + str(len(newOffsets)))
    else:
      self.offsets = newOffsets
      self.offsets2normDate = newOffset2NormDate
      self.additionalNameOffsets = newAdditionalNameOffsets
      self.nameOffsets = newNameOffsets
      self.origNameOffsets = newOrigNameOffsets
      self.additionalCorefFillers = newAdditionalCorefFillers

    self.ignoreQuoteSections()
    self.normalizeCasing()

  def isFullNameInDoc(self, listOfAlias):
    if self.docId in self.doc2wordsAndOffsets:
      # document has already been processed
      return 0
    # open document: get words and their offsets
    sentencesOffsets = getOffsets.getOffsets(self.docId, self.docPath)
    if len(sentencesOffsets) == 0:
      self.logger.error("error in parsing document")
      return 0
    thisSentences = " ".join([so[0] for so in sentencesOffsets])
    thisOffsets = " ".join([str(so[1]) for so in sentencesOffsets])
    self.doc2wordsAndOffsets[self.docId] = []
    for so in zip(thisSentences.split(), thisOffsets.split()):
      self.doc2wordsAndOffsets[self.docId].append(so)

    # test whether name is in document!
    sentenceString = thisSentences.decode('utf-8')
    sentenceString = re.sub(r'\s+', ' ', sentenceString)
    sentenceString = re.sub(r'(\W+)', ' \\1 ', sentenceString)
    sentenceString = re.sub(r'\s+', ' ', sentenceString)
    sentenceString = sentenceString.encode('utf-8')
    foundName = 0
    for myAlias in listOfAlias:
      myA = myAlias.decode('utf-8')
      myA = re.sub(r'(\W+)', ' \\1 ', myA)
      myA = re.sub(r'\s+', ' ', myA)
      myA = myA.encode('utf-8')
      sentenceString = re.sub(ur'(\W+)', ' \\1 ', sentenceString, re.UNICODE)
      sentenceString = re.sub(ur'\s+', ' ', sentenceString, re.UNICODE)
      foundList = compareNamesImprovedLc(myA, sentenceString)
      if len(foundList) > 0:
        foundName = 1
        break
    return foundName
    

  def setDocId(self, d):
    self.docId = d

  def setDocPath(self, docPath):
    self.docPath = docPath

  def resetDocInfos(self):
    self.docId = ""
    self.docPath = ""
    self.doc2wordsAndOffsets = {}

  def __init__(self, doCoref, loggerMain):
    self.docId = ""
    self.docPath = ""
    self.doc2wordsAndOffsets = {}
    self.isLcDoc = 0
    self.doCoref = doCoref

    self.logger = loggerMain.getChild(__name__)
