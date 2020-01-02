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
import requests
import re
import json

class EntityLinking:

  def getWikiIdWAT(self, text, entity): # entity in the same form as it is contained in the text
    self.logger.debug('looking for entity ' + entity + ' with context ' + text + ' in wikipedia with WAT')
    url='https://wat.d4science.org/wat/tag/tag'
    token=ENTER_YOUR_AUTHORIZATION_TOKEN # see README for more information
    myParams = {'text' : text, 'gcube-token' : token}
    try:
      r = requests.get(url, params=myParams, timeout=100)
      try:
        resultList = r.json()["annotations"]
      except ValueError as j:
        self.logger.error("JSON: cannot decode: " + str(r))
        return -1
    except requests.exceptions.RequestException as e:
      self.logger.error("RequestException: " + str(e))
      return -1
    for rl in resultList:
      if rl["spot"] == entity:
        curId = int(rl["id"])
        return curId
    return -1

  def getWikiId(self, text, entity):
    result = self.getWikiIdWAT(text, entity) # interface to the outer classes; can be used to easily replace WAT with another entity linker
    return result

  def setEntityId(self, curName, nameOffset, sentences, offsets):
    nameStart = nameOffset[0]
    nameEnd = nameOffset[1] # end is the offset of the last token of the name (is not included in offsets)
    foundSentence = False
    for s, o in zip(sentences, offsets):
      offsetList = o.split()
      if int(nameStart) >= int(offsetList[0]) and int(nameStart) <= int(offsetList[-1]):
        # found correct sentence
        if nameStart in offsetList: # everything is fine
          indexStart = offsetList.index(nameStart)
          indexEnd = indexStart
          while indexEnd + 1 < len(offsetList) and int(offsetList[indexEnd + 1]) < int(nameEnd):
            indexEnd += 1
          nameString = " ".join(s.split()[indexStart : indexEnd + 1])
          foundSentence = True
          s_cleaned = self.cleanSentence(s)
          self.curEntityId = self.getWikiId(s_cleaned, nameString)
        else: # something wrong with offsets?
          # try to find curName:
          if curName in s:
            s_cleaned = self.cleanSentence(s)
            self.curEntityId = self.getWikiId(s_cleaned, curName)
            foundSentence = True
          else:
            self.logger.warning("Entity linking: could not find base entity offsets " + str(nameOffset) + " in " + o + " for sentence " + s)
        break
    if not foundSentence:
      # try to find curName somewhere in document:
      for s in sentences:
        if curName in s:
          self.logger.debug("Entity Linking: found name in " + s)
          s_cleaned = self.cleanSentence(s)
          self.curEntityId = self.getWikiId(s_cleaned, curName)
          foundSentence = True
          break
    if not foundSentence:
      self.logger.warning('Entity linking: could not find name of entity somewhere in document')
    else:
      self.logger.info('setting curEntityId to ' + str(self.curEntityId))
      if self.curEntityId == -1:
        self.logger.info('Entity has no ID in Wikipedia')

  def cleanSentence(self, sent):
    toClean = []
    sentenceList = sent.split()
    for indWord, curWord in enumerate(sentenceList):
      if curWord[0] == '<' and curWord[-1] == '>' and curWord != "<filler>" and curWord != "<name>":
        toClean.append(indWord)
      elif re.search(ur'^\&(amp\;)?lt\;.*\&(amp\;)?gt\;$', curWord, re.UNICODE):
        toClean.append(indWord)
      elif re.search(ur'^(\w+\:\/\/)?(www\.)?([\w\_\-\.])+\.([A-Za-z]){2,}([\w\_\-\.\/\?\=])*$', curWord, re.UNICODE):
        toClean.append(indWord)
        if indWord > 0 and sentenceList[indWord - 1] == "<":
          toClean.append(indWord - 1)
        if indWord < len(sentenceList) - 1 and sentenceList[indWord + 1] == ">":
          toClean.append(indWord + 1)
    sentenceList = sent.split()
    toCleanKeySorted = sorted(toClean, reverse = True)
    for curInd in toCleanKeySorted:
      sentenceList.pop(curInd)
    return " ".join(sentenceList)

  def isTheSameEntity(self, nameOffsets, sentences, offsets):
    if self.curEntityId == -2:
      self.logger.info('Entity Id has not been set yet')
      return True
    elif self.curEntityId == -1:
      return True
    else:
      # test mentions in document:
      for nameOccurrence in nameOffsets:
        nameId, nameO = nameOccurrence
        nameOList = nameO.split(',')
        nameStart = nameOList[0]
        nameEnd = nameOList[-1]
        for s, o in zip(sentences, offsets):
          offsetList = o.split()
          if nameStart in offsetList and nameEnd in offsetList:
            indexStart = offsetList.index(nameStart)
            indexEnd = offsetList.index(nameEnd)
            nameString = " ".join(s.split()[indexStart : indexEnd + 1])
            self.logger.debug("name string: " + nameString + ", corresponding offsets: " + str(nameO))
            s_cleaned = self.cleanSentence(s)
            testEntityId = self.getWikiId(s_cleaned, nameString)
            if testEntityId == -1 or self.curEntityId == testEntityId:
              return True
      return False

  def resetEntityId(self):
    self.curEntityId = -2

  def __init__(self, loggerMain):
    self.curEntityId = -2
    self.logger = loggerMain.getChild(__name__)
