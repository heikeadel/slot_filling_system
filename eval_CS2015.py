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

import logging
logging.basicConfig()

import re
import doQuery
import string
import doLoad
from utilities import tokenizeNames, isNameInSentenceOffsets, isSentenceAdditional, getsubidx
import copy
import numpy
from modul_postprocessing import Postprocessing
from modul_output import Output
from modul_candEvaluation import CandidateEvaluation
from modul_candExtraction import CandidateExtraction
from modul_alias import Alias
from modul_document import Document
from modul_discussionForum_document import DiscussionForumDocument
from modul_entityLinking import EntityLinking
import readNerAndCoref
import io
import cPickle
import os

class Evaluation():

  def process(self):

    isPerson = 0
    isOrg = 0
    isLoc = 0
    processedCurName = 1
    foundWebsiteInName = 1
    if self.curType == "PER":
      isPerson = 1
    elif self.curType == "ORG":
      isOrg = 1
      processedCurName = 0
      foundWebsiteInName = 0
    elif self.curType == "GPE":
      isLoc = 1
    else:
      self.logger.error("unknown type " + self.curType)
      return
    self.doc2wordsAndOffsets = {}
    doc2additionalCorefFillers = {}

    isPersonFiller = 1
    if self.curSlot != "" and not self.curSlot in self.slotsWithPerFillers:
      isPersonFiller = 0

    ######### setting information about entity in modules: ######

    self.alias.setCurName(self.curName)
    self.alias.setCurType(self.curType)
    if self.doEntityLinking == 1:
      self.entityLinker.resetEntityId()
    self.extraction.setCurName(self.curName)
    self.extraction.setType(self.curType)
    self.extraction.setCurSlot(self.curSlot)
    self.extraction.resetCandidates()
    self.postprocess.setCurType(self.curType)
    self.postprocess.setCurName(self.curName)
    self.postprocess.setCurQueryId(self.curQueryId)
    self.postprocess.setOrigSlot(self.curSlot)

    ########## ALIAS COMPONENT 1 #########################  

    self.alias.createListOfAliasForIR()
    self.logger.info("############## " + self.curQueryId + ": " + str(self.alias.listOfAlias) + " #############")
    self.logger.info("######## slot: " + self.curSlot + " #########")

    ###################################################

    ########## IR COMPONENT ############################
    # first: look for documents containing name
    maxCountD = 100
    queryResult = doQuery.query(self.alias.listOfAlias, 3*maxCountD,  self.docList, isLoc, terrierDir=self.terrierDir)
    #queryResult = []

    if not self.curDocId in queryResult:
      queryResult.insert(0, self.curDocId)
    else:
      if not queryResult.index(self.curDocId) == 0: # start with sample document (because of entity linking and because it might be an important document)
        queryResult.remove(self.curDocId)
        queryResult.insert(0, self.curDocId)

    # second: extract sentences with markables of name
    countD = 0
    globalSlot2fillerCandidatesAndConfidence = {}

    ########################################################

    aliasOffsets = {}
    curNameOffsets = {}

    ########### ALIAS COMPONENT 2 ################################

    self.alias.createListOfAliasAfterIR()
    self.alias.cleanListOfAlias()
    originalListOfAlias = copy.deepcopy(self.alias.listOfAlias)
    originalListOfTokenizedAlias = tokenizeNames(originalListOfAlias, tmpfilename=self.resultfile)

    # for persons: if we know that full name is in document, we also want to extract sentences with only the first name or only the last name!
    aliasForPersons = self.alias.getAliasForPersons()
    aliasForPersonsTokenized = tokenizeNames(aliasForPersons, tmpfilename=self.resultfile)

    numberOfFullAlias = len(self.alias.listOfAlias)

    self.logger.info("searching for the following aliases: " + str(self.alias.listOfAlias))

    ##############################################################

    doc2offsets2normDate = {}
    curNameOffsets = {}

    spellingVariationsOffsets = []

    self.discussionForumDocument.resetDocInfos()
    self.document.resetDocInfos()

    for d in queryResult:

      self.listOfAlias = copy.deepcopy(originalListOfAlias)
      listOfTokenizedAlias = copy.deepcopy(originalListOfTokenizedAlias)
  
      if d == "eng-NG-31-131713-9425730" or d == "bolt-eng-DF-170-181103-8916659" or d == "bolt-eng-DF-212-191668-3057972": # this document cannot be tagged because sentences cannot be split
        continue

      if "bolt" in d:
        thisDocument = self.discussionForumDocument
      else:
        thisDocument = self.document
  
      if countD > maxCountD:
        break
      if not d in self.doc2path:
        self.logger.error("doc id " + d + " not in docid2path file")
        continue

      self.logger.info("opening " + d)

      docPath = self.doc2path[d]

      thisDocument.setDocId(d)
      thisDocument.setDocPath(docPath)

      if not thisDocument.isFullNameInDoc(self.listOfAlias): # full name in doc?
        self.logger.debug("did not find name " + str(self.listOfAlias) + " in file " + docPath)
        continue

      self.doc2wordsAndOffsets[d] = thisDocument.doc2wordsAndOffsets[d]

      #############################################################
      # for persons: we know that full name is in document: so we also want to extract sentences with only the first name or only the last name!
      if isPerson and len(self.listOfAlias) == numberOfFullAlias: # nothing has been added so far
        self.listOfAlias.extend(aliasForPersons)
        listOfTokenizedAlias.extend(aliasForPersonsTokenized)
      
      if isPerson:
        self.extraction.setTriggerWordsHyphen(self.triggerWordsHyphenPER)
      elif isOrg:
        self.extraction.setTriggerWordsHyphen(self.triggerWordsHyphenORG)
      else:
        self.extraction.setTriggerWordsHyphen(self.triggerWordsHyphenGPE)

      if thisDocument.preprocessDoc(listOfTokenizedAlias, isPerson, isOrg, isLoc, isPersonFiller) == -1:  # preprocess doc
        continue # an ERROR occurred!

      # results of preprocessing:
      nameOffsets = thisDocument.nameOffsets
      origNameOffsets = thisDocument.origNameOffsets
      offsets2normDate = thisDocument.offsets2normDate
      additionalNameOffsets = thisDocument.additionalNameOffsets

      doc2offsets2normDate[d] = offsets2normDate
   
      additionalNameOffsetsFlattened = [val for sublist in additionalNameOffsets for val in sublist]

      self.extraction.setDocumentInfos(thisDocument.sentences, thisDocument.nerInSentences, thisDocument.offsets, thisDocument.posInSentences, thisDocument.lemmasInSentence)
      self.extraction.resetCandidatesForDoc()

      curAliasOffsets, curNameOffsetsThisDoc, smallestFullOffset = self.extraction.getMappingAlias2Offsets(self.listOfAlias, numberOfFullAlias, d, origNameOffsets)
      if not d in curNameOffsets:
        curNameOffsets[d] = []
      if d in curNameOffsetsThisDoc:
        curNameOffsets[d] = copy.deepcopy(curNameOffsetsThisDoc[d])
      curAliasOffsets = self.alias.deleteNamePartsBeforeFullName(curAliasOffsets, numberOfFullAlias, smallestFullOffset)
      nameOffsets = self.alias.deleteCorefMentionsBeforeFullName(nameOffsets, smallestFullOffset)
  
      # check whether there appears a full name / alias at all:
      foundAlias = sorted(list(curAliasOffsets.keys()))
      if len(foundAlias) == 0 or foundAlias[0] >= numberOfFullAlias:
        # no full name / alias in document (another person has the same name)
        self.logger.debug("did not find any full name in document " + d)
        origNameOffsets = []

      if len(origNameOffsets) == 0:
        self.logger.debug("did not find offsets for name " + str(self.listOfAlias) + " in file " + docPath)
        # set entity id nevertheless
        if d == self.curDocId:
          if self.doEntityLinking == 1:
            self.entityLinker.setEntityId(self.curName, [self.curBegin, self.curEnd], thisDocument.sentences, thisDocument.offsets)
        continue

      for newAliasOffset in curAliasOffsets:
        if not newAliasOffset in aliasOffsets:
          aliasOffsets[newAliasOffset] = []
        aliasOffsets[newAliasOffset].append(curAliasOffsets[newAliasOffset])

      if self.doEntityLinking == 1:
        if d == self.curDocId:
          self.entityLinker.setEntityId(self.curName, [self.curBegin, self.curEnd], thisDocument.sentences, thisDocument.offsets)
        else:
          if not self.entityLinker.isTheSameEntity(origNameOffsets, thisDocument.sentences, thisDocument.offsets):
            self.logger.debug("Entity in document " + d + " is another entity according to entity linker!")
            continue
 
      countD += 1
  
      slot2fillerCandidates = {}

      ####################### SPECIAL SLOT: WEBSITE ########################
  
      if isOrg:
        # special slot: org:website: look for websites in whole document!
        s = "org:website"
        if foundWebsiteInName == 0 and (s == self.curSlot or self.curSlot == ""):
          foundWebsite = self.extraction.getWebsiteResults(self.listOfAlias, d)
          if len(foundWebsite) > 0:
            if not s in globalSlot2fillerCandidatesAndConfidence:
              globalSlot2fillerCandidatesAndConfidence[s] = []
            if len(foundWebsite) != 7:
              self.logger.error("something is wrong with website extraction result")
            globalSlot2fillerCandidatesAndConfidence[s].append(foundWebsite)

        if processedCurName == 0:
          # try to extract fillers from self.curName directly
          s = "org:website"
          if s == self.curSlot or self.curSlot == "":
            if d in curNameOffsetsThisDoc and self.extraction.isWebsiteInName() > -1: # cur name appears in doc:
              candidateOffsets = curNameOffsetsThisDoc[d][0]
              if not s in globalSlot2fillerCandidatesAndConfidence:
                globalSlot2fillerCandidatesAndConfidence[s] = []
              candidate = [self.curName, self.curName, 1.0, candidateOffsets, " ".join(candidateOffsets.split(',')), d, "STRING"]
              self.logger.info("found candidate for website in name: " + str(candidate))
              globalSlot2fillerCandidatesAndConfidence[s] = [candidate] # single valued slot
              foundWebsiteInName = 1
          s = "org:location_of_headquarters"      
          if s == self.curSlot or self.curSlot == "" and d in curNameOffsetsThisDoc:
            curNameList = self.curName.split()
            curNameListSize = len(curNameList)
            subwords = [" ".join(curNameList[i:i+size]) for size in range(1, curNameListSize + 1) for i in range(0, curNameListSize - size + 1)]
            subwordsSorted = sorted(subwords, key=lambda x:len(x.split()), reverse = True)
            for subName in subwordsSorted:
              if string.lower(subName) in self.countries or string.lower(subName) in self.states or string.lower(subName) in self.cities:
                if string.lower(" of " + subName) in string.lower(self.curName) or re.search(r'^' + re.escape(subName), self.curName):
                  candidateOffsets = curNameOffsetsThisDoc[d][0]
                  indices = getsubidx(curNameList, subName.split())
                  if len(indices) == 0:
                    self.logger.warning("error in extraction headquarters of self.curName: possible headquarter: " + subName + ", self.curName: " + self.curName)
                    continue
                  fillerOffsets = ",".join(candidateOffsets.split(',')[indices[0] : indices[0]+len(subName.split())])
                
                  if not s in globalSlot2fillerCandidatesAndConfidence:
                    globalSlot2fillerCandidatesAndConfidence[s] = []
                  candidate = [subName, self.curName, 1.0, fillerOffsets, " ".join(candidateOffsets.split(',')), d, "GPE"]
                  globalSlot2fillerCandidatesAndConfidence[s].append(candidate)
                  self.logger.info("found candidate for location of headquarters in name: " + str(candidate))
                  break
          processedCurName = 1 

      #####################################################################

      ############ FIND SENTENCES WITH NAME ###############################

      # get sentences where mention of name appears
      self.extraction.getSentencesWithName(nameOffsets, additionalNameOffsetsFlattened)

      # clean sentences from HTML tags
      self.extraction.cleanFromHTML()

      ######################################################################

      ############## EXTRACT CANDIDATES ####################################

      self.extraction.extractCandidates(nameOffsets, origNameOffsets, additionalNameOffsets, d, thisDocument.additionalCorefFillers)

      doc2additionalCorefFillers[d] = thisDocument.additionalCorefFillers

      self.logger.info("this was the " + str(countD) + "-th document for name " + self.curName)
  
      extractionResults = self.extraction.globalSlot2fillerCandidatesAndConfidence
      for s in extractionResults:
        if not s in globalSlot2fillerCandidatesAndConfidence:
          globalSlot2fillerCandidatesAndConfidence[s] = []
        for x in extractionResults[s]:
          if len(x) != 7: 
            self.logger.error("something is wrong with extraction result")
        globalSlot2fillerCandidatesAndConfidence[s].extend(extractionResults[s])

      spellingVariationsOffsets.extend(thisDocument.spellingVariationsOffsets)


    ################# CANDIDATE EVALUATION ################################
 
    # 3.: evaluate candidates with PAT+SVM+CNN, patterns or proximity

    self.evaluation.resetGlobalConfidences()

    self.evaluation.setForProximity(self.extraction.forProximity)
    self.evaluation.setForClassifier(self.extraction.forClassifier)
    self.evaluation.setForPatternMatcher(self.extraction.forPatternMatcher)

    self.evaluation.evaluateProximity()
    self.evaluation.evaluateClassifiers()
    self.evaluation.evaluatePatternMatcher()

    evaluationResults = self.evaluation.globalSlot2fillerCandidatesAndConfidence
    for s in evaluationResults:
      if not s in globalSlot2fillerCandidatesAndConfidence:
        globalSlot2fillerCandidatesAndConfidence[s] = []
      for x in evaluationResults[s]:
        if len(x) != 7:
          self.logger.error("something is wrong with evaluation result")
      globalSlot2fillerCandidatesAndConfidence[s].extend(evaluationResults[s])

    ##### end of loop over query results ######
  
    ######################### POSTPROCESSING COMPONENT #################################

    self.postprocess.resetResults()
    self.postprocess.setCandidates(globalSlot2fillerCandidatesAndConfidence)
    self.postprocess.applyThreshold()
    
    if "cit" in self.curSlot or "state" in self.curSlot or "countr" in self.curSlot or self.curSlot == "":
      self.postprocess.postprocessLocation()
    
    self.postprocess.setListOfAlias(self.listOfAlias)
    if "per:alternate_names" == self.curSlot or self.curSlot == "org:alternate_names" or self.curSlot == "":
      self.postprocess.alternateNamesFillers(aliasOffsets, curNameOffsets, spellingVariationsOffsets, self.doc2wordsAndOffsets)

    if "date" in self.curSlot or self.curSlot == "":
      self.postprocess.postprocessDates(self.slots_orig, doc2offsets2normDate)

    self.postprocess.rankingPerSlot(queryResult)

    self.postprocess.postprocessForOutput(self.slots_orig, self.doc2wordsAndOffsets, doc2additionalCorefFillers)

    self.postprocess.reduceFPs()

    for mr in self.postprocess.myResults:
      self.logger.info("FINAL\t" + self.origName + "\t" + mr)

    ################################ OUTPUT COMPONENT ###############################################

    self.output.setResults(self.postprocess.myResults)

    self.output.writeResults(self.resultfile)

    self.logger.info("found name " + self.curName + " in " + str(countD) + " documents")

    ######################### FINISHED ##################################################  


  def __init__(self, queryfile, resultfile, terrierDir, doEntityLinking, doCoref, weightfile, thresholdfile, svmVersion="binarySkip", cnnVersion="binary"):

    self.logger = logging.getLogger(__name__)
    self.logger.setLevel(logging.DEBUG)

    self.doEntityLinking = doEntityLinking
    self.doCoref = doCoref
    self.resultfile = resultfile
    self.terrierDir = terrierDir
  
    ## read slot2type ###
  
    slot2typeInfos = doLoad.readSlot2Type2015()
    self.slots_orig = slot2typeInfos[0]
    self.slots = slot2typeInfos[1]
    slot2types = slot2typeInfos[2]
    type2slots = slot2typeInfos[3]
  
    ## read valid document ids ###
    self.docList = doLoad.readDocList()
  
    ## read patterns ###
    self.rothPatterns = doLoad.readPatterns(PATH_TO_PATTERNS)
  
    ## read alias list ###
    aliasResults = doLoad.readAliasList()
    self.baseNameToAliasList = aliasResults[0]
    self.nameToBaseName = aliasResults[1]
    self.baseNameToAliasListForIR = aliasResults[2]
  
    ## read docid2path ###
    self.doc2path = doLoad.readDocId2Path()
  
    ## read world knowledge ###
    worldResults = doLoad.readWorldKnowledge(self.baseNameToAliasList, self.nameToBaseName, self.baseNameToAliasListForIR)
    self.countries = worldResults[0]
    self.states = worldResults[1]
    self.baseNameToAliasList = worldResults[2]
    self.nameToBaseName = worldResults[3]
    self.baseNameToAliasListForIR = worldResults[4]
    self.cities = worldResults[5]

    city2stateAndCountry, state2country = doLoad.readInferenceInformation()
    slot2inverse = doLoad.getSlot2Inverse()

    slotsForPatternMatching = ['org:political_religious_affiliation', 'org:shareholders', 'org:date_dissolved', 'org:number_of_employees_members', 'per:religion', 'per:charges']
    for s in slot2inverse:
      sEval = slot2inverse[s]
      if sEval in slotsForPatternMatching:
        slotsForPatternMatching.append(s)
  
    websiteRegex = "^(\w+\:\/\/)?(www\.)?([\w\_\-\.])+\.([A-Za-z]){2,}(\W)?$"
  
    slot2proximity = {'per:other_family': ['aunts', 'aunt', 'brother-in-law', 'cousins', 'cousin', 'granchild', 'grandchildren', 'granddaughter', 'granddaughters', 'grandfather', 'grandmother', 'grandparents', 'grandson', 'grandsons', 'nephews', 'nephew', 'niece', 'nieces', 'sister-in-law', 'uncle', 'uncles', 'partner']}
    for s in slot2inverse:
      sEval = slot2inverse[s]
      if sEval in slot2proximity:
        slot2proximity.append(s)
  
    self.triggerWordsHyphenORG = {"org:location_of_headquarters" : ["-based"], "org:number_of_employees_members" : ["-member"], "org:students" : ["-educated"]}
    self.triggerWordsHyphenPER = {"per:location_of_birth" : ["-born"], "per:age" : ["-year-old", "- year-old", "-month-old", "- month-old"], "per:schools_attended" : ["-educated"], "per:locations_of_residence" : ["-based"]}
    self.triggerWordsHyphenGPE = {"gpe:headquarters_in_location" : ["-based"], "gpe:births_in_location" : ["-born"]}

    self.slotsWithPerFillers = ["per:children", "per:parents", "per:other_family", "per:siblings", "per:spouse", "org:employees_or_members", "gpe:employees_or_members", "org:students", "gpe:births_in_location", "gpe:deaths_in_location", "gpe:residents_of_location", "org:shareholders", "org:founded_by", "org:top_members_employees", "gpe:births_in_city", "gpe:deaths_in_stateorprovince", "gpe:residents_of_country", "gpe:births_in_stateorprovince", "gpe:deaths_in_city", "gpe:residents_of_city", "gpe:births_in_country", "gpe:deaths_in_country", "gpe:residents_of_stateorprovince"]
  
    ## read possible slot fills for slots "cause of death", "title", "charges"
    slot2possibleFills = doLoad.readSpecialSlotFills()
  
    name2nicknames = doLoad.getNicknames()

    ### read slot to weight information for classification ####
    slot2weightsSVM, slot2weightsCNN, slot2weightsPAT = doLoad.readSlot2WeightFile(weightfile)

    ### read thresholds for output ####
    slot2thresholdPF = doLoad.readSlotThresholdFile(thresholdfile)
    for s in slot2inverse:
      sEval = slot2inverse[s]
      if sEval in slot2thresholdPF:
        slot2thresholdPF[s] = slot2thresholdPF[sEval]

    self.doc2wordsAndOffsets = {}

    #### initialize modules #####
    self.alias = Alias(self.countries, self.states, self.baseNameToAliasList, self.baseNameToAliasListForIR, self.nameToBaseName, name2nicknames, self.logger)
    self.entityLinker = None
    if self.doEntityLinking == 1:
      self.entityLinker = EntityLinking(self.logger)
    self.document = Document(self.doCoref, self.logger)
    self.discussionForumDocument = DiscussionForumDocument(self.doCoref, self.logger)
    self.extraction = CandidateExtraction(self.slots, slot2possibleFills, slot2types, type2slots, slotsForPatternMatching, slot2proximity, websiteRegex, self.countries, self.states, self.cities, self.logger)
    self.evaluation = CandidateEvaluation(slot2proximity, slot2weightsSVM, slot2weightsCNN, slot2weightsPAT, self.rothPatterns, slot2inverse, svmVersion=svmVersion, cnnVersion=cnnVersion, loggerMain = self.logger)
    self.postprocess = Postprocessing(self.slots, self.countries, self.states, self.cities, slot2thresholdPF, city2stateAndCountry, state2country, self.doc2path, name2nicknames, 1, terrierDir=self.terrierDir, aliasModule = self.alias, loggerMain = self.logger)
    self.output = Output(self.logger)

    ## read and process queries ###
    self.curQueryId = ""
    self.curName = ""
    self.curDocId = ""
    self.curType = ""
    self.curBegin = 0
    self.curEnd = 0
    self.curSlot = ""
    self.origName = ""
    f = io.open(queryfile, encoding="utf-8")
    for line in f:
      line = line.strip()
      if "</query>" in line:
        if self.curType in ["PER", "ORG", "GPE"]:
          self.process()
      elif "<query id=" in line:
        # read new query id
        self.curQueryId = re.sub(ur'^\s*\<query id\=\"(.*?)\"\>\s*$', '\\1', line, re.UNICODE)
      elif "<name>" in line:
        # read new name
        self.curName = re.sub(ur'^\s*\<name\>(.*?)\<\/name\>\s*$', '\\1', line, re.UNICODE)
        self.origName = self.curName
      elif "<docid>" in line:
        # read new doc id
        self.curDocId = re.sub(ur'^\s*\<docid\>(.*?)\<\/docid\>\s*$', '\\1', line, re.UNICODE)
      elif "<enttype>" in line:
        # read new type
        self.curType = re.sub(ur'^\s*\<enttype\>(.*?)\<\/enttype\>\s*$', '\\1', line, re.UNICODE)
        if self.curType.islower():
          self.curType = self.curType.upper()
        if self.curType in ["PER", "GPE"] and self.curName.isupper(): # normalize uppercased names
          self.curName = self.curName.title()
      elif "<beg>" in line:
        # read begin offset
        self.curBegin = re.sub(ur'^\s*\<beg\>(.*?)\<\/beg\>\s*$', '\\1', line, re.UNICODE)
      elif "<end>" in line:
        # read end offset
        self.curEnd = re.sub(ur'^\s*\<end\>(.*?)\<\/end\>\s*$', '\\1', line, re.UNICODE)
      elif "<slot>" in line:
        # read slot to be filled
        self.curSlot = re.sub(ur'^\s*\<slot\>(.*?)\<\/slot\>\s*$', '\\1', line, re.UNICODE)
  
    f.close()

