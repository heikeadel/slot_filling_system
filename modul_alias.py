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
import re
from nltk.corpus import stopwords
import string
import unicodedata

class Alias:

  def normalize(self, thisString):
    return unicodedata.normalize('NFD', thisString).encode('ascii', 'ignore')

  def getAlias(self, name):
    aliasList = []
    if name in self.nameToBaseName:
      basename = self.nameToBaseName[name]
      aliasList = self.baseNameToAliasList[basename]
    return aliasList

  def getAliasIR(self, name):
    aliasList = []
    if name in self.nameToBaseName:
      basename = self.nameToBaseName[name]
      if basename in self.baseNameToAliasListForIR:
        aliasList = self.baseNameToAliasListForIR[basename]
    return aliasList

  def completeListOfAlias(self):
    listOfAliasTmp = []
    if self.curType == "ORG":
      if " AG" in self.curName:
        alias = re.sub(ur' AG', '', self.curName, re.UNICODE)
        listOfAliasTmp.append(alias.strip())
      if "Organization" in self.curName:
        alias = re.sub(ur'Organization', 'Corporation', self.curName, re.UNICODE)
        listOfAliasTmp.append(alias)
        alias = re.sub(ur'Organization', 'Corp', self.curName, re.UNICODE)
        listOfAliasTmp.append(alias)
        alias = re.sub(ur'Organization', 'Corps', self.curName, re.UNICODE)
        listOfAliasTmp.append(alias)
        alias = re.sub(ur'Organization', 'Co', self.curName, re.UNICODE)
        listOfAliasTmp.append(alias)
        alias = re.sub(ur'Organization', '', self.curName, re.UNICODE)
        alias = alias.strip()
        listOfAliasTmp.append(alias)
      elif "Corporation" in self.curName:
        alias = re.sub(ur'Corporation', 'Corps', self.curName, re.UNICODE)
        listOfAliasTmp.append(alias)
        alias = re.sub(ur'Corporation', 'Corp', self.curName, re.UNICODE)
        listOfAliasTmp.append(alias)
        alias = re.sub(ur'Corporation', 'Co', self.curName, re.UNICODE)
        listOfAliasTmp.append(alias)
        alias = re.sub(ur'Corporation', 'Organization', self.curName, re.UNICODE)
        listOfAliasTmp.append(alias)
        alias = re.sub(ur'Corporation', '', self.curName, re.UNICODE)
        alias = alias.strip()
        listOfAliasTmp.append(alias)
      elif " Corps" in self.curName:
        alias = re.sub(ur'Corps', 'Corporation', self.curName, re.UNICODE)
        listOfAliasTmp.append(alias)
        alias = re.sub(ur'Corps', 'Corp', self.curName, re.UNICODE)
        listOfAliasTmp.append(alias)
        alias = re.sub(ur'Corps', 'Co', self.curName, re.UNICODE)
        listOfAliasTmp.append(alias)
        alias = re.sub(ur'Corps', 'Organization', self.curName, re.UNICODE)
        listOfAliasTmp.append(alias)
        alias = re.sub(ur'Corps', '', self.curName, re.UNICODE)
        alias = alias.strip()
        listOfAliasTmp.append(alias)
      elif " Corp" in self.curName:
        alias = re.sub(ur'Corp', 'Corporation', self.curName, re.UNICODE)
        listOfAliasTmp.append(alias)
        alias = re.sub(ur'Corp', 'Corps', self.curName, re.UNICODE)
        listOfAliasTmp.append(alias)
        alias = re.sub(ur'Corp', 'Co', self.curName, re.UNICODE)
        listOfAliasTmp.append(alias)
        alias = re.sub(ur'Corp', 'Organization', self.curName, re.UNICODE)
        listOfAliasTmp.append(alias)
        alias = re.sub(ur'Corp', '', self.curName, re.UNICODE)
        alias = alias.strip()
        listOfAliasTmp.append(alias)
      if " of " in self.curName:
        alias = re.sub(ur' of .*$', '', self.curName, re.UNICODE)
        alias = alias.strip()
        listOfAliasTmp.append(alias)
      if re.search(ur'^[Tt]he ', self.curName):
        alias = re.sub(ur'^[Tt]he ', '', self.curName)
        listOfAliasTmp.append(alias)
    elif self.curType == "PER":
      # add name without middle names:
      nameParts = self.curName.split()
      alias = ""
      if len(nameParts) > 2:
        alias = nameParts[0] + " " + nameParts[-1]
      listOfAliasTmp.append(alias)
      # add name without middle initial:
      alias = ""
      for np in nameParts:
        if len(np) == 2 and "." in np:
          continue
        elif len(np) > 1:
          alias += np + " "
      alias = alias.strip()
      listOfAliasTmp.append(alias)

      # add name splitted at hyphen
      if "-" in self.curName:
        alias = re.sub(ur'\s*\-\s*', ' ', self.curName, re.UNICODE)
        listOfAliasTmp.append(alias)

      if " Jr." in self.curName:
        alias = re.sub(ur'Jr\.', 'Junior', self.curName, re.UNICODE)
        listOfAliasTmp.append(alias)
        alias = re.sub(ur'Jr\.', 'Jr', self.curName, re.UNICODE)
        listOfAliasTmp.append(alias)
      elif " Jr" in self.curName:
        alias = re.sub(ur'Jr', 'Junior', self.curName, re.UNICODE)
        listOfAliasTmp.append(alias)
        alias = re.sub(ur'Jr', 'Jr.', self.curName, re.UNICODE)
        listOfAliasTmp.append(alias)
      elif " Junior" in self.curName:
        alias = re.sub(ur'Junior', 'Jr.', self.curName, re.UNICODE)
        listOfAliasTmp.append(alias)
        alias = re.sub(ur'Junior', 'Jr', self.curName, re.UNICODE)
        listOfAliasTmp.append(alias)

      # add nicknames
      for curAlias in [self.curName] + self.listOfAlias + listOfAliasTmp:
        nameParts = curAlias.split()
        for index,np1 in enumerate(nameParts):
          if np1 in self.name2nicknames:
            nicknames = self.name2nicknames[np1]
            for nick1 in nicknames:
              aliasNew = " ".join(nameParts[0:index] + [nick1] + nameParts[index+1:])
              self.logger.info("got alias from nickname: " + aliasNew)
              if not aliasNew in listOfAliasTmp:
                listOfAliasTmp.append(aliasNew)

    else: # LOC alias
      if re.search(ur'\s+[cC]ity$', self.curName):
        alias = re.sub(ur'\s+[cC]ity$', '', self.curName)
        listOfAliasTmp.append(alias)
      if re.search(ur'\s+[pP]rovince$', self.curName):
        alias = re.sub(ur'\s+[pP]rovince$', '', self.curName)
        listOfAliasTmp.append(alias)
      if re.search(ur'\s+[sS]tate$', self.curName):
        alias = re.sub(ur'\s+[sS]tate$', '', self.curName)
        listOfAliasTmp.append(alias)
      if re.search(ur'\s+[cC]ountry$', self.curName):
        alias = re.sub(ur'\s+[cC]ountry$', '', self.curName)
        listOfAliasTmp.append(alias)

    normed_name = self.normalize(self.curName)
    if normed_name != self.curName:
      listOfAliasTmp.append(normed_name)
  
    return listOfAliasTmp

  def getAliasForPersons(self):
    aliasForPersons = []
    if "PER" in self.curType:
      curNameParts = self.curName.split()
      firstName = curNameParts[0]
      lastName = curNameParts[-1]
      aliasForPersons.append(firstName)
      aliasForPersons.append(lastName)
    return aliasForPersons

  def setCurName(self, curName):
    self.curName = curName

  def setCurType(self, curType):
    self.curType = curType

  def resetListOfAlias(self):
    self.listOfAlias = []

  def deleteNamePartsBeforeFullName(self, curAliasOffsets, numberOfFullAlias, smallestFullOffset):
    # only allow name parts if they appear AFTER a full name / alias
    for ind in range(numberOfFullAlias, len(self.listOfAlias)):
      if ind in curAliasOffsets:
        aoNew = []
        for ao in curAliasOffsets[ind]:
          d, myOffsets = ao
          moNew = []
          for mo in myOffsets:
            offList = mo.split(',')
            toDelete = []
            for o in offList:
              if int(o) < smallestFullOffset:
                toDelete.append(o)
                self.logger.debug("offset too small: " + str(o))
            for toDel in toDelete:
              offList.remove(toDel)
            if len(offList) > 0:
              moNew.append(",".join(offList))
          if len(moNew) > 0:
            aoNew.append([d, moNew])
        if len(aoNew) > 0:
          curAliasOffsets[ind] = aoNew
        else:
          del curAliasOffsets[ind]
    return curAliasOffsets

  def deleteCorefMentionsBeforeFullName(self, nameOffsets, smallestFullOffset):
    nameOffsetsNew = []
    for no in nameOffsets:
      offList = no.split(',')
      toDelete = []
      for o in offList:
        if o == "":
          toDelete.append(o)
        elif int(o) < smallestFullOffset:
          toDelete.append(o)
      for toDel in toDelete:
        offList.remove(toDel)
      if len(offList) > 0:
        nameOffsetsNew.append(",".join(offList))
    return nameOffsetsNew


  def createListOfAliasForIR(self):
     # select alternative names to search documents for:
    self.listOfAlias = self.getAliasIR(self.curName)
    while self.curName in self.listOfAlias:
      self.listOfAlias.remove(self.curName)
    self.listOfAlias.insert(0, self.curName)

    if self.curType == "ORG":
      # for IR: search for name without Organization/Corp extension
      curNameWithoutDots = re.sub(ur'\.', '', self.curName, re.UNICODE)
      curNameWithoutDots = " " + curNameWithoutDots + " "
      newAlias = ""
      if " Organization " in curNameWithoutDots:
        newAlias = re.sub(ur' Organization', '', curNameWithoutDots, re.UNICODE)
      elif " Corps " in curNameWithoutDots:
        newAlias = re.sub(ur' Corps', '', curNameWithoutDots, re.UNICODE)
      elif " Co " in curNameWithoutDots:
        newAlias = re.sub(ur' Co', '', curNameWithoutDots, re.UNICODE)
      elif " Corporation " in curNameWithoutDots:
        newAlias = re.sub(ur' Corporation', '', curNameWithoutDots, re.UNICODE)
      elif " Corp " in curNameWithoutDots:
        newAlias = re.sub(ur' Corp', '', curNameWithoutDots, re.UNICODE)
      elif "Inc." in curNameWithoutDots:
        newAlias = re.sub(ur' Inc\.', '', curNameWithoutDots, re.UNICODE)
      elif " Inc" in curNameWithoutDots:
        newAlias = re.sub(ur' Inc', '', curNameWithoutDots, re.UNICODE)
      elif " Association of " in curNameWithoutDots:
        newAlias = re.sub(ur' Association of ', '', curNameWithoutDots, re.UNICODE)
      newAlias = newAlias.strip()
      if newAlias != "" and not newAlias in self.listOfAlias:
        self.listOfAlias[0] = newAlias
      # spelling variation of 'center'
      if " Center " in curNameWithoutDots:
        newAlias = re.sub(ur' Center ', ' Centre ', curNameWithoutDots, re.UNICODE)
        newAlias = newAlias.strip()
        if not newAlias in self.listOfAlias:
          self.listOfAlias.append(newAlias)
      elif " Centre " in curNameWithoutDots:
        newAlias = re.sub(ur' Centre ', ' Center ', curNameWithoutDots, re.UNICODE)
        newAlias = newAlias.strip()
        if not newAlias in self.listOfAlias:
          self.listOfAlias.append(newAlias)
      # add additional alias for IR: acronym
      acronym = ""
      for part in self.curName.split():
        if part[0].isupper():
          acronym += part[0]
      acronymLc = string.lower(acronym)
      if not len(acronym) <= 1 and not acronymLc in self.noAcronyms and not acronym in self.listOfAlias:
        self.listOfAlias.append(acronym)
      # cut 'of XX' off name
      if re.search(ur'^(.*?) of (.*?)$', self.curName, re.UNICODE):
        newAlias = re.sub(ur'^(.*?) of (.*?)$', '\\1', self.curName, re.UNICODE)
        if not newAlias in self.listOfAlias:
          self.listOfAlias.append(newAlias)
      # if self.curName is an url: get its basename:
      if re.search(ur'^(\w+\:\/\/)?(www\.)?([\w\_\-\.])+\.([A-Za-z]){2,}([\w\_\-\.\/\?\=])*$', self.curName, re.UNICODE):
        newAlias = re.sub(ur'^(\w+\:\/\/)?(www\.)?([\w\_\-\.]+)\.([A-Za-z]){2,}([\w\_\-\.\/\?\=])*$', '\\3', self.curName, re.UNICODE)
        if newAlias != self.curName:
          if not newAlias in self.listOfAlias:
            self.listOfAlias.append(newAlias)

  def createListOfAliasAfterIR(self):
    listOfAlias2 = []

    if self.listOfAlias[0] != self.curName:
      firstAlias = self.listOfAlias[0]
      self.listOfAlias.insert(0, self.curName) # if Inc/Corp etc extension has been removed: add original curName again to list of names!
      listOfAlias2.extend(self.getAlias(firstAlias)) # get alias for curName without Inc/Corp etc extension

    listOfAlias2.extend(self.getAlias(self.curName))

    for al in listOfAlias2:
      if not al in self.listOfAlias:
        self.listOfAlias.append(al)


    listOfAlias2 = self.completeListOfAlias()

    for al in listOfAlias2:
      if not al in self.listOfAlias:
        self.listOfAlias.append(al)

    if "ORG" in self.curType:
      furtherAlias = []
      for al in self.listOfAlias:
        for word in al.split():
          if re.search(ur'^[A-Z]+$', word, re.UNICODE): # possible abbreviation of name
            if not word in self.listOfAlias and not word in furtherAlias: # add as single alias
              furtherAlias.append(word)
      self.listOfAlias.extend(furtherAlias)

    if "" in self.listOfAlias:
      self.listOfAlias.remove("")

  def cleanListOfAlias(self):
    newList = []
    for alias in self.listOfAlias:
      if string.lower(alias) in self.noAlias:
        continue
      if len(alias) < 2: # don't allow single characters as aliases!
        continue
      if not re.search(ur'\w', alias):
        continue
      if self.curType == "ORG":
        if string.lower(alias) in self.countries or string.lower(alias) in self.states:
          continue
        if not alias in newList:
          newList.append(alias)
      elif self.curType == "GPE":
        # don't allow aliases like 'XX of GPE' (unless XX includes 'Republic') or 'XX in GPE'
        if " of " in alias:
          if "Republic " in alias or "State " in alias:
            if not alias in newList:
              newList.append(alias)
          else:
            pass
        elif " in " in alias:
          pass
        elif "," in alias:
          newAlias = re.sub(ur'^(.*?)\,.*$', '\\1', alias)
          if not newAlias in newList:
            newList.append(newAlias)
        elif "(" in alias:
          if ")" in alias:
            newAlias = re.sub(ur'\(.*?\)', '', alias)
            newAlias = newAlias.strip()
          else:
            newAlias = re.sub(ur'\(.*$', '', alias)
            newAlias = newAlias.strip()
          if not newAlias in newList:
            newList.append(newAlias)
        else:
          if not alias in newList:
            newList.append(alias)
      else:
        if len(re.findall('\d', alias)) > 2:
          continue # don't allow more than 2 digit in alias for PERSON
        if len(alias.split()) > 6:
          continue # don't allow more than 6 words in one name for PERSON
        if not alias in newList:
          newList.append(alias)

      # normalize not fully titled names apart from included stopwords
      if not (self.curType == "ORG" and alias.isupper()):
        # don't title acronyms!
        alias1 = ""
        for part in alias.split():
          if not part.istitle():
            if not part in self.stopwords:
              alias1 += part.title() + " "
            else:
              alias1 += part + " "
          else:
            alias1 += part + " "
        alias1 = alias1.strip()
        if alias1 != alias and not alias1 in newList:
          newList.append(alias1)
        # normalize non-titled names
        alias2 = alias.title()
        if alias2 != alias and not alias2 in newList:
          newList.append(alias2)

    self.listOfAlias = newList

  def __init__(self, countries, states, baseNameToAliasList, baseNameToAliasListForIR, nameToBaseName, name2nicknames, loggerMain):
    self.countries = countries
    self.states = states
    self.noAcronyms = {}
    for k in countries:
      self.noAcronyms[k] = 1
    for k in states:
      self.noAcronyms[k] = 1
    for k in ["usa", "us", "eu", "ai", "ag", "un", "uno", "nato", "who", "fao", "iaea", "icao", "ifad", "ilo", "imo", "imf", "itu", "unesco", "upu", "wbg", "wipo", "wmo", "unwto", "unodc"]:
      self.noAcronyms[k] = 1
    self.noAlias = {}
    for k in ["association", "company", "organization", "ag", "corporation", "corp", "corp.", "corps", "co", "junior", "senior", "jr.", "sr.", "jr", "sr", "inc.", "llc", "inc"]:
      self.noAlias[k] = 1
    self.baseNameToAliasListForIR = baseNameToAliasListForIR
    self.baseNameToAliasList = baseNameToAliasList
    self.nameToBaseName = nameToBaseName
    self.curName = ""
    self.listOfAlias = []
    self.stopwords = stopwords.words('english')
    self.name2nicknames = name2nicknames

    self.logger = loggerMain.getChild(__name__)
