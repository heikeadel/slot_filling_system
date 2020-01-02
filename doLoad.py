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
import string
import io

import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
logger.setLevel(logging.DEBUG)

def readInferenceInformation():
  toDeleteCity = []
  toDeleteState = []
  city2stateAndCountry = {}
  state2country = {}
  f = io.open("data/city2stateAndCountry", 'r', encoding="utf-8")
  for line in f:
    line = line.strip()
    parts = (line + " ").split(' :: ')
    if len(parts) != 3:
      logger.warning("unexpected line in file city2stateAndCountry: " + line)
      continue
    city, state, country = parts
    country = country.strip()
    if city in city2stateAndCountry:
      toDeleteCity.append(city) # do not use ambiguous information!
    else:
      city2stateAndCountry[city] = (state, country)
  f.close()
  f = io.open("data/state2country", 'r', encoding="utf-8")
  for line in f:
    line = line.strip()
    parts = (line + " ").split(' :: ')
    if len(parts) != 2:
      logger.warning("unexpected line in file state2country: " + line)
      continue
    state, country = parts
    country = country.strip()
    if country == "":
      continue
    if state in state2country:
      toDeleteState.append(state) # do not use ambiguous information!
    else:
      state2country[state] = country
  f.close()

  # delete ambiguous information
  for dc in list(set(toDeleteCity)):
    del city2stateAndCountry[dc]
  for ds in list(set(toDeleteState)):
    del state2country[ds]

  return city2stateAndCountry, state2country

def readSlot2WeightFile(filename):
  slot2weightsSVM = {}
  slot2weightsCNN = {}
  slot2weightsPAT = {}
  f = io.open(filename, 'r', encoding="utf-8")
  for line in f:
    line = line.strip()
    parts = line.split('\t')
    slot = parts[0]
    weightSVM = parts[1]
    weightCNN = parts[2]
    weightPAT = parts[3]
    slot2weightsPAT[slot] = float(weightPAT)
    slot2weightsCNN[slot] = float(weightCNN)
    slot2weightsSVM[slot] = float(weightSVM)
  f.close()
  return slot2weightsSVM, slot2weightsCNN, slot2weightsPAT

def readSlotThresholdFile(filename):
    # reading slot2threshold specific for pattern match information
    slot2thresholdPF = {}
    f = io.open(filename, 'r', encoding='utf-8')
    for line in f:
      parts = line.strip().split(' : ')
      slot = parts[0]
      thresh0 = float(parts[1])
      if not slot in slot2thresholdPF:
        slot2thresholdPF[slot] = [thresh0, thresh0]
      else:
        logger.warning("got several pf thresholds for slot " + slot + " - taking the first one")
    f.close()
    return slot2thresholdPF

def readSlot2Type2015():
  slots_orig = []
  slots = []
  slot2types = {}
  type2slots = {}
  f = io.open("data/slots2types2015", 'r', encoding="utf-8")
  for line in f:
    line = line.strip()
    if not "per:" in line and not "org:" in line and not "gpe" in line:
      continue
    parts = line.split()
    slot = parts[0]
    if not slot in slots_orig:
      slots_orig.append(slot)
    if "cit" in slot or "countr" in slot or "province" in slot:
      slot = re.sub(ur'city', 'location', slot, re.UNICODE)
      slot = re.sub(ur'country', 'location', slot, re.UNICODE)
      slot = re.sub(ur'statesorprovinces', 'locations', slot, re.UNICODE)
      slot = re.sub(ur'cities', 'locations', slot, re.UNICODE)
      slot = re.sub(ur'countries', 'locations', slot, re.UNICODE)
      slot = re.sub(ur'stateorprovince', 'location', slot, re.UNICODE)
    types = parts[1]
    typeList = types.split(',')
    if slot in slot2types:
      for t in typeList:
        if not t in slot2types[slot]:
          slot2types[slot].append(t)
    else:
      slot2types[slot] = typeList
      slots.append(slot)
    for t in typeList:
      if not t in type2slots:
        type2slots[t] = []
      type2slots[t].append(slot)
  f.close()
  return [slots_orig, slots, slot2types, type2slots]

def readDocList():
  docList = {}
  f = io.open('data/docList_corpus2015', 'r', encoding='utf-8')
  for line in f:
    line = line.strip()
    docList[line] = 1
  f.close()
  return docList

def readSpecialSlotFills():
  resultMap = {}
  for s in ["per:title", "per:charges", "per:cause_of_death", "per:religion", "org:political_religious_affiliation"]:
    resultMap[s] = []
    filename = "data/" + s + "_fromFreebase_cleanedManually"
    f = io.open(filename, 'r', encoding="utf-8")
    for line in f:
      line = line.strip()
      lineLc = string.lower(line)
      lineLc1 = lineLc
      lineLc2 = lineLc
      if " & " in lineLc:
        lineLc1 = re.sub(ur' \& ', ' and ', lineLc, re.UNICODE)
        lineLc2 = re.sub(ur'\&', ' and ', lineLc, re.UNICODE)
      elif "&" in lineLc:
        lineLc1 = re.sub(ur'\&', ' & ', lineLc, re.UNICODE)
        lineLc2 = re.sub(ur'\&', ' and ', lineLc, re.UNICODE)
      elif " and " in lineLc:
        lineLc1 = re.sub(ur' and ', ' & ', lineLc, re.UNICODE)
        lineLc2 = re.sub(ur' and ', '&', lineLc, re.UNICODE)
      if not lineLc1 in resultMap[s]:
        resultMap[s].append(lineLc1)
        if s == "per:title":
          resultMap[s].append(lineLc1 + 's') # allow plural forms of titles
      if not lineLc2 in resultMap[s]:
        resultMap[s].append(lineLc2)
        if s == "per:title":
          resultMap[s].append(lineLc2 + 's') # allow plural forms of titles
    f.close()
  return resultMap

def readPatterns(filename):
  patterns = {}
  f = io.open(filename, 'r', encoding="utf-8")
  for line in f:
    line = line.strip()
    parts = line.split(' : ')
    slot = parts[0]
    pattern = " : ".join(parts[1:])
    # escape pattern
    pattern_reg = re.escape(pattern.lower())
    pattern_reg = re.sub(ur'\\\*\\ ', '(\S+ ){0,4}', pattern_reg, re.UNICODE)
    pattern_reg = ".*" + pattern_reg + ".*" 
    # compile patten
    compiledPattern = re.compile(pattern_reg)
    # store compiled pattern
    if not slot in patterns:
      patterns[slot] = []
    patterns[slot].append(compiledPattern)
  f.close()
  return patterns

def readAliasList():
  baseNameToAliasList = {}
  nameToBaseName = {}
  f = io.open("data/alias_englishWiki.cleaned", 'r', encoding="utf-8")
  for line in f:
    if "ERROR" in line:
      continue
    line = line.strip()
    parts = line.split(' :: ')
    if len(parts) < 2:
      continue # no alias provided
    name = parts[0]
    if name in nameToBaseName:
      continue # do not overwrite already existing names 
    else:
      nameToBaseName[name] = name
      basename = name
      tmpAliasList = [basename]
    for i in range(1, len(parts)):
      myName = parts[i]
      if myName in nameToBaseName:
        continue 
      else:
        nameToBaseName[myName] = basename
      tmpAliasList.append(myName)
    baseNameToAliasList[basename] = tmpAliasList
  f.close()

  baseNameToAliasForIR = {}
  f = io.open("data/alias_englishWiki.forIR", 'r', encoding="utf-8")
  for line in f:
    if "ERROR" in line:
      continue
    line = line.strip()
    parts = line.split(' :: ')
    if len(parts) < 1:
      continue # no alias provided
    name = parts[0]
    if name in nameToBaseName:
      basename = nameToBaseName[name]
      tmpAliasList = [name]
      if basename != name:
        tmpAliasList.append(basename)
    else:
      basename = name
      nameToBaseName[name] = basename
      tmpAliasList = [name]
    for i in range(1, len(parts)):
      myName = parts[i]
      myName = myName.encode('utf-8')
      myName = re.sub(ur'\xe2\x80\x93', '-', myName, re.UNICODE)
      myName = myName.decode('utf-8')
      equalName = 0
      for t in tmpAliasList:
        if string.lower(t) == string.lower(myName):
          equalName = 1
          break
      if equalName == 0:
        tmpAliasList.append(myName)
    baseNameToAliasForIR[basename] = tmpAliasList
  f.close()

  return [baseNameToAliasList, nameToBaseName, baseNameToAliasForIR]

def readDocId2Path():
  doc2pathfile = "data/docId2path_corpus2015"
  doc2path = {}
  f = io.open(doc2pathfile, 'r', encoding="utf-8")
  for line in f:
    line = line.strip()
    parts = line.split()
    docid = parts[0]
    docpath = parts[1]
    doc2path[docid] = docpath
  f.close()
  return doc2path

def readWorldKnowledge(baseNameToAliasList, nameToBaseName, baseNameToAliasForIR):
  countryfile = "data/list_countryAbbr.cleaned"
  countries = {}
  f = io.open(countryfile, 'r', encoding="utf-8")
  for line in f:
    line = line.strip()
    if line == "" or line[0] == '#':
      continue
    parts = line.split(' : ')
    for p in parts:
      if "-" in p:
        p_new = " ".join(p.split('-'))
        countries[string.lower(p_new)] = 1
      countries[string.lower(p)] = 1
    if len(parts) > 1: # new: 2015 July 3
      basename = parts[1]
      if "-" in basename:
        basename = " ".join(basename.split('-'))
      if basename in baseNameToAliasList:
        if not basename in nameToBaseName:
          nameToBaseName[basename] = basename
        # append additional entries to existing list
        myName = parts[0]
        if not myName in nameToBaseName and not myName in baseNameToAliasList[basename]:
          baseNameToAliasList[basename].append(myName)
        for i in range(2, len(parts)):
          myName = parts[i]
          if not myName in nameToBaseName and not myName in baseNameToAliasList[basename]:
            baseNameToAliasList[basename].append(myName)
      else:
        nameToBaseName[basename] = basename
        tmpAliasList = [basename]
        myName = parts[0]
        if not myName in nameToBaseName: # skip entries which have already been appended
          tmpAliasList.append(myName)
          nameToBaseName[myName] = basename
        for i in range(2, len(parts)):
          myName = parts[i]
          if not myName in nameToBaseName: # skip entries which have already been appended
            tmpAliasList.append(myName)
            nameToBaseName[myName] = basename
        baseNameToAliasList[basename] = tmpAliasList 
      if basename in baseNameToAliasForIR:
        if not basename in nameToBaseName:
          nameToBaseName[basename] = basename
        # append additional entries to existing list
        myName = parts[0]
        if not myName in nameToBaseName and not myName in baseNameToAliasForIR[basename]:
          baseNameToAliasForIR[basename].append(myName)
      else:
        nameToBaseName[basename] = basename
        tmpAliasList = [basename]
        myName = parts[0]
        if not myName in nameToBaseName: # skip entries which have already been appended
          tmpAliasList.append(myName)
          nameToBaseName[myName] = basename
        baseNameToAliasForIR[basename] = tmpAliasList

  f.close()

  statefile = "data/list_stateProvinces.cleaned"
  states = {}
  f = io.open(statefile, 'r', encoding="utf-8")
  for line in f:
    line = line.strip()
    parts = line.split(' : ')
    for p in parts:
      states[string.lower(p)] = 1
    if len(parts) > 1:
      basename = parts[0]
      if basename in baseNameToAliasList:
        if not basename in nameToBaseName:
          nameToBaseName[basename] = basename
        # append additional entries to existing list
        for i in range(1, len(parts)):
          myName = parts[i]
          if not myName in nameToBaseName and not myName in baseNameToAliasList[basename]:
            baseNameToAliasList[basename].append(myName)
      else:
        nameToBaseName[basename] = basename
        tmpAliasList = [basename]
        for i in range(1, len(parts)):
          myName = parts[i]
          if not myName in nameToBaseName: # skip entries which have already been appended
            tmpAliasList.append(myName)
            nameToBaseName[myName] = basename
        baseNameToAliasList[basename] = tmpAliasList
    if len(parts) > 2: # in the third field there is a potential alias for IR (abbreviation of a US state)
      basename = parts[0]
      if basename in baseNameToAliasForIR:
        if not basename in nameToBaseName:
          nameToBaseName[basename] = basename
        myName = parts[2]
        if not myName in nameToBaseName and not myName in baseNameToAliasList[basename]:
          baseNameToAliasForIR[basename].append(myName)
      else:
        nameToBaseName[basename] = basename
        tmpAliasList = [basename]
        myName = parts[2]
        if not myName in nameToBaseName: # skip entries which have already been appended
          tmpAliasList.append(myName)
          nameToBaseName[myName] = basename
        baseNameToAliasForIR[basename] = tmpAliasList
  f.close()

  cities = {}
  locationRelationFile = "data/freebase.locationRelation"
  f = io.open(locationRelationFile, 'r', encoding="utf-8")
  for line in f:
    line = line.strip()
    parts = (" " + line + " ").split(' :: ')
    city = parts[0].strip()
    state = parts[1].strip()
    country = parts[2].strip()
    if not string.lower(country) in countries:
      countries[string.lower(country)] = 1
    if not string.lower(state) in states:
      states[string.lower(state)] = 1
    if not string.lower(city) in cities:
      cities[string.lower(city)] = 1
  f.close()

  return [countries, states, baseNameToAliasList, nameToBaseName, baseNameToAliasForIR, cities]

def getNicknames():
  name2nicknames = {}
  toDelete = []
  filename = "data/nicknames.cleaned"
  f = io.open(filename, 'r', encoding='utf-8')
  for line in f:
    line = line.strip()
    if re.search(ur'^\#', line):
      continue
    parts = line.split(' : ')
    if len(parts) >= 2: # no nicknames given
      for i in range(0, len(parts)):
        name = parts[i]
        nicknames = parts[0:i] + parts[i+1:]
        if name in name2nicknames:
          toDelete.append(name)
          continue
        name2nicknames[name] = nicknames
  f.close()
  for n in set(toDelete):
    del name2nicknames[n]
  return name2nicknames

def getSlot2Inverse():
  slot2inverse = {}
  slot2inverse["per:parents"] = "per:children"
  slot2inverse["org:employees_or_members"] = "per:employee_or_member_of"
  slot2inverse["gpe:employees_or_members"] = "per:employee_or_member_of"
  slot2inverse["org:students"] = "per:schools_attended"
  slot2inverse["gpe:births_in_city"] = "per:city_of_birth"
  slot2inverse["gpe:births_in_country"] = "per:country_of_birth"
  slot2inverse["gpe:births_in_stateorprovince"] = "per:stateorprovince_of_birth"
  slot2inverse["gpe:births_in_location"] = "per:location_of_birth"
  slot2inverse["gpe:deaths_in_city"] = "per:city_of_death"
  slot2inverse["gpe:deaths_in_country"] = "per:country_of_death"
  slot2inverse["gpe:deaths_in_stateorprovince"] = "per:stateorprovince_of_death"
  slot2inverse["gpe:deaths_in_location"] = "per:location_of_death"
  slot2inverse["gpe:residents_of_city"] = "per:cities_of_residence"
  slot2inverse["gpe:residents_of_country"] = "per:countries_of_residence"
  slot2inverse["gpe:residents_of_stateorprovince"] = "per:statesorprovinces_of_residence"
  slot2inverse["gpe:residents_of_location"] = "per:locations_of_residence"
  slot2inverse["per:holds_shares_in"] = "org:shareholders"
  slot2inverse["org:holds_shares_in"] = "org:shareholders"
  slot2inverse["gpe:holds_shares_in"] = "org:shareholders"
  slot2inverse["per:organizations_founded"] = "org:founded_by"
  slot2inverse["org:organizations_founded"] = "org:founded_by"
  slot2inverse["gpe:organizations_founded"] = "org:founded_by"
  slot2inverse["per:top_member_employee_of"] = "org:top_members_employees"
  slot2inverse["org:member_of"] = "org:members"
  slot2inverse["gpe:member_of"] = "org:members"
  slot2inverse["org:subsidiaries"] = "org:parents"
  slot2inverse["gpe:subsidiaries"] = "org:parents"
  slot2inverse["gpe:headquarters_in_city"] = "org:city_of_headquarters"
  slot2inverse["gpe:headquarters_in_country"] = "org:country_of_headquarters"
  slot2inverse["gpe:headquarters_in_stateorprovince"] = "org:stateorprovince_of_headquarters"
  slot2inverse["gpe:headquarters_in_location"] = "org:location_of_headquarters"
  return slot2inverse

