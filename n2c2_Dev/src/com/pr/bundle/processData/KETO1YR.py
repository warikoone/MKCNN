'''
Created on Apr 26, 2018

@author: neha
'''
import os
import sys
import re
import json
import nltk
import numpy as np
from xml.dom import minidom
import xml.etree.ElementTree as xml
from sklearn.cross_validation import train_test_split

MODEL_VARIABLE = {2:'ADVANCED-CAD',4:'ASP-FOR-MI',6:'DIETSUPP-2MOS',8:'ENGLISH',10:'KETO-1YR',12:'MAKES-DECISIONS'}
engStopwords = set(nltk.corpus.stopwords.words('english'))
BASIC_FILTERS = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n'
TRIGGER_FEATURE = []
TRIGGER_ATTRIBUTE = ""
COSINE_SIMILARITY_THRESHOLD = 0.1
COSINE_DITRIBUTION_THRESHOLD = 98.5
EMBEDDING_FILE = "/home/iasl/Disk_R/Bio_NLP/N2C2_Task/Data/embedding/GoogleNews-vectors-negative300.bin"
EMBEDDING_DIM = 300
EMBEDDING_MATRIX=np.zeros((1,EMBEDDING_DIM))
VARIABLE_ATTRIBUTE_THRESHOLD = 0.0
MAX_SEQUENCE_LENGTH = [0]
SENTENCE_TRIM_LIMIT = 7
category_index = {"not met":0, "met":1}
XMLTEXT = {}

def openConfigurationFile(configFile,jsonVariable):
    
    jsonVariableValue = None
    with open(configFile, "r") as json_file:
        data = json.load(json_file)
        jsonVariableValue = data[jsonVariable]
        json_file.close()
        
    if jsonVariableValue is not None:
        return(jsonVariableValue)
    else:
        print("\n\t Variable load failure")
        sys.exit()

def populateAttributeStatus(xmlSubTree,keyAttributes):
    
    attributeExistStatus = {}
    for attributeTag in xmlSubTree:
        #identifiedKeyset = list(filter(lambda keyTerm : attributeTag.find(keyTerm), keyAttributes))
        keyElements = list(map(lambda identifiedTerm:attributeTag.getElementsByTagName(identifiedTerm),keyAttributes))
        for currentElement in keyElements:
            keyTermList = list((filter(lambda keyPattern : re.search(str(keyPattern).strip(),str(currentElement),flags=re.RegexFlag.IGNORECASE),keyAttributes)))
            keyTermList = list(filter(None,keyTermList))
            if len(keyTermList)>0:
                keyTerm = keyTermList[0]
                keyValueList = list(map(lambda elementSpan : elementSpan.getAttribute('met'),currentElement))
                attributeExistStatus[keyTerm] = keyValueList[0]
    
    if len(attributeExistStatus)==0:
        for identifiedTerm in keyAttributes:
            attributeExistStatus.update({identifiedTerm:'na'})
    #print(">>>",attributeExistStatus)
    return(attributeExistStatus)

def updateOutputXml(predictOutput,keyAttributes,XMLTEXT):
    
    global TRIGGER_ATTRIBUTE
    predictDataPath = openConfigurationFile(configFile,"predictDataPath")
    for fileId in predictOutput:
        decoyDictionary={}
        fileAddress = "".join([predictDataPath,"/",fileId])
        if (os.path.isfile(fileAddress)):
            doc = minidom.parse(fileAddress)
            xmlSubTree = doc.getElementsByTagName("TAGS")
            decoyElements = []
            for attributeTag in xmlSubTree:
                decoyElements = list(map(lambda identifiedTerm:{identifiedTerm:attributeTag.getElementsByTagName(identifiedTerm)[0].getAttribute('met')},keyAttributes))
                
            for listValues in decoyElements:
                decoyDictionary.update(listValues)
                
        root = xml.Element('PatientMatching')
        textNode = xml.SubElement(root,'TEXT')
        textNode.text = XMLTEXT.get(fileId)
        attributeNode = xml.SubElement(root,'TAGS')
        for attributeValue in keyAttributes:
            tag = attributeValue
            attrStatus = ""
            if len(decoyDictionary) > 0:
                attrStatus = decoyDictionary.get(tag)
            if tag == TRIGGER_ATTRIBUTE:
                attrStatus ='met'
                if predictOutput[fileId] == 0:
                    attrStatus ='not met'
            xml.SubElement(attributeNode,tag, {'met':attrStatus})
            
        completeTree = xml.ElementTree(root)
        with open(fileAddress, "wb") as bufferWriter:
            completeTree.write(bufferWriter)
        bufferWriter.close()

''' set config path '''
path = os.path.dirname(sys.argv[0])
tokenMatcher = re.search(".*n2c2_Dev\/", path)
if tokenMatcher:
    configFile = tokenMatcher.group(0)
    configFile="".join([configFile,"config.json"])
    #print(configFile)
            
''' open the config file '''
trainDataPath = openConfigurationFile(configFile,"trainDataPath")
ctAttributes = openConfigurationFile(configFile,"ctAttributes")
ctAttributes = list(re.split('\,',ctAttributes,flags=re.RegexFlag.IGNORECASE))


print("\n\t>> Select model no from dictionary listing::", MODEL_VARIABLE)
loopCounter = 0
while loopCounter == 0:
    keyValue = int(input("Enter key value::"))
    if keyValue in list(MODEL_VARIABLE.keys()):
        loopCounter = 1
        print("\n Requested model>>",MODEL_VARIABLE[int(keyValue)])
        TRIGGER_ATTRIBUTE = str(MODEL_VARIABLE[int(keyValue)])
        #tempTriggerFeatures = openConfigurationFile(configFile,TRIGGER_ATTRIBUTE)
        #TRIGGER_FEATURE = list(re.split('\,',tempTriggerFeatures,flags=re.RegexFlag.IGNORECASE))
        TRIGGER_FEATURE = openConfigurationFile(configFile,TRIGGER_ATTRIBUTE)
        
    else:
        print("\n Invalid model requested, please try again")
        
listedPatientRecords = os.listdir(trainDataPath)
listedPatientRecords = sorted(listedPatientRecords)
#trainRecords, testRecords, trainSummary, testSummary = train_test_split(listedPatientRecords,listedPatientRecords, test_size=0.25, random_state=0)
#trainRecords, testRecords, trainSummary, testSummary = map(lambda recordList : sorted(recordList),list([trainRecords, testRecords, trainSummary, testSummary]))

testRecords = []
testSummary = []
fieldIndex=0
trainingFiles = []

for filename in list(listedPatientRecords):
    
    fieldIndex += 1
    fileAddress = "".join([trainDataPath,"/",filename])
    doc = minidom.parse(fileAddress)
    descriptionText = doc.getElementsByTagName("TEXT")[0]
    visitRecord = descriptionText.firstChild.data
    XMLTEXT.update({filename:visitRecord})
    decisionTree = doc.getElementsByTagName("TAGS")
    attributeExistStatus = populateAttributeStatus(decisionTree,ctAttributes)
    
    if attributeExistStatus[TRIGGER_ATTRIBUTE] == 'na':
        testRecords.append(filename)
        testSummary.append(filename)
    else:
        trainingFiles.append(filename)

predictOutput={} 
for fileId in testRecords:
    predictOutput[fileId] = 0

updateOutputXml(predictOutput,ctAttributes,XMLTEXT)
