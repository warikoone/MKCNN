'''
Created on Apr 25, 2018

@author: neha
'''
import os
import sys
import re
import json
import nltk
import numpy as np
from collections import Counter
from datetime import *
from dateutil import relativedelta
from xml.dom import minidom
import xml.etree.ElementTree as xml
from operator import itemgetter
from nltk.tokenize import word_tokenize
from gensim.models import KeyedVectors
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report,f1_score,precision_score,recall_score
from keras.preprocessing.text import Tokenizer
#from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Conv1D, GlobalMaxPool1D, Flatten, Embedding
from keras.layers import Dense, Input, Dropout,Activation, MaxPool1D
from keras.layers.merge import Concatenate
from keras.optimizers import Adam
from fileinput import filename
from tensorflow import set_random_seed
from numpy.random import seed

MODEL_VARIABLE = {1:'ABDOMINAL',3:'ALCOHOL-ABUSE',5:'CREATININE',7:'DRUG-ABUSE',9:'HBA1C',11:'MAJOR-DIABETES',13:'MI-6MOS'}
engStopwords = set(nltk.corpus.stopwords.words('english'))
BASIC_FILTERS = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n'
TRIGGER_FEATURE = []
TRIGGER_ATTRIBUTE = ""
COSINE_SIMILARITY_THRESHOLD = 0.1
COSINE_DITRIBUTION_THRESHOLD = 98.5
EMBEDDING_FILE = "/home/iasl/Disk_R/Bio_NLP/N2C2_Task/Data/embedding/GoogleNews-vectors-negative300.bin"
EMBEDDING_DIM = 300
VARIABLE_ATTRIBUTE_THRESHOLD = 0.0
MAX_SEQUENCE_LENGTH = [0]
SENTENCE_TRIM_LIMIT = 4
category_index = {"not met":0, "met":1}
wordIDFDict={}
BALANCE_BIAS = 0
XMLTEXT = {}

''' Initialize embedding matrix '''
PRETRAIN_WORD2VEC = KeyedVectors.load_word2vec_format(EMBEDDING_FILE,binary=True)

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
        
def populateArray(currArray,appendArray):
    
    if currArray.shape[0] == 0:
        currArray = appendArray
    else:
        currArray = np.insert(currArray, currArray.shape[0], appendArray, 0)
    return(currArray)  


def performSentenceSplit(sentenceBundle):
    tokenizer = nltk.data.load('nltk:tokenizers/punkt/english.pickle')
    bundledSent = tokenizer.tokenize(sentenceBundle)
    return(bundledSent)

def performWordSplit(sentence):
    return(word_tokenize(sentence))

def removeNewlines(sentence,pattern):
    
    '''
    sentence = re.split(pattern, sentence,flags=re.RegexFlag.IGNORECASE)
    sentence = list(str(eachSentence).strip() for eachSentence in sentence)
    sentence = list(filter(lambda eachSentence : str(eachSentence)!='', sentence))
    sentence = list(map(lambda eachSentence : re.sub('\t', '', eachSentence,flags=re.RegexFlag.IGNORECASE),sentence))
    '''
    sentence = re.sub(pattern, ' ', sentence)
    sentence = sentence.strip()
    sentence = re.sub('\t', '', sentence,flags=re.RegexFlag.IGNORECASE)
    return(sentence)

def removeStopWords(sentence):

    #sentence = removeNewlines(sentence)
    tokenizedSentence = word_tokenize(sentence)
    filteredSentence = filter(lambda word: word not in engStopwords, tokenizedSentence)
    filteredSentence = ' '.join(filteredSentence)
    return(filteredSentence)
              
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

def verifyForAbdomenMention(sentence):
    
    sentence = re.sub('\W*ABD\W*',' abdomen ',sentence,flags=re.RegexFlag.IGNORECASE).strip()
    sentence = re.sub('\W*PAP\W*',' papsmear ',sentence,flags=re.RegexFlag.IGNORECASE).strip()
    return(sentence)

def extractVisitRecords(recordBatch):
    splitBatchRecords = re.split("\*+", recordBatch)
    count=0;
    tokenIndexMapper = []
    sentenceRecord = []
    splitBatchRecords.reverse()
    for splitSubRecords in splitBatchRecords:
        if not (str(splitSubRecords).isspace()):
            #print(">>",count)
            #print(">>>",splitSubRecords)
            bundledSentence = performSentenceSplit(splitSubRecords)
            for splitSentence in bundledSentence:
                #print("new >>>",splitSentence)
                splitSentence = removeNewlines(splitSentence,'\n')
                splitSentence = removeStopWords(splitSentence)
                splitSentence = verifyForAbdomenMention(splitSentence)
                ''' dictionary data'''
                tokenIndexMapper.append(splitSentence)
                ''' sentence record'''
                #sentenceRecord.append(removeStopWords(splitSentence))
                sentenceRecord.append(splitSentence)
                #break
            count += 1
    return(tokenIndexMapper,sentenceRecord)    

''' generate tokens ''' 
def indexRecordTokens(recordTokens):
    
    global BASIC_FILTERS
    initializeRecordTokenizer = Tokenizer(filters= BASIC_FILTERS , split=' ',lower=True)
    initializeRecordTokenizer.fit_on_texts(recordTokens)
    return(initializeRecordTokenizer)    

def calculatePreTrainedWordSimilarity(wordIndexMapper,triggerFeature):
    
    global COSINE_SIMILARITY_THRESHOLD, COSINE_DITRIBUTION_THRESHOLD, PRETRAIN_WORD2VEC
    similarityDictionary = {}
    for word in wordIndexMapper.keys():
        if word in PRETRAIN_WORD2VEC.vocab:
            cosineSimilarity = np.around(PRETRAIN_WORD2VEC.similarity(triggerFeature,word),decimals=3)
            if cosineSimilarity > COSINE_SIMILARITY_THRESHOLD:
                similarityDictionary[word] = cosineSimilarity
    #print("\n>>",similarityDictionary)            
    if len(similarityDictionary) > 0:
        similarityDictionary = dict(sorted(similarityDictionary.items(),key=itemgetter(1),reverse=True))
        functionalCutOff = np.around(np.percentile(
                                    list(similarityDictionary.values()),COSINE_DITRIBUTION_THRESHOLD),decimals=3)
        similarityDictionary = dict(filter((
            lambda tokenItem : (tokenItem[1] >= functionalCutOff)),similarityDictionary.items()))
    return(similarityDictionary)
   

def screenForTriggerFeatures(wordIndexMapper,dictionaryList,initialTriggerFeature,triggerIndex):
    
    global TRIGGER_FEATURE
    wordList=[]
    triggerFeature = initialTriggerFeature
    similarityDictionary = calculatePreTrainedWordSimilarity(wordIndexMapper,triggerFeature)
    dictionaryList.append(similarityDictionary)
    if len(similarityDictionary) > 0:
        #if(triggerIndex) == 0:
        for triggerWord in similarityDictionary.keys():
            if triggerWord != TRIGGER_FEATURE[triggerIndex]:
                ''' restrict context branching to first order trigger word similarity vector '''
                if triggerFeature == TRIGGER_FEATURE[triggerIndex] :
                    wordList.extend(screenForTriggerFeatures(wordIndexMapper, dictionaryList, triggerWord,triggerIndex))
                else:
                    break
        wordList.extend(list(similarityDictionary.keys()))
    return(wordList)

def mixedCharacterTokenizer(token):
    
    specialCharIter = re.finditer('\W+', token, flags=re.RegexFlag.IGNORECASE)
    specialCharcterExists = False
    tokenArray=[]
    startIndex = 0
    for charIter in specialCharIter:
        specialCharcterExists = True
        endIndex = charIter.start()
        if startIndex == endIndex:
            tokenArray.append(charIter.group())
        elif startIndex!=endIndex:
            tokenArray.append(token[startIndex:endIndex])
            tokenArray.append(charIter.group())
        startIndex = charIter.end()
        
    if specialCharcterExists:
        ''' special character tokens '''
        endIndex = len(token)
        if startIndex != endIndex:
            tokenArray.append(token[startIndex:endIndex])
        #print("Special Character InitialArray>>",tokenArray)
    else:
        ''' without special character tokens '''
        tokenArray = [token]
        #print("Non-Special Character InitialArray>>",tokenArray)
    return(tokenArray)
        
def buildMatternPattern(rawPattern,startAppend,terminalAppend):
    
    patternTerm = [startAppend]
    characterPattern = list(rawPattern)
    for charTerm in characterPattern:
        if re.match('\s+', charTerm, flags=re.RegexFlag.IGNORECASE):
            charTerm = ' '
        elif re.match('\W+', charTerm, flags=re.RegexFlag.IGNORECASE):
            charTerm = '\\'+charTerm
        patternTerm.append(charTerm)
    patternTerm.append(terminalAppend)
    #print("final pattern>>",patternTerm)
    completePattern = str(''.join(word for word in patternTerm))
    #print("final pattern>>",patternTerm,"\n\t>>",completePattern)
    return(completePattern)

def checkForSurgicalProcedures(sentenceRecord):
    
    decoyDictionaryList=[]
    suffixList = ['ctomy','otomy','rrapy','ostomy','spplant']
    #prefixList = ['surg']
    otherSymp = ['obstruction','abortion','abortions','transplant','transplants']
    termFlag = False
    for sentence in sentenceRecord:
        for patternToken in suffixList:
            patternToken = buildMatternPattern(patternToken,'\W*\w+','\W*')
            #selectValue = list(filter(lambda triggerTerm: re.fullmatch(patternToken, triggerTerm.strip(), flags=re.RegexFlag.IGNORECASE),decoyDict.keys()))
            patternMatcher = re.finditer(patternToken, sentence, flags=re.RegexFlag.IGNORECASE)
            for matchValue in patternMatcher:
                termFlag = True
                dumpString = str(re.sub('\W*', '', matchValue.group().strip())).lower()
                decoyDictionaryList.append(dumpString)
    
    for sentence in sentenceRecord:
        for patternToken in otherSymp:
            patternToken = buildMatternPattern(patternToken,'\W*','\W*')
            patternMatcher = re.finditer(patternToken, sentence, flags=re.RegexFlag.IGNORECASE)
            for matchValue in patternMatcher:
                termFlag = True
                dumpString = str(re.sub('\W*', '', matchValue.group().strip())).lower()
                decoyDictionaryList.append(dumpString)
    '''
    if not termFlag:
            for patternToken in prefixList:
                patternToken = buildMatternPattern(patternToken,'\W*','\w+\W*')
                patternMatcher = re.finditer(patternToken, sentence, flags=re.RegexFlag.IGNORECASE)
                for matchValue in patternMatcher:
                    dumpString = str(re.sub('\W*', '', matchValue.group().strip())).lower()
                    decoyDictionaryList.append(dumpString)'''
    
    return(decoyDictionaryList)

def existsBranchedRelation(sourceList, targetList, sentimentList):
    
    relation = False
    for word in sourceList:
        if word in targetList:
            sentimentList.append(word)
            relation = True
            #print("relatd term>>",word)
        
    return(relation)  

def verfiySentimentBranching(dictionaryList,sourceSentiment,seedIndex,triggerDominatedSentiment):
    
    if len(dictionaryList) > 1:
        ''' compare tokens amongst themselves to find relational relevance'''
        for index in range(1,len(dictionaryList)):
            branchedSentiment = list(dictionaryList[index].keys())
            leaderTerm = branchedSentiment[seedIndex]
            indexLeaderTerm = sourceSentiment.index(leaderTerm)
            #print("\n leaderTerm>>",leaderTerm)
            #print("\n branchedSentiment>>",branchedSentiment)
            spliceIndex = index-1
            ''' if the branched array source term is the root of relational association in current document'''
            if indexLeaderTerm == 0:
                if len(sourceSentiment) == 1:
                    triggerDominatedSentiment.append(leaderTerm)
                else:
                    ''' look up in succeeding terms '''
                    sourceSubSentiment = list(sourceSentiment[(indexLeaderTerm+1):len(sourceSentiment)])
                    #print("\n succeeding sourceSubSentiment>>",sourceSubSentiment)
                    if(existsBranchedRelation(sourceSubSentiment, branchedSentiment, triggerDominatedSentiment)):
                        triggerDominatedSentiment.append(leaderTerm)
            elif indexLeaderTerm >=1 :
                ''' look up in preceding terms '''
                sourceSubSentiment = list(sourceSentiment[0:indexLeaderTerm])
                #print("\n preceding sourceSubSentiment>>",sourceSubSentiment)
                if(existsBranchedRelation(sourceSubSentiment, branchedSentiment, triggerDominatedSentiment)):
                    triggerDominatedSentiment.append(leaderTerm)
            #print("selection>>",tier1TriggerDominatedSentiment)

        
''' isolate candidate instances from document record '''
def isolateCrCandidateInstances(currentDocumentRecord, triggerDominatedSentiment, recordCandidateInstances, modelId):
    
    global SENTENCE_TRIM_LIMIT,MAX_SEQUENCE_LENGTH,VOCAB_WORDSIZE,COSINE_DITRIBUTION_THRESHOLD
    for sentence in currentDocumentRecord:
        #print("\t original sent**",sentence)
        sentenceTokens = list(performWordSplit(sentence))
        startIndexList=[];endIndexList=[]
        for wordPattern in triggerDominatedSentiment:
            for index, token in enumerate(sentenceTokens):
                tokenArray = mixedCharacterTokenizer(str(token).strip()) 
                #print("token>>",str(token),"tokenarray>",tokenArray)
                wordPattern = buildMatternPattern(wordPattern,'','')
                #print("\n\t wordPattern>>",wordPattern)
                wordMatcherArray = list(filter(lambda word : re.fullmatch(wordPattern.strip(), word.strip(), flags=re.RegexFlag.IGNORECASE), tokenArray))
                #wordMatcher = re.match(wordPattern.strip(), str(token).strip(), flags=re.RegexFlag.IGNORECASE)
                if len(wordMatcherArray)>0:
                    #print("\n matcher>>",wordMatcherArray)
                    currIndex = (index-SENTENCE_TRIM_LIMIT)
                    #print("\n 1s currIndex>>",currIndex)
                    if currIndex < 0:
                        currIndex = 0
                    startIndexList.append(currIndex)
                    currIndex = (index+SENTENCE_TRIM_LIMIT)
                    #print("\n 2n currIndex>>",currIndex)
                    if currIndex > (len(sentenceTokens)-1):
                        currIndex = len(sentenceTokens)-1
                    endIndexList.append(currIndex)
        if (len(startIndexList)>0) and (len(endIndexList)>0):
            #print("sen>>",sentenceTokens)
            #print("start aftr>>",startIndexList)
            #print("end aftr>>",endIndexList)
            startIndex = min(startIndexList)
            endIndex = max(endIndexList)
            #print("limits\t >>",startIndex,'\t>>',endIndex)
            #print("\t >>",sentenceTokens[startIndex:endIndex+1])
            sentenceTokens = list(filter(lambda token : re.match('[^\W]', token.strip(), flags=re.RegexFlag.IGNORECASE), sentenceTokens[startIndex:endIndex+1]))
            '''
            decoyToken =[]
            for token in sentenceTokens[startIndex:endIndex+1]:
                decoyToken.extend(mixedCharacterTokenizer(token.strip()))
            sentenceTokens = decoyToken
            '''
            #sentenceTokens = sentenceTokens[startIndex:endIndex+1]
            if(len(sentenceTokens) > MAX_SEQUENCE_LENGTH[modelId]):
                MAX_SEQUENCE_LENGTH[modelId] = len(sentenceTokens)
            sentence = ' '.join(word.strip() for word in sentenceTokens)
            #sentence = re.sub('\s*aspirin\s*', '', sentence, flags=re.RegexFlag.IGNORECASE)
            #print("final token>>",sentence)
            if len(sentence) > 0:
                recordCandidateInstances.append(sentence.strip())

def vectorizeDecisionStatus(recordCandidateInstances,statusValue):
    
    vectorizedStatus = []
    for index in range(len(recordCandidateInstances)):
        statusList = np.array([0.0,0.0])
        if statusValue != "na":
            statusList = to_categorical(category_index[statusValue], len(category_index))
        vectorizedStatus.insert(index,statusList)
    return(vectorizedStatus)

def checkForTriggerWord(triggerDominatedSentiment,sentence):
    
    retStatus = False
    retTerm = []
    for triggerTerm in triggerDominatedSentiment:
        triggerTerm = buildMatternPattern(triggerTerm,'\W*\w*','\W*')
        tokenMatcher = re.finditer(triggerTerm, sentence, flags=re.RegexFlag.IGNORECASE)
        for matchCase in tokenMatcher:
            retStatus = True
            retTerm.append(matchCase.group())
    return(retStatus,retTerm)

def printRecords(candidateRecordsDictionary,summary):
    
    for recordName in candidateRecordsDictionary.keys():
        print("\n File::",recordName)
        recordList = list(candidateRecordsDictionary[recordName])
        for sentence in recordList:
            print("\t::",sentence,"\t>>",np.sum(np.array(summary[recordName]),0))
            
def generateTokenBasedFrequency(candidateRecordsDictionary):
    
    global wordIDFDict
    for filename in candidateRecordsDictionary.keys():
        decoyList = candidateRecordsDictionary.get(filename)
        decoyTokens = []
        for sentence in decoyList:
            if ((sentence is not 'history of abdomen surgery') and (sentence is not 'no abdomen surgery')):
                sentenceTokens = performWordSplit(sentence)
                sentenceTokens = list(filter(lambda token : not re.fullmatch('\d+|\W+', token, flags=re.RegexFlag.IGNORECASE),sentenceTokens))
                decoyTokens.extend(sentenceTokens)
        if len(decoyTokens) > 0:
            decoyTokens = list(np.unique(decoyTokens))
            for token in decoyTokens:
                token = str(token).lower()
                count=0
                if token in wordIDFDict.keys():
                    count = wordIDFDict.get(token)
                count += 1
                wordIDFDict.update({token:count})
    
    #print("1>>>",wordIDFDict)
    docSize = len(tier1CandidateRecordsDictionary)
    for item in wordIDFDict.items():
        if item[1] != 1:
            idfScore = int(np.rint(np.log(docSize/item[1])))
            wordIDFDict.update({item[0]:idfScore})
        elif item[1] == 1:
            wordIDFDict.update({item[0]:0})
    
    #print("2>>>",wordIDFDict)
    wordIDFDict = dict(sorted(wordIDFDict.items(),key=itemgetter(1),reverse=True))
    wordIDFDict = dict(filter(lambda wordScore : wordScore[1]>0, wordIDFDict.items()))
    #print("3>>>",wordIDFDict)
    return()                        
            
def defineSampleRandomizationSize(positiveCases,negativeCases):
    
    segmentLength = 1000
    leaderCase = positiveCases
    trailerCase = negativeCases 
    if (len(negativeCases) < len(positiveCases)):
        if ((len(negativeCases) > 0) and (len(negativeCases) < segmentLength)):
            segmentLength = len(negativeCases)
        elif (len(negativeCases) == 0):
            segmentLength = 0
            
    else:
        leaderCase = negativeCases
        trailerCase = positiveCases
        if ((len(positiveCases) > 0) and (len(positiveCases) < segmentLength)):
            segmentLength = len(positiveCases)
        elif (len(negativeCases) == 0):
            segmentLength = 0
            
    iterateLength = int(np.rint(len(leaderCase)/segmentLength))
        
    return(leaderCase,trailerCase,iterateLength,segmentLength)

def retreiveSeenPreTrainedEmbedding(embeddingMatrix, index, token):
    
    global EMBEDDING_DIM, PRETRAIN_WORD2VEC,wordIDFDict
    multiplicationFactor = float(100)
    if token in wordIDFDict.keys():
        multiplicationFactor = float(1/wordIDFDict.get(token))
    #print("m fac>>>",multiplicationFactor)
    #print("b4>>",PRETRAIN_WORD2VEC.word_vec(token)[0:EMBEDDING_DIM])
    embeddingMatrix[index] = (PRETRAIN_WORD2VEC.word_vec(token)[0:EMBEDDING_DIM]*multiplicationFactor)
    #print("aftr>>",embeddingMatrix[index])
    #exit()
    '''
    global EMBEDDING_DIM, PRETRAIN_WORD2VEC
    embeddingMatrix[index] = PRETRAIN_WORD2VEC.word_vec(token)[0:EMBEDDING_DIM]
    '''
    
def buildPattern(currPatternMap):
    
    pattern = ''.join(str(currPattern[1]) for currPattern in currPatternMap.items())
    return(pattern)

def unseenExistsInPreTrainedEmbedding(pattern):
    
    global PRETRAIN_WORD2VEC
    if pattern in PRETRAIN_WORD2VEC.vocab:
        return(True,pattern)
    else:
        return(False,'')
    
def checkForDataType(tokenGroup):
    
    groupTokenType = []
    groupTokenSubstitutePattern = []
    patternType ={1:'\d+',2:'[a-zA-Z]+'}
    for index,wordGroup in enumerate(tokenGroup):
        tokenType = {}
        tokenSubstitutePattern={}
        for patternItem in patternType.items():
            iterPattern = re.finditer(patternItem[1], wordGroup, flags=re.RegexFlag.IGNORECASE)
            for match in iterPattern:
                tokenType[match.start()] = patternItem[0]
                ''' handle data pattern conversion'''
                if patternItem[0] == 1:
                    tokenSubstitutePattern[match.start()] = re.sub('\d', '#', match.group())
                else:
                    tokenSubstitutePattern[match.start()] = match.group()
        
        tokenType = dict(sorted(tokenType.items(),key=itemgetter(0)))
        tokenSubstitutePattern = dict(sorted(tokenSubstitutePattern.items(),key=itemgetter(0)))
        
        groupTokenType.insert(index, tokenType)
        groupTokenSubstitutePattern.insert(index, tokenSubstitutePattern)
    
    return(groupTokenType, groupTokenSubstitutePattern)

def existsGroupEmbedding(tokenArray,characterTokens):
    
    pattern=''
    for index in range(len(tokenArray)):
        currPatternMap = characterTokens[1][index] 
        if len(currPatternMap) > 0:
            ''' appending non special character pattern '''
            pattern = pattern + buildPattern(currPatternMap)
        else:
            ''' appending special character pattern '''
            pattern = pattern+ str(tokenArray[index])
    
    #print("\n group PATTERN>>",pattern)
    return(unseenExistsInPreTrainedEmbedding(pattern))

def generateWordHypernym(pattern):

    hypernymList = []    
    patternSynonyms = nltk.corpus.wordnet.synsets(pattern)
    for patternSynonym in patternSynonyms:
        wordLemma = patternSynonym.lemma_names()[0]
        if(re.match(pattern, wordLemma, flags = re.RegexFlag.IGNORECASE)):
            for hypernymWord in patternSynonym.hypernyms(): 
                hypernymWord = re.sub('\_', '-',hypernymWord.lemma_names()[0])
                hypernymList.append(hypernymWord.strip())
                
    return(hypernymList)    

def assembleComponentEmbedding(characterTokens):
    
    convergeEmbeddingMatrix = np.ones((1,EMBEDDING_DIM))
    for tokenDict in characterTokens[1]:
        if len(tokenDict) > 0:
            for itemValue in tokenDict.items():
                pattern = itemValue[1]
                embeddingStatus = unseenExistsInPreTrainedEmbedding(pattern)
                tempEmbeddingMatrix = np.zeros((1,EMBEDDING_DIM))
                if(embeddingStatus[0]):
                    #print("\n\t assembleComponentEmbedding() ~ embedding retrieved for>>", embeddingStatus[1])
                    retreiveSeenPreTrainedEmbedding(tempEmbeddingMatrix, 0, embeddingStatus[1])
                else:
                    #print("\n\t assembleComponentEmbedding() ~ embedding absent for>>", pattern)
                    ''' look up for hypernyms in Wordnet'''
                    hypernymList = generateWordHypernym(pattern)
                    if (len(hypernymList)>0):
                        ''' look up again in the pre-trained vector '''
                        for hypernymWord in hypernymList:
                            #print("\t Hypernym word>>",hypernymWord)
                            retreiveUnseenPreTrainedEmbedding(tempEmbeddingMatrix,0,hypernymWord)
                            ''' select singular hypernym with reference in w2v '''
                            if np.sum(tempEmbeddingMatrix[[0]],axis=1) != 0:
                                break
                    else:
                        ''' issue default embedding'''
                        #print("\n default embed word>>",pattern)
                        tempEmbeddingMatrix = np.full((1,EMBEDDING_DIM),0.1)
                            
                #print(" b4 current coverage>>>",convergeEmbeddingMatrix,"\t tempEmbed>>",tempEmbeddingMatrix)
                convergeEmbeddingMatrix = np.multiply(np.array(convergeEmbeddingMatrix),np.array(tempEmbeddingMatrix))
                #print(" aftr current coverage>>>",convergeEmbeddingMatrix)
                           
    #print("FINAL ADJUSTED>>>",convergeEmbeddingMatrix)
    return(convergeEmbeddingMatrix)
    
def retreiveUnseenPreTrainedEmbedding(embeddingMatrix, index, token):

    tokenArray = mixedCharacterTokenizer(str(token))
             
    tokenCharSubset = checkForDataType(tokenArray)
    #print("tokenCharSubset>>>",tokenCharSubset[0],"\t>>",tokenCharSubset[1])
    embeddingStatus = existsGroupEmbedding(tokenArray,tokenCharSubset)
    if(embeddingStatus[0]):
        #print("\n\t embedding retrieved for>>", embeddingStatus[1])
        retreiveSeenPreTrainedEmbedding(embeddingMatrix, index, embeddingStatus[1])
    else:
        #print("\n\t embedding absent for>>", embeddingStatus[1])
        embeddingMatrix[index] = assembleComponentEmbedding(tokenCharSubset)                                     

def assimilatePreTrainedEmbeddings(sentenceTokens, recordMatrix,modelId):
    
    global EMBEDDING_DIM, PRETRAIN_WORD2VEC,MAX_SEQUENCE_LENGTH
    embeddingMatrix = np.zeros((MAX_SEQUENCE_LENGTH[modelId],EMBEDDING_DIM))
    for index,token in enumerate(sentenceTokens):
        if token in PRETRAIN_WORD2VEC.vocab:
            retreiveSeenPreTrainedEmbedding(embeddingMatrix, index, token)
        else:
            retreiveUnseenPreTrainedEmbedding(embeddingMatrix, index, token)
    
    recordMatrix = populateArray(recordMatrix, embeddingMatrix)
    return(recordMatrix)   
      
def findSentenceVectorEmbeddings(candidateRecordsDictionary,referenceRecords, candidateRecordsDecisionDictionary, referenceRecordsSummary,modelId):
    
    global EMBEDDING_DIM
    x_recordDict = {}
    y_summaryDict = {}
    #print("\n ref rec>>",referenceRecords,"\n\t refSum>>",referenceRecordsSummary)
    for fileId in candidateRecordsDictionary.keys():
        x_record = np.array([])
        y_summary = np.array([])
        if ((candidateRecordsDictionary[fileId] is not None) and (fileId in referenceRecords) and (fileId in referenceRecordsSummary)):
            for sentence in list(candidateRecordsDictionary[fileId]):
                x_record = assimilatePreTrainedEmbeddings(performWordSplit(sentence),x_record,modelId)
            #appendArray = np.array(list(candidateRecordsDecisionDictionary[fileId])).reshape(-1,1)
            appendArray = np.array(list(candidateRecordsDecisionDictionary[fileId]))
            #print("append>>",appendArray,"\tshape>>",appendArray.shape,"file>>",fileId)
            if appendArray.shape[1] == 2:
                y_summary = populateArray(y_summary,appendArray)
                
        x_recordDict.update({fileId:x_record})
        y_summaryDict.update({fileId:y_summary})
                
    return(x_recordDict,y_summaryDict)        

def defineCovNetsModel(modelId, filterSize):
    
    global MAX_SEQUENCE_LENGTH, EMBEDDING_DIM
    
    modelInput = Input(shape=(MAX_SEQUENCE_LENGTH[modelId],EMBEDDING_DIM))
    embeddingLayer = modelInput
    #print("\n input layer>>>",embeddingLayer)
    
    convBlocks = []
    for eachFilter in filterSize:
        currFilter = int(np.rint(MAX_SEQUENCE_LENGTH[modelId]/(eachFilter)))
        if currFilter == 0:
            currFilter = 2
        singleConv = Conv1D(filters=currFilter,kernel_size=eachFilter,padding='valid',activation='relu',strides=1)(embeddingLayer)
        print("\n convolution layer>>>",singleConv)
        
        singleConv = MaxPool1D(pool_size = 1)(singleConv)
        print("\n MaxPool1D layer>>>",singleConv)
    
        singleConv = Flatten()(singleConv)
        print("\n Flatten layer>>>",singleConv)
        
        convBlocks.append(singleConv)
            
    tranformLayer = Concatenate()(convBlocks) if len(convBlocks) > 1 else convBlocks[0] 
    
    #conv = Dropout(0.5)(conv)
    tranformLayer = Dense(10,activation='relu')(tranformLayer)
    #print("\n 1st Dense layer>>>",tranformLayer)
    
    modelOutput = Dense(2,activation='sigmoid')(tranformLayer)
    #print("\n 2nd Dense layer>>>",modelOutput)
    
    model = Model(inputs = modelInput, outputs=modelOutput)
    #print("\n model>>>",model)
    
    lrAdam = Adam(lr=0.01,decay=0.001)
    #lrAdam = Adam(lr=0.01)
    model.compile(optimizer=lrAdam, loss='binary_crossentropy',metrics=['accuracy'])
        
        #model.summary()
        
    return(model)

def trannslateMinScoresAsVotes(predictionArray):
    
    votePoolSize = predictionArray.shape[0]
    col0Votes = []
    col1Votes = []
    status = 0
    for rowId in range(votePoolSize):
        if predictionArray[rowId][0] < predictionArray[rowId][1]:
            col1Votes.insert(rowId, 1)
        else:
            col0Votes.insert(rowId, 1)
            
    if (np.sum(np.array(col1Votes),0) > 0):
        status = 1
    else:
        status = 0
    
    return(status)

def translateScoresAsVotes(predictionArray):
    
    votePoolSize = predictionArray.shape[0]
    col0Votes = []
    col1Votes = []
    status = 0
    for rowId in range(votePoolSize):
        if predictionArray[rowId][0] < predictionArray[rowId][1]:
            col1Votes.insert(rowId, 1)
        else:
            col0Votes.insert(rowId, 1)
    if (len(col0Votes) > len(col1Votes)):
        status = 0
    elif (len(col0Votes) < len(col1Votes)):
        status = 1 
    else:
        comparativeScore = (np.sum(predictionArray,0)/predictionArray.shape[0])
        status = 1
        if comparativeScore[0]>comparativeScore[1]:
            status = 0
    
    return(status)

def reorganizeModelBasedVoting(votingList,dataSummary,index,MODEL_SIZE):
    
    decoyVoteDictionary = {}
    if index in range(len(votingList)):
        decoyVoteDictionary = votingList[index]

    for fileId in dataSummary.keys():
        currVoteStatus = dataSummary[fileId]
        decoyVoteList = []
        if fileId in decoyVoteDictionary.keys():
            decoyVoteList = decoyVoteDictionary[fileId]
        decoyVoteList.append(currVoteStatus)
        decoyVoteDictionary.update({fileId:decoyVoteList})
    
    if len(votingList)<MODEL_SIZE:
        votingList.insert(index,decoyVoteDictionary)
    else:
        votingList[index] = decoyVoteDictionary
    return(votingList)  

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

seed(1)
set_random_seed(2)

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
        TRIGGER_FEATURE.append(openConfigurationFile(configFile,TRIGGER_ATTRIBUTE))
        
    else:
        print("\n Invalid model requested, please try again")
        
listedPatientRecords = os.listdir(trainDataPath)
listedPatientRecords = sorted(listedPatientRecords)
#trainRecords, testRecords, trainSummary, testSummary = train_test_split(listedPatientRecords,listedPatientRecords, test_size=0.25, random_state=0)
#trainRecords, testRecords, trainSummary, testSummary = map(lambda recordList : sorted(recordList),list([trainRecords, testRecords, trainSummary, testSummary]))

featureRecordDimension = []
featureSummaryDimension = []
tier1CandidateRecordsDictionary = {}
tier1CandidateRecordsDecisionDictionary = {}
indexMapperDictionary = []
recordDictionary = []
testRecords = []
testSummary = []
fieldIndex=0
trainingFiles = []
posTier1TriggerDominatedSentiment=[]
negTier1TriggerDominatedSentiment=[]
    
for filename in list(listedPatientRecords):
    #if filename in ['102.xml']:
        fieldIndex += 1
        fileAddress = "".join([trainDataPath,"/",filename])
        doc = minidom.parse(fileAddress)
        descriptionText = doc.getElementsByTagName("TEXT")[0]
        visitRecord = descriptionText.firstChild.data
        XMLTEXT.update({filename:visitRecord})
        decisionTree = doc.getElementsByTagName("TAGS")
        attributeExistStatus = populateAttributeStatus(decisionTree,ctAttributes)
        
        tokenIndexMapper,sentenceRecord = extractVisitRecords(visitRecord)
        indexMapperDictionary.extend(tokenIndexMapper)
        recordDictionary.insert(fieldIndex, sentenceRecord)
        
        ''' create document record index '''
        initializeRecordTokenizer = indexRecordTokens(tokenIndexMapper)
        wordIndexMapper = initializeRecordTokenizer.word_index
        
        ''' Isolate feature words from document records'''
        tier1TriggerDominatedSentiment=[]
        tempDecoy = checkForSurgicalProcedures(sentenceRecord)
        tier1TriggerDominatedSentiment.extend(tempDecoy)
        
        if ((tempDecoy is not None) and (filename in listedPatientRecords)):
            if (attributeExistStatus[TRIGGER_ATTRIBUTE] != 'na'):
                if (attributeExistStatus[TRIGGER_ATTRIBUTE] == 'met'):
                    posTier1TriggerDominatedSentiment.extend(tempDecoy)
                else:
                    negTier1TriggerDominatedSentiment.extend(tempDecoy)
                
        #print("\t tier1TriggerDominatedSentiment>>",tier1TriggerDominatedSentiment,"\t>>",filename)
        tier1RecordCandidateInstances=list()
        if len(tier1TriggerDominatedSentiment) > 0:
            triggerIndex=0
            isolateCrCandidateInstances(sentenceRecord, tier1TriggerDominatedSentiment, tier1RecordCandidateInstances,triggerIndex)
            tier1RecordCandidateInstances = list(set(tier1RecordCandidateInstances))
            
        if((len(tier1TriggerDominatedSentiment)==0) or (len(tier1RecordCandidateInstances)==0)):
            if ((filename in listedPatientRecords) and (attributeExistStatus[TRIGGER_ATTRIBUTE] == 'met')):
                tier1RecordCandidateInstances = ['history of abdomen surgery']
            else:
                tier1RecordCandidateInstances = ['no abdomen surgery']
    
        if attributeExistStatus[TRIGGER_ATTRIBUTE] == 'na':
            testRecords.append(filename)
            testSummary.append(filename)
        else:
            trainingFiles.append(filename)
            
        tier1CandidateRecordsDictionary[filename] = tier1RecordCandidateInstances
        tier1CandidateRecordsDecisionDictionary[filename] = vectorizeDecisionStatus(tier1RecordCandidateInstances,attributeExistStatus[TRIGGER_ATTRIBUTE])

        

posTier1TriggerDominatedSentiment = list(set(posTier1TriggerDominatedSentiment))
negTier1TriggerDominatedSentiment = list(set(negTier1TriggerDominatedSentiment))
decoyPosList = []
otherSymp = ['obstruction','abortion','abortions','transplant','transplants']
for relatedTriggerTerm in posTier1TriggerDominatedSentiment:
    
    status = True
    for triggerTerm in negTier1TriggerDominatedSentiment:
        if(re.search(triggerTerm.strip(),relatedTriggerTerm.strip(), flags=re.RegexFlag.IGNORECASE)):
            status = False
            print("common >>>",relatedTriggerTerm)
            break
    if status:
        decoyPosList.append(relatedTriggerTerm)
        
posTier1TriggerDominatedSentiment = decoyPosList

print("\t>>",posTier1TriggerDominatedSentiment)
print("\t>>",negTier1TriggerDominatedSentiment)

keyList = tier1CandidateRecordsDictionary.keys()

removeList=[]
for filename in keyList:
    
    #if filename in ['130.xml']:
        decisionMatrix = np.array(list(tier1CandidateRecordsDecisionDictionary[filename]))
        #print("\t decisionMatrix>>",decisionMatrix,"\t>>",decisionMatrix.shape)
        status = np.sum(decisionMatrix[:,1])
        if ((filename in listedPatientRecords) and (status > 0)):
            tier1RecordCandidateInstances = tier1CandidateRecordsDictionary[filename]
            decoyRecordInstances = []
            tempList=[]
            for index,sentence in enumerate(tier1RecordCandidateInstances):
                positiveStatus,positiveMatch = checkForTriggerWord(posTier1TriggerDominatedSentiment,sentence)
                negativeStatus,negativeMatch = checkForTriggerWord(negTier1TriggerDominatedSentiment,sentence)
                if negativeStatus:
                    if positiveStatus:
                        for matchWord in negativeMatch:
                            sentence = re.sub(matchWord, ' ', sentence,flags=re.RegexFlag.IGNORECASE)
                    else:
                        sentence=None
                if sentence is not None:
                    decoyRecordInstances.append(sentence)
                else:
                    #print("\t>>",index)
                    tempList.append(index)
                    
            if len(tempList)>0:
                decisionMatrix = np.delete(decisionMatrix,tempList,0)
                #print("\t decisionMatrix>>",decisionMatrix,"\t>>",decisionMatrix.shape)
             
            if decisionMatrix.shape[0]!=0:   
                tier1CandidateRecordsDecisionDictionary.update({filename:decisionMatrix})
                tier1CandidateRecordsDictionary.update({filename:decoyRecordInstances})
            else:
                removeList.append(filename)
            
for filename in removeList:
    print(" removed>>",filename)
    #del tier1CandidateRecordsDecisionDictionary[filename]
    #del tier1CandidateRecordsDictionary[filename]
    #listedPatientRecords.remove(filename)
    
#printRecords(tier1CandidateRecordsDictionary,tier1CandidateRecordsDecisionDictionary)

featureRecordDimension.append(tier1CandidateRecordsDictionary)
featureSummaryDimension.append(tier1CandidateRecordsDecisionDictionary)

generateTokenBasedFrequency(tier1CandidateRecordsDictionary)

MODEL_SIZE = len(featureRecordDimension)        

''' data embedding'''
x_dataList = []
y_summaryList = []
for modelIter in range(0,MODEL_SIZE):
    
    candidateRecordsDictionary = dict(featureRecordDimension[modelIter])
    candidateRecordsDecisionDictionary = dict(featureSummaryDimension[modelIter])
    
    print("max seq>>",MAX_SEQUENCE_LENGTH[modelIter])        
    
    x_dataDict,y_summaryDict = findSentenceVectorEmbeddings(candidateRecordsDictionary, listedPatientRecords, candidateRecordsDecisionDictionary, listedPatientRecords,modelIter)
    
    x_dataList.append(x_dataDict)
    y_summaryList.append(y_summaryDict)
   
stratifiedPredictOut = {}
splitSize = int(len(trainingFiles)*0.025)
stratSplit = KFold(n_splits=splitSize)
stratifiedSplitIndex = 0
print(">>",len(trainingFiles))
for trainIndex, validationIndex in stratSplit.split(trainingFiles):
    print("\t split Index>>",stratifiedSplitIndex)
    trainRecords = sorted(list(map(lambda index : trainingFiles[index], trainIndex)))
    trainSummary = trainRecords
    validationRecords = sorted(list(map(lambda index : trainingFiles[index], validationIndex)))
    validationSummary = validationRecords
    
    ''' proportion +ve/-ve training samples'''
    positiveCases = []
    negativeCases = []
    for fileId in featureRecordDimension[0]:
        if fileId in trainRecords:
            ''' not met'''
            if np.sum(np.array(list(dict(featureSummaryDimension[0])[fileId]))[:,1]) == 0:
                negativeCases.append(fileId)
                BALANCE_BIAS = 0
            else:
                positiveCases.append(fileId)
    
    if len(positiveCases) < len(negativeCases):
        BALANCE_BIAS = 1
    
    print("\n\t size of +ve cases>>",len(positiveCases),"\t -ve cases>>",len(negativeCases))
    
    leaderCase,trailerCase,iterateLength,segmentLength = defineSampleRandomizationSize(positiveCases,negativeCases)
    print("\n iterate length>>",iterateLength,"\t segment length>>",segmentLength)   
    
    if segmentLength > 0:
        '''  validation data'''
        x_validationList = []
        y_validationList = []
        for modelIter in range(0,MODEL_SIZE):
            x_dataDict = x_dataList[modelIter]
            y_summaryDict = y_summaryList[modelIter]
            x_validation = np.array([])
            y_validation = np.array([])
            for fileId in validationRecords:
                x_validation = populateArray(x_validation,x_dataDict.get(fileId))
                y_validation = populateArray(y_validation,y_summaryDict.get(fileId))
                
            print("\n x_validation>>",x_validation.shape,"\t y_validation>>",y_validation.shape)
            
            validationSentenceDimension = int(np.rint(x_validation.shape[0]/MAX_SEQUENCE_LENGTH[modelIter]))
            validationSequenceSpan = MAX_SEQUENCE_LENGTH[modelIter]
            validationFeatureDimension = EMBEDDING_DIM 
            x_validation = x_validation.reshape(validationSentenceDimension,validationSequenceSpan,validationFeatureDimension)
            print(" validation tensor shape::",x_validation.shape,"\t y::",y_validation.shape)
            
            x_validationList.append(x_validation)
            y_validationList.append(y_validation)
            
        '''  test data'''
        x_testList = []
        y_testList = []
        for modelIter in range(0,MODEL_SIZE):
            x_dataDict = x_dataList[modelIter]
            y_summaryDict = y_summaryList[modelIter]
            x_test = np.array([])
            y_test = np.array([])
            for fileId in testRecords:
                x_test = populateArray(x_test,x_dataDict.get(fileId))
                y_test = populateArray(y_test,y_summaryDict.get(fileId))
                
            print("\n x_test>>",x_test.shape,"\t y_test>>",y_test.shape)
            
            testSentenceDimension = int(np.rint(x_test.shape[0]/MAX_SEQUENCE_LENGTH[modelIter]))
            testSequenceSpan = MAX_SEQUENCE_LENGTH[modelIter]
            testFeatureDimension = EMBEDDING_DIM 
            x_test = x_test.reshape(testSentenceDimension,testSequenceSpan,testFeatureDimension)
            print(" test tensor shape::",x_test.shape,"\t y::",y_test.shape)
            
            x_testList.append(x_test)
            y_testList.append(y_test)
          
        ''' fit training data on model '''
        modelVotingList = []
        for i in range(1):
            print("\n Iteration range>>",i)
            runIndex=0
            startIndex = runIndex
            while runIndex < iterateLength:
                print("\t INDEX>>",runIndex)
                #caseBalNo = int(np.rint(segmentLength/2))
                #caseBalNo=0
                #endIndex = startIndex+(segmentLength+caseBalNo)
                endIndex = startIndex+(len(leaderCase))
                print("startIndex>>",startIndex,"endIndex>>",endIndex)
                if startIndex >= len(leaderCase):
                    break
                sampleCases = leaderCase[startIndex:endIndex]
                print("size>>",len(sampleCases))
                sampleCases.extend(trailerCase)
                print("trainRecords>>",sampleCases,"trainSummary>>",sampleCases)
            
                scoreDictList=[]
                modelIter = 0
                for modelIter in range(MODEL_SIZE):
                    scoreDict={}
                    x_dataDict = x_dataList[modelIter]
                    y_summaryDict = y_summaryList[modelIter]
                    x_train = np.array([])
                    y_train = np.array([])
                    for fileId in trainRecords:
                        x_train = populateArray(x_train,x_dataDict.get(fileId))
                        y_train = populateArray(y_train,y_summaryDict.get(fileId))
            
                    print(" x_train>>",x_train.shape,"\t y_train>>",y_train.shape)
                    
                    ''' Input Tensor Structure'''
                    trainSentenceDimension = int(np.rint(x_train.shape[0]/MAX_SEQUENCE_LENGTH[modelIter]))
                    trainSequenceSpan = MAX_SEQUENCE_LENGTH[modelIter]
                    trainFeatureDimension = EMBEDDING_DIM
                    
                    x_train = x_train.reshape(trainSentenceDimension,trainSequenceSpan,trainFeatureDimension)
                    print(" training tensor shape::",x_train.shape,"\t y::",y_train.shape)
                    
                    if modelIter == 0:
                        ''' Model 1'''
                        filterSize = [2,3]
                        epochSize = 5
                        model = defineCovNetsModel(modelIter, filterSize)
                    elif modelIter == 1:
                        ''' Model 2'''
                        filterSize = [4,5]
                        epochSize = 7
                        model = defineCovNetsModel(modelIter, filterSize)
                    
                    print("\n\t\t CURRENT MODEL:",modelIter)
                    
                    x_validation = x_validationList[modelIter]
                    y_validation = y_validationList[modelIter]
                    
                    model.fit(x_train,y_train, batch_size = int(np.rint(trainSentenceDimension/2)), epochs =epochSize, verbose=0,validation_data=(x_validation,y_validation))
                    score = model.evaluate(x_validation, y_validation, verbose=0)
                    
                    reset = False
                    if(score[1] <= 0.30):
                        print("reset>>",score)
                        reset = True
                    
                    x_test = x_testList[modelIter]
                    y_test = y_testList[modelIter]
                    candidateRecordsDictionary = dict(featureRecordDimension[modelIter])
                    candidateRecordsDecisionDictionary = dict(featureSummaryDimension[modelIter])
                    if not reset:
                        x_start = 0
                        print('total loss>>>',score)
                        for fileId in candidateRecordsDecisionDictionary.keys():
                            if fileId in testRecords:
                                y_size = len(list(candidateRecordsDecisionDictionary[fileId]))
                                x_size = x_start + y_size
                                predictionScore = np.array([])
                                for predictIndex in range(x_start,x_size):
                                    testSequenceSpan = MAX_SEQUENCE_LENGTH[modelIter]
                                    testFeatureDimension = EMBEDDING_DIM 
                                    testExample = x_test[predictIndex].reshape(1,testSequenceSpan,testFeatureDimension)
                                    probability = model.predict(testExample, verbose=2)
                                    predictionScore = populateArray(predictionScore, np.array(probability[0]).reshape(1,2))
            
                                #print("prior score>>",predictionScore,"\t>>",fileId,"\n\t>>",candidateRecordsDictionary[fileId])
                                if modelIter == 0:
                                    status = trannslateMinScoresAsVotes(predictionScore)           
                                    #status = translateScoresAsVotes(predictionScore)
                                scoreDict[fileId] = status
                                x_start = x_size
                        scoreDictList.append(scoreDict)
                    
                    if reset:
                        break
                    
                if (len(scoreDictList) == MODEL_SIZE):
                    runIndex = runIndex+1
                    startIndex = endIndex
                    for index, predictSummary in enumerate(scoreDictList):
                        modelVotingList = reorganizeModelBasedVoting(modelVotingList,predictSummary,index,MODEL_SIZE)
                        
        
        predictOutput={}             
        for index,voteDictionary in enumerate(modelVotingList):
            print("results Model:",index)
            for fileId in testRecords:
                status = 0
                if fileId in voteDictionary.keys():
                    print("\t",fileId,"::",voteDictionary[fileId])
                    decoyVoteDictionary = dict(Counter(voteDictionary[fileId]))
                    '''
                    if len(set(decoyVoteDictionary.values())) == 1:
                        status = BALANCE_BIAS
                    else:'''
                    status = int(max(decoyVoteDictionary.items(),key=itemgetter(1))[0])
                decoyVoteList = []
                if fileId in predictOutput.keys():
                    decoyVoteList = predictOutput[fileId]
                decoyVoteList.append(status)
                if index == (MODEL_SIZE-1):
                    finalStatus = 0
                    if np.sum(decoyVoteList) == MODEL_SIZE:
                        finalStatus = 1
                    decoyStatusList = []
                    if fileId in stratifiedPredictOut.keys():
                        decoyStatusList = stratifiedPredictOut[fileId]
                    decoyStatusList.append(finalStatus)
                    stratifiedPredictOut.update({fileId:decoyStatusList})
                else:
                    predictOutput[fileId] = decoyVoteList
        
    stratifiedSplitIndex +=1  
    
for fileId in stratifiedPredictOut.keys():
    decoyVoteDictionary = dict(Counter(stratifiedPredictOut.get(fileId)))
    '''
    if len(set(decoyVoteDictionary.values())) == 1:
        status = BALANCE_BIAS
    else:'''
    status = int(max(decoyVoteDictionary.items(),key=itemgetter(1))[0])
        
    print("\t",fileId,"::",stratifiedPredictOut.get(fileId),"\t>>>",status)
    stratifiedPredictOut.update({fileId:status})
    
updateOutputXml(stratifiedPredictOut,ctAttributes,XMLTEXT)

actualOutput = {}
testDimension = dict(featureSummaryDimension[0])
for fileId in testDimension:
    if fileId in testRecords:
        status = 1
        if np.sum(np.array(list(testDimension[fileId]))[:,1]) == 0:
            status = 0
        actualOutput[fileId] = status
        

y_predict = []
y_actual = []
#if len(predictOutput) == len(actualOutput):
for fileId in stratifiedPredictOut.keys():
    print("\t>>",fileId,"\t",stratifiedPredictOut[fileId],"\t",actualOutput[fileId])
    value = stratifiedPredictOut[fileId]
    if type(value) == list:
        value = 0
    y_predict.append(value)
    y_actual.append(actualOutput[fileId])


zeroRatio = 0
oneRatio = 0
for fileId in stratifiedPredictOut.keys():
    if int(stratifiedPredictOut[fileId]) == 0:
        zeroRatio += 1
    else:
        oneRatio += 1
        
zeroProp = zeroRatio/len(stratifiedPredictOut)
oneProp = oneRatio/len(stratifiedPredictOut)

print("o/p prop \t not-met >>",zeroProp,"\n\t met >>",oneProp) 

print(classification_report(y_actual, y_predict))

print(precision_score(y_actual, y_predict, average="macro"))
print(precision_score(y_actual, y_predict, average="micro"))
print(precision_score(y_actual, y_predict, average="weighted"))
print(precision_score(y_actual, y_predict, average=None))

print(recall_score(y_actual, y_predict, average="macro"))
print(recall_score(y_actual, y_predict, average="micro"))
print(recall_score(y_actual, y_predict, average="weighted"))
print(recall_score(y_actual, y_predict, average=None))        
