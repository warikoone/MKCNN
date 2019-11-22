'''
Created on Mar 27, 2018

@author: neha
'''

import os
import sys
import re
import json
import nltk
import numpy as np
from collections import Counter
from xml.dom import minidom
from operator import itemgetter
from nltk.tokenize import word_tokenize
from gensim.models import KeyedVectors
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Conv1D, GlobalMaxPool1D, Flatten, Embedding
from keras.layers import Dense, Input, Dropout,Activation, MaxPool1D
from keras.layers.merge import Concatenate
from keras.optimizers import Adam


TRIGGER_FEATURE = ""
TRIGGER_ATTRIBUTE = ""
VARIABLE_ATTRIBUTE_THRESHOLD = 0.3
COSINE_SIMILARITY_THRESHOLD = 0.1
COSINE_DITRIBUTION_THRESHOLD = 98.5
SENTENCE_TRIM_LIMIT = 4
MAX_NB_WORDS = 200000
MAX_SEQUENCE_LENGTH = 0
VOCAB_WORDSIZE = 0
EMBEDDING_DIM = 300
EMBEDDING_MATRIX=np.zeros((1,EMBEDDING_DIM))
EMBEDDING_FILE = "/home/neha/Disk_R/Bio_NLP/N2C2_Task/Data/embedding/GoogleNews-vectors-negative300.bin"
#EMBEDDING_FILE = "/home/neha/Disk_R/Bio_NLP/N2C2_Task/Data/embedding/wiki.en.vec"
BASIC_FILTERS = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n'
MODEL_VARIABLE = {2:'ADVANCED-CAD',4:'ASP-FOR-MI',6:'DIETSUPP-2MOS',8:'ENGLISH',10:'KETO-1YR',12:'MAKES-DECISIONS'}

''' Initialize embedding matrix '''
PRETRAIN_WORD2VEC = KeyedVectors.load_word2vec_format(EMBEDDING_FILE,binary=True)

category_index = {"not met":0, "met":1}
#print(category_reverse_index)
engStopwords = set(nltk.corpus.stopwords.words('english'))
#print(engStopwords)

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
    
    #print(">>>",attributeExistStatus)
    return(attributeExistStatus)

#remove newlines from the string descriptionText
def removeNewlines(sentence):
    sentence = re.sub('\n', ' ', sentence)
    sentence = sentence.strip()
    return(sentence)

#remove stop words from each sentence
def removeStopWords(sentence):

    sentence = removeNewlines(sentence)
    tokenizedSentence = word_tokenize(sentence)
    filteredSentence = filter(lambda word: word not in engStopwords, tokenizedSentence)
    filteredSentence = ' '.join(filteredSentence)
    return(filteredSentence)
    
# Split each section of the record into constituent sentences
def performSentenceSplit(sentenceBundle):
    tokenizer = nltk.data.load('nltk:tokenizers/punkt/english.pickle')
    bundledSent = tokenizer.tokenize(sentenceBundle)
    return(bundledSent)

def performWordSplit(sentence):
    return(word_tokenize(sentence))

def extractVisitRecords(recordBatch):
    splitBatchRecords = re.split("\*+", recordBatch)
    count=0;
    tokenIndexMapper = []
    sentenceRecord = []
    for splitSubRecords in splitBatchRecords:
        if str(splitSubRecords).isspace():
            m=9
            #print("token",count)
            #break
        else:
            bundledSentence = performSentenceSplit(splitSubRecords)
            for splitSentence in bundledSentence:
                ''' dictionary data'''
                tokenIndexMapper.append(removeNewlines(splitSentence))
                ''' sentence record'''
                sentenceRecord.append(removeStopWords(splitSentence))
                #break
        count+=1
    return(tokenIndexMapper,sentenceRecord);

def vectorRecords(initializeRecordTokenizer, record):
    sentenceVectorList = initializeRecordTokenizer.texts_to_sequences(record)
    return(sentenceVectorList)

def paddingVector(recordVectorList):
    global MAX_SEQUENCE_LENGTH
    paddedVector = pad_sequences(recordVectorList, maxlen=MAX_SEQUENCE_LENGTH)
    print(len(paddedVector),'>>>',MAX_SEQUENCE_LENGTH)
    return(paddedVector)

def checkForSelectiveTags(posTaggedTokens):
    matchedWords = []
    for posTag in posTaggedTokens:
        posTag = str(posTag)
        posTag = re.sub('^\(|\)$','',posTag)
        tokenSplit = re.split('\,',posTag)
        tagMatcher = re.match('\'(VB|NN).{0,1}\'', str(tokenSplit[1]).strip())
        if tagMatcher:
            matchedWords.append(re.sub('^\'|\'$', '',str(tokenSplit[0]).strip()))
    return(matchedWords)

def triggerFeatureTextLookUp(identifiedToken, recordDictionary, similarityDictionary):
    
    for record in recordDictionary:
        for subRecord in record:
            for token in identifiedToken:
                tokenMatcher = re.search(token, subRecord)
                if tokenMatcher:
                    print('record>>',subRecord)
                    posTaggedTokens = nltk.pos_tag(performWordSplit(subRecord))
                    matchedWords = checkForSelectiveTags(posTaggedTokens)
                    if len(matchedWords) != 0:
                        print('matchedWord>>',matchedWords)
                        for word in matchedWords:
                            if word in similarityDictionary.keys():
                                print('similaity>>',similarityDictionary[word],'\t>>',word)
    return()

def calculatePreTrainedWordSimilarity(wordIndexMapper,triggerFeature):
    
    global COSINE_SIMILARITY_THRESHOLD, COSINE_DITRIBUTION_THRESHOLD, PRETRAIN_WORD2VEC
    similarityDictionary = {}
    for word in wordIndexMapper.keys():
        if word in PRETRAIN_WORD2VEC.vocab:
            cosineSimilarity = np.around(PRETRAIN_WORD2VEC.similarity(triggerFeature,word),decimals=3)
            if cosineSimilarity > COSINE_SIMILARITY_THRESHOLD:
                similarityDictionary[word] = cosineSimilarity
                
    if len(similarityDictionary) > 0:
        similarityDictionary = dict(sorted(similarityDictionary.items(),key=itemgetter(1),reverse=True))
        functionalCutOff = np.around(np.percentile(
                                    list(similarityDictionary.values()),COSINE_DITRIBUTION_THRESHOLD),decimals=3)
        similarityDictionary = dict(filter((
            lambda tokenItem : (tokenItem[1] >= functionalCutOff)),similarityDictionary.items()))
    return(similarityDictionary)

def screenForTriggerFeatures(initializeRecordTokenizer,dictionaryList,initialTriggerFeature):
    
    wordList=[]
    wordIndexMapper = initializeRecordTokenizer.word_index
    triggerFeature = initialTriggerFeature
    similarityDictionary = calculatePreTrainedWordSimilarity(wordIndexMapper,triggerFeature)
    dictionaryList.append(similarityDictionary)
    if len(similarityDictionary) > 0:
        for triggerWord in similarityDictionary.keys():
            if triggerWord != TRIGGER_FEATURE:
                ''' restrict context branching to first order trigger word similarity vector '''
                if triggerFeature == TRIGGER_FEATURE :
                    wordList.extend(screenForTriggerFeatures(initializeRecordTokenizer, dictionaryList, triggerWord))
                else:
                    break
        wordList.extend(list(similarityDictionary.keys()))
    return(wordList)
    #if len(identifiedToken) != 0 :
        #triggerFeatureTextLookUp(identifiedToken, recordDictionary, similarityDictionary)

def populateCountDictionary(wordCount,word):
    counter = 1    
    if word in wordCount.keys():
        counter = wordCount[word] + 1
        wordCount.update({word:counter})
    else:
        wordCount[word] = counter        

''' generate tokens ''' 
def indexRecordTokens(recordTokens):
    
    global BASIC_FILTERS
    initializeRecordTokenizer = Tokenizer(filters= BASIC_FILTERS , split=' ',lower=True)
    initializeRecordTokenizer.fit_on_texts(recordTokens)
    return(initializeRecordTokenizer)    

''' generate pattern to identify instances for different attributes '''
def identifyTokensForInstanceScreening(wordCount, dictionaryList):
    
    featureVectorSize = len(dictionaryList)
    wordCount = dict(filter(lambda scorePair:((scorePair[1]/featureVectorSize)>=0.5), wordCount.items()))
    if len(wordCount) > 0:            
        wordCount = dict(sorted(wordCount.items(),key=itemgetter(1),reverse=True))
    else:
        forcedCutOff = np.around(np.percentile(list(dictionaryList[0].values()),COSINE_DITRIBUTION_THRESHOLD),decimals=3)
        wordCount = dict(filter((lambda tokenItem : (tokenItem[1] >= forcedCutOff)),dictionaryList[0].items()))
       
    #print("\n updated>>",wordCount)
    wordList = list(wordCount.keys())
    wordPattern = '|'.join(word for word in wordList)
    #print("wordPattern>>",wordPattern)
    return(wordPattern)

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

''' isolate candidate instances from document record '''
def isolateCandidateInstances(currentDocumentRecord, triggerDominatedSentiment, recordCandidateInstances):
    
    global SENTENCE_TRIM_LIMIT,MAX_SEQUENCE_LENGTH,VOCAB_WORDSIZE
    for sentence in currentDocumentRecord:
        #print("\t original sent**",sentence)
        sentenceTokens = list(performWordSplit(sentence))
        startIndexList=[];endIndexList=[]
        for wordPattern in triggerDominatedSentiment:
            for index, token in enumerate(sentenceTokens):
                tokenArray = mixedCharacterTokenizer(str(token).strip()) 
                #print("token>>",str(token),"tokenarray>",tokenArray)
                wordMatcherArray = list(filter(lambda word : re.match(wordPattern.strip(), word.strip(), flags=re.RegexFlag.IGNORECASE), tokenArray))
                #wordMatcher = re.match(wordPattern.strip(), str(token).strip(), flags=re.RegexFlag.IGNORECASE)
                if len(wordMatcherArray)>0:
                    currIndex = (index-SENTENCE_TRIM_LIMIT)
                    if currIndex < 0:
                        currIndex = 0
                    startIndexList.append(currIndex)
                    currIndex = (index+SENTENCE_TRIM_LIMIT)
                    if currIndex > (len(sentenceTokens)-1):
                        currIndex = len(sentenceTokens)-1
                    endIndexList.append(currIndex)
        if (len(startIndexList)>0) and (len(endIndexList)>0):
            #print("start aftr>>",startIndexList)
            #print("end aftr>>",endIndexList)
            startIndex = min(startIndexList)
            endIndex = max(endIndexList)
            #print("limits\t >>",startIndex,'\t>>',endIndex)
            sentenceTokens = list(filter(lambda token : re.match('[^\W]', token.strip(), flags=re.RegexFlag.IGNORECASE), sentenceTokens[startIndex:endIndex+1]))
            #print("updated>>",sentenceTokens)
            if(len(sentenceTokens) > MAX_SEQUENCE_LENGTH):
                MAX_SEQUENCE_LENGTH = len(sentenceTokens)
            sentence = ' '.join(word.strip() for word in sentenceTokens)
            #print("final token>>",sentence)
            if len(sentence) > 0:
                recordCandidateInstances.append(sentence.strip())

def existsBranchedRelation(sourceList, targetList, sentimentList):
    
    relation = False
    for word in sourceList:
        if word in targetList:
            sentimentList.append(word)
            relation = True
            #print("relatd term>>",word)
        
    return(relation)

def retreiveSeenPreTrainedEmbedding(embeddingMatrix, index, token):
    
    global EMBEDDING_DIM, PRETRAIN_WORD2VEC
    embeddingMatrix[index] = PRETRAIN_WORD2VEC.word_vec(token)[0:EMBEDDING_DIM]
    
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

def buildPattern(currPatternMap):
    
    pattern = ''.join(str(currPattern[1]) for currPattern in currPatternMap.items())
    return(pattern)

def unseenExistsInPreTrainedEmbedding(pattern):
    
    global PRETRAIN_WORD2VEC
    if pattern in PRETRAIN_WORD2VEC.vocab:
        return(True,pattern)
    else:
        return(False,'')

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

def populateArray(currArray,appendArray):
    
    if currArray.shape[0] == 0:
        currArray = appendArray
    else:
        currArray = np.insert(currArray, currArray.shape[0], appendArray, 0)
    return(currArray)        
    
def assimilatePreTrainedEmbeddings(sentenceTokens, recordMatrix):
    
    global EMBEDDING_DIM, PRETRAIN_WORD2VEC, VOCAB_WORDSIZE,EMBEDDING_MATRIX
    embeddingMatrix = np.zeros((MAX_SEQUENCE_LENGTH,EMBEDDING_DIM))
    for index,token in enumerate(sentenceTokens):
        if token in PRETRAIN_WORD2VEC.vocab:
            retreiveSeenPreTrainedEmbedding(embeddingMatrix, index, token)
        else:
            retreiveUnseenPreTrainedEmbedding(embeddingMatrix, index, token)
    
    recordMatrix = populateArray(recordMatrix, embeddingMatrix)
    EMBEDDING_MATRIX = np.concatenate((EMBEDDING_MATRIX,embeddingMatrix),0)
    return(recordMatrix)

def findSentenceVectorEmbeddings(candidateRecordsDictionary,referenceRecords, candidateRecordsDecisionDictionary, referenceRecordsSummary):
    
    global EMBEDDING_DIM
    x_record = np.array([])
    y_summary = np.array([])
    #print("\n ref rec>>",referenceRecords,"\n\t refSum>>",referenceRecordsSummary)
    for fileId in candidateRecordsDictionary.keys():
        if ((candidateRecordsDictionary[fileId] is not None) and (fileId in referenceRecords) and (fileId in referenceRecordsSummary)):
            for sentence in list(candidateRecordsDictionary[fileId]):
                x_record = assimilatePreTrainedEmbeddings(performWordSplit(sentence),x_record)
            #appendArray = np.array(list(candidateRecordsDecisionDictionary[fileId])).reshape(-1,1)
            appendArray = np.array(list(candidateRecordsDecisionDictionary[fileId]))
            #print("append>>",appendArray,"\tshape>>",appendArray.shape,"file>>",fileId)
            if appendArray.shape[1] == 2:
                y_summary = populateArray(y_summary,appendArray)
                
    return(x_record,y_summary)
            
def vectorizeDecisionStatus(recordCandidateInstances,statusValue):
    
    vectorizedStatus = []
    for index in range(len(recordCandidateInstances)):
        statusList = to_categorical(category_index[statusValue], len(category_index))
        vectorizedStatus.insert(index,statusList)
    return(vectorizedStatus)  

def printRecords(candidateRecordsDictionary):
    
    for recordName in candidateRecordsDictionary.keys():
        print("\n File::",recordName)
        recordList = candidateRecordsDictionary[recordName]
        for sentence in recordList:
            tagSet = dict(nltk.pos_tag(performWordSplit(sentence)))
            posSet =str(' '.join(word[1] for word in tagSet.items()))
            print("\t::",sentence,"\t:::",posSet)

    
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
        TRIGGER_FEATURE = openConfigurationFile(configFile,TRIGGER_ATTRIBUTE)
    else:
        print("\n Invalid model requested, please try again")

listedPatientRecords = os.listdir(trainDataPath)
listedPatientRecords = sorted(listedPatientRecords)
trainRecords, testRecords, trainSummary, testSummary = train_test_split(listedPatientRecords,listedPatientRecords, test_size=0.25, random_state=0)
trainRecords, testRecords, trainSummary, testSummary = map(lambda recordList : sorted(recordList),list([trainRecords, testRecords, trainSummary, testSummary]))

positiveTrigger =[]
negativeTrigger=[]
candidateRecordsDictionary = {}
candidateRecordsDecisionDictionary = {}
indexMapperDictionary = []
recordDictionary = []
fieldIndex=0
for filename in listedPatientRecords:
    #if filename.endswith("259.xml"):
        fieldIndex += 1
        #print("".join([dirname,"/",filename]))
        fileAddress = "".join([trainDataPath,"/",filename])
        doc = minidom.parse(fileAddress)
        descriptionText = doc.getElementsByTagName("TEXT")[0]
        visitRecord = descriptionText.firstChild.data
        decisionTree = doc.getElementsByTagName("TAGS")
        attributeExistStatus = populateAttributeStatus(decisionTree,ctAttributes)
        
        #print(visitRecord)
        tokenIndexMapper,sentenceRecord = extractVisitRecords(visitRecord)
        #print("\n currentDocumentRecord>>",currentDocumentRecord)
        indexMapperDictionary.extend(tokenIndexMapper)
        recordDictionary.insert(fieldIndex, sentenceRecord)
        
        ''' create document record index '''
        initializeRecordTokenizer = indexRecordTokens(tokenIndexMapper)
        
        ''' Isolate feature words from document records'''
        dictionaryList = []
        wordList = screenForTriggerFeatures(initializeRecordTokenizer, dictionaryList, TRIGGER_FEATURE)
        wordList = set(wordList)
        #print("\n\t dictionaryList>>",dictionaryList)
        #print("\n wordList",wordList)
        
        ''' Compare scored feature tokens for instance screening '''
        seedIndex=0
        triggerDominatedSentiment = []
        if(len(dictionaryList) > 0):
            sourceSentimentDictionary = dict(dictionaryList[seedIndex])
            sourceSentiment = list(sourceSentimentDictionary.keys())
            rootTrigger = sourceSentiment[seedIndex]
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
            
            ''' allow tokens which are branched from a similar > variable threshold'''
            # print("rechecking for threshold>>", sourceSentimentDictionary[rootTrigger])
            if filename in testRecords:
                VARIABLE_ATTRIBUTE_THRESHOLD = 0.2
            if (sourceSentimentDictionary[rootTrigger] >= VARIABLE_ATTRIBUTE_THRESHOLD):
                if rootTrigger not in triggerDominatedSentiment:
                    triggerDominatedSentiment.append(rootTrigger)
            triggerDominatedSentiment = list(set(triggerDominatedSentiment))
            #if filename in testRecords:
                #print(" file>>",filename,"\t>>",triggerDominatedSentiment)
        #print("\n triggerDominatedSentiment>>",triggerDominatedSentiment)

        '''                    
        wordCount = {}
        for similarityWords in dictionaryList:
            for word in wordList:
                if word in similarityWords.keys():
                    populateCountDictionary(wordCount,word)
                    
        wordPattern = identifyTokensForInstanceScreening(wordCount, dictionaryList)
        ''' 
        
        ''' Screen for candidate instances using trigger Sentiment'''
        recordCandidateInstances=list()
        if len(triggerDominatedSentiment) > 0:
            isolateCandidateInstances(sentenceRecord, triggerDominatedSentiment, recordCandidateInstances)
            
            recordCandidateInstances = list(set(recordCandidateInstances))
            candidateRecordsDictionary[filename] = recordCandidateInstances
            candidateRecordsDecisionDictionary[filename] = vectorizeDecisionStatus(recordCandidateInstances,attributeExistStatus[TRIGGER_ATTRIBUTE])
            if attributeExistStatus[TRIGGER_ATTRIBUTE] == 'not met':
                negativeTrigger.extend(triggerDominatedSentiment)
            else:
                positiveTrigger.extend(triggerDominatedSentiment)

#print("max seq>>",MAX_SEQUENCE_LENGTH)
#printRecords(candidateRecordsDictionary)
#printRecords(candidateRecordsDecisionDictionary)

positiveTrigger = set(positiveTrigger)
negativeTrigger = set(negativeTrigger)
allowedNegativeTerms = negativeTrigger - positiveTrigger
negRecords=[]
for fileId in candidateRecordsDictionary:
    if np.sum(np.array(list(candidateRecordsDecisionDictionary[fileId]))[:,1]) == 0:
        sentenceList = list()
        #print("file::",fileId)
        for sentence in list(candidateRecordsDictionary[fileId]):
            #print("b4::",sentence)
            sentenceArray = []
            for token in performWordSplit(sentence):
                tokenArray = mixedCharacterTokenizer(str(token))
                term =''
                for item in tokenArray:
                    matchTerm = list(filter(lambda word : re.match(item.strip(), word, flags=re.RegexFlag.IGNORECASE),positiveTrigger))
                    if len(matchTerm) == 0:
                        term = term+item 
                sentenceArray.append(term)
            sent = str(' '.join(word.strip()for word in sentenceArray))    
            #print("aftr::",sent)
            sentenceList.append(sent.strip())
        #candidateRecordsDictionary[fileId] = sentenceList
        if fileId in trainRecords:
            negRecords.append(fileId)

''' duplication of records'''
'''            
trainRecords = list(trainRecords)
trainSummary = list(trainSummary)
for fileId in negRecords:
    replacedId = re.sub('\.xml', '', fileId)
    replacedId = int(replacedId+str(100))
    for i in range(0,20):
        updatedId = str(replacedId+i)+'.xml'
        trainRecords.append(updatedId)
        trainSummary.append(updatedId)
        candidateRecordsDictionary[updatedId] = candidateRecordsDictionary[fileId]
        candidateRecordsDecisionDictionary[updatedId] = candidateRecordsDecisionDictionary[fileId]
        
printRecords(candidateRecordsDictionary)
'''            

''' proportion +ve/-ve training samples'''
positiveCases = []
negativeCases = []
for fileId in candidateRecordsDictionary:
    if fileId in trainRecords:
        ''' not met'''
        if np.sum(np.array(list(candidateRecordsDecisionDictionary[fileId]))[:,1]) == 0:
            negativeCases.append(fileId)
        else:
            positiveCases.append(fileId)

scoreDict={}
index=0
iterateLength = int(np.rint(len(positiveCases)/6))
print("\n iterate length>>",iterateLength)
#iterateLength = 3
while index < iterateLength:
    print("INDEX>>",index)              
    subSetPositiveCases = list(np.random.choice(positiveCases,6,replace=False))
    print("+ve records>>",subSetPositiveCases,"-ve cases>>",negativeCases)
    subSetPositiveCases.extend(negativeCases)
    trainRecords = subSetPositiveCases
    trainSummary = trainRecords
    print("trainRecords>>",trainRecords,"trainSummary>>",trainSummary)

    EMBEDDING_MATRIX=np.zeros((1,EMBEDDING_DIM))
    x_train,y_train = findSentenceVectorEmbeddings(candidateRecordsDictionary, trainRecords, candidateRecordsDecisionDictionary, trainSummary)
    #print("\n  train EMBEDDING_MATRIX>>",EMBEDDING_MATRIX.shape)
    #print("\n x_train>>",x_train.shape,"\t y_train>>",y_train.shape)
    x_validation,y_validation = findSentenceVectorEmbeddings(candidateRecordsDictionary, testRecords, candidateRecordsDecisionDictionary, testSummary)
    #print("\n validation EMBEDDING_MATRIX>>",EMBEDDING_MATRIX.shape)
    #print("\n x_validation>>",x_validation.shape,"\t y_validation>>",y_validation.shape)
    #print('Null word embeddings: %d' % np.sum(np.sum(EMBEDDING_MATRIX, axis=1) == 0))
    
    #print("\n embedding matrix>>",EMBEDDING_MATRIX)           
    EMBEDDING_MATRIX = np.delete(EMBEDDING_MATRIX,0,0)
    
    #print("\n AFTR EMBEDDING_MATRIX>>",EMBEDDING_MATRIX.shape,'\t>>>',MAX_SEQUENCE_LENGTH)
    #print("\n AFTR x_record>>",x_train.shape,'\t AFTR x_validation>>>',x_validation.shape)
    #print("\n AFTR y_summary>>",y_train.shape,"\t AFTR y_validation>>",y_validation.shape)
    
    modelInput = Input(shape=(MAX_SEQUENCE_LENGTH,EMBEDDING_DIM))
    
    '''
    embeddingLayer = Embedding(EMBEDDING_MATRIX.shape[0],
                               EMBEDDING_MATRIX.shape[1],
                               weights = [EMBEDDING_MATRIX],
                               input_length=MAX_SEQUENCE_LENGTH,
                               trainable = False)(modelInput)
    
    print("\n EMbedlayer>>>",embeddingLayer)
    '''
    embeddingLayer = modelInput
    
    #print("\n input layer>>>",embeddingLayer)
    
    convBlocks = []
    variableFilterSize = [2,3,4]
    for eachFilter in variableFilterSize:
        currFilter = int(np.rint(MAX_SEQUENCE_LENGTH/(2*eachFilter)))  
        singleConv = Conv1D(filters=currFilter,kernel_size=eachFilter,padding='valid',activation='relu',strides=1)(embeddingLayer)
        #print("\n convolution layer>>>",singleConv)
        
        singleConv = MaxPool1D(pool_size = 2)(singleConv)
        #print("\n MaxPool1D layer>>>",singleConv)
    
        singleConv = Flatten()(singleConv)
        #print("\n Flatten layer>>>",singleConv)
        
        convBlocks.append(singleConv)
        
    tranformLayer = Concatenate()(convBlocks) if len(convBlocks) > 1 else convBlocks[0] 
    
    #conv = Dropout(0.5)(conv)
    tranformLayer = Dense(10,activation='relu')(tranformLayer)
    #print("\n 1st Dense layer>>>",tranformLayer)
    
    modelOutput = Dense(2,activation='sigmoid')(tranformLayer)
    #print("\n 2nd Dense layer>>>",modelOutput)
    
    model = Model(input = modelInput, output=modelOutput)
    #print("\n model>>>",model)
    
    lrAdam = Adam(lr=0.01,decay=0.0001)
    model.compile(optimizer=lrAdam, loss='binary_crossentropy',metrics=['accuracy'])
    
    #model.summary()
    
    '''
    x = np.random.rand(2,9,5)
    y=np.ones((2,1))
    
    x1 = np.random.rand(2,9,5)
    y1=np.ones((2,1))
    '''
    ''' Input Tensor Structure'''
    trainSentenceDimension = int(np.rint(x_train.shape[0]/MAX_SEQUENCE_LENGTH))
    trainSequenceSpan = MAX_SEQUENCE_LENGTH
    trainFeatureDimension = EMBEDDING_DIM
    
    x_train = x_train.reshape(trainSentenceDimension,trainSequenceSpan,trainFeatureDimension)
    #print(" training tensor shape::",x_train.shape,"\t y::",y_train.shape)
    
    validationSentenceDimension = int(np.rint(x_validation.shape[0]/MAX_SEQUENCE_LENGTH))
    validationSequenceSpan = MAX_SEQUENCE_LENGTH
    validationFeatureDimension = EMBEDDING_DIM 
    
    x_validation = x_validation.reshape(validationSentenceDimension,validationSequenceSpan,validationFeatureDimension)
    #print(" validation tensor shape::",x_validation.shape,"\t y::",y_validation.shape)
    
    model.fit(x_train,y_train, batch_size = int(np.rint(trainSentenceDimension/3)), epochs =9, verbose=2,validation_data=(x_validation,y_validation))
    score = model.evaluate(x_validation, y_validation, verbose=0)
    
    reset = False
    if(score[1] <= 0.50):
        print("reset>>",score)
        reset = True
    
    if not reset:
        x_start = 0
        for fileId in candidateRecordsDecisionDictionary.keys():
            print('total loss>>>',score)
            if fileId in testSummary:
                y_size = len(list(candidateRecordsDecisionDictionary[fileId]))
                x_size = x_start + y_size
                predictionScore = np.array([])
                for predictIndex in range(x_start,x_size):
                    testExample = x_validation[predictIndex].reshape(1,validationSequenceSpan,validationFeatureDimension)
                    probability = model.predict(testExample, verbose=2)
                    predictionScore = populateArray(predictionScore, np.array(probability[0]).reshape(1,2))
                comparativeScore = (np.sum(predictionScore,0)/len(predictionScore))
                status = 1
                if comparativeScore[0]>comparativeScore[1]:
                    status = 0
                tempScoreList = []
                if fileId in scoreDict.keys():
                    tempScoreList = list(scoreDict[fileId])
                tempScoreList.append(status)
                scoreDict[fileId] = tempScoreList
                x_start = x_size
        index = index+1
    

actualOutput = {}
for fileId in candidateRecordsDecisionDictionary:
    if fileId in testSummary:
        status = 1
        if np.sum(np.array(list(candidateRecordsDecisionDictionary[fileId]))[:,1]) == 0:
            status = 0
        actualOutput[fileId] = status

predictOutput={} 
for fileId in scoreDict.keys():
    #print("\t>>",fileId,':::',list(scoreDict[fileId]))
    criteriaDict = dict(Counter(list(scoreDict[fileId])))
    status = int(max(criteriaDict.items(),key=itemgetter(1))[0])
    predictOutput[fileId] = status
    
''' Evaluation Matrix'''
'''    
TP=0
FP=0
TN=0
FN=0
if len(predictOutput) == len(actualOutput):
    for fileId in predictOutput.keys():
        if ((predictOutput[fileId]==1) and (predictOutput[fileId]==actualOutput[fileId])):
            TP += 1
        elif ((predictOutput[fileId]==0) and (predictOutput[fileId]==actualOutput[fileId])):
            TN += 1
        elif ((predictOutput[fileId]==1) and (predictOutput[fileId]!=actualOutput[fileId])):
            FP += 1
        elif ((predictOutput[fileId]==0) and (predictOutput[fileId]!=actualOutput[fileId])):
            FN += 1
else:
    print("\n\t SIZE MISMATCH")
 '''
y_predict = []
y_actual = []
if len(predictOutput) == len(actualOutput):
    for fileId in predictOutput.keys():
        y_predict.append(predictOutput[fileId])
        y_actual.append(actualOutput[fileId])

print(classification_report(y_actual, y_predict))


