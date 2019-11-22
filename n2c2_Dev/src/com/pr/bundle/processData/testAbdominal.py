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
from collections import Counter
from datetime import *
from dateutil import relativedelta
from xml.dom import minidom
import xml.etree.ElementTree as xml
from operator import itemgetter
from nltk.tokenize import word_tokenize
from gensim.models import KeyedVectors
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report,f1_score,precision_score,recall_score
from keras.preprocessing.text import Tokenizer
#from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Conv1D, GlobalMaxPool1D, Flatten, Embedding
from keras.layers import Dense, Input, Dropout,Activation, MaxPool1D
from keras.layers.merge import Concatenate
from keras.optimizers import Adam

MODEL_VARIABLE = {1:'ABDOMINAL',3:'ALCOHOL-ABUSE',5:'CREATININE',7:'DRUG-ABUSE',9:'HBA1C',11:'MAJOR-DIABETES',13:'MI-6MOS'}
engStopwords = set(nltk.corpus.stopwords.words('english'))
BASIC_FILTERS = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n'
TRIGGER_FEATURE = []
TRIGGER_ATTRIBUTE = ""
COSINE_SIMILARITY_THRESHOLD = 0.1
COSINE_DITRIBUTION_THRESHOLD = 98.5
EMBEDDING_FILE = "/home/neha/Disk_R/Bio_NLP/N2C2_Task/Data/embedding/GoogleNews-vectors-negative300.bin"
EMBEDDING_DIM = 300
EMBEDDING_MATRIX=np.zeros((1,EMBEDDING_DIM))
VARIABLE_ATTRIBUTE_THRESHOLD = 0.0
MAX_SEQUENCE_LENGTH = [0]
SENTENCE_TRIM_LIMIT = 4
category_index = {"not met":0, "met":1}

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
        statusList = to_categorical(category_index[statusValue], len(category_index))
        vectorizedStatus.insert(index,statusList)
    return(vectorizedStatus)

def printRecords(candidateRecordsDictionary,summary):
    
    for recordName in candidateRecordsDictionary.keys():
        print("\n File::",recordName)
        recordList = list(candidateRecordsDictionary[recordName])
        for sentence in recordList:
            print("\t::",sentence,"\t>>",np.sum(np.array(summary[recordName]),0))
    
def defineSampleRandomizationSize(positiveCases,negativeCases):
    
    segmentLength = 1000
    leaderCase = positiveCases
    trailerCase = negativeCases 
    if (len(negativeCases) < len(positiveCases)):
        if ((len(negativeCases) > 0) and (len(negativeCases) < segmentLength)):
            segmentLength = len(negativeCases)
            
    else:
        leaderCase = negativeCases
        trailerCase = positiveCases
        if ((len(positiveCases) > 0) and (len(positiveCases) < segmentLength)):
            segmentLength = len(positiveCases)
            
    iterateLength = int(np.rint(len(leaderCase)/segmentLength))
        
    return(leaderCase,trailerCase,iterateLength,segmentLength)

def retreiveSeenPreTrainedEmbedding(embeddingMatrix, index, token):
    
    global EMBEDDING_DIM, PRETRAIN_WORD2VEC
    embeddingMatrix[index] = PRETRAIN_WORD2VEC.word_vec(token)[0:EMBEDDING_DIM]
    
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
    
    global EMBEDDING_DIM, PRETRAIN_WORD2VEC,EMBEDDING_MATRIX,MAX_SEQUENCE_LENGTH
    embeddingMatrix = np.zeros((MAX_SEQUENCE_LENGTH[modelId],EMBEDDING_DIM))
    for index,token in enumerate(sentenceTokens):
        if token in PRETRAIN_WORD2VEC.vocab:
            retreiveSeenPreTrainedEmbedding(embeddingMatrix, index, token)
        else:
            retreiveUnseenPreTrainedEmbedding(embeddingMatrix, index, token)
    
    recordMatrix = populateArray(recordMatrix, embeddingMatrix)
    EMBEDDING_MATRIX = np.concatenate((EMBEDDING_MATRIX,embeddingMatrix),0)
    return(recordMatrix)       
      
def findSentenceVectorEmbeddings(candidateRecordsDictionary,referenceRecords, candidateRecordsDecisionDictionary, referenceRecordsSummary,modelId):
    
    global EMBEDDING_DIM
    x_record = np.array([])
    y_summary = np.array([])
    #print("\n ref rec>>",referenceRecords,"\n\t refSum>>",referenceRecordsSummary)
    for fileId in candidateRecordsDictionary.keys():
        if ((candidateRecordsDictionary[fileId] is not None) and (fileId in referenceRecords) and (fileId in referenceRecordsSummary)):
            for sentence in list(candidateRecordsDictionary[fileId]):
                x_record = assimilatePreTrainedEmbeddings(performWordSplit(sentence),x_record,modelId)
            #appendArray = np.array(list(candidateRecordsDecisionDictionary[fileId])).reshape(-1,1)
            appendArray = np.array(list(candidateRecordsDecisionDictionary[fileId]))
            #print("append>>",appendArray,"\tshape>>",appendArray.shape,"file>>",fileId)
            if appendArray.shape[1] == 2:
                y_summary = populateArray(y_summary,appendArray)
                
    return(x_record,y_summary)        

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
        singleConv = Conv1D(filters=currFilter,kernel_size=eachFilter,padding='valid',activation='relu',strides=300)(embeddingLayer)
        #print("\n convolution layer>>>",singleConv)
        
        singleConv = MaxPool1D(pool_size = 1)(singleConv)
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
            
    if (np.sum(np.array(col0Votes),0) > 0):
        status = 0
    else:
        status = 1
    
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

def updateOutputXml(predictOutput,keyAttributes):
    
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
        TRIGGER_FEATURE.append(openConfigurationFile(configFile,TRIGGER_ATTRIBUTE))
        
    else:
        print("\n Invalid model requested, please try again")
        
listedPatientRecords = os.listdir(trainDataPath)
listedPatientRecords = sorted(listedPatientRecords)
trainRecords, testRecords, trainSummary, testSummary = train_test_split(listedPatientRecords,listedPatientRecords, test_size=0.25, random_state=0)
trainRecords, testRecords, trainSummary, testSummary = map(lambda recordList : sorted(recordList),list([trainRecords, testRecords, trainSummary, testSummary]))

featureRecordDimension = []
featureSummaryDimension = []
tier1CandidateRecordsDictionary = {}
tier1CandidateRecordsDecisionDictionary = {}
indexMapperDictionary = []
recordDictionary = []
positiveTrigger =[]
negativeTrigger=[]
fieldIndex=0

for filename in list(listedPatientRecords):
    #if filename in ['102.xml','109.xml']:
        fieldIndex += 1
        fileAddress = "".join([trainDataPath,"/",filename])
        doc = minidom.parse(fileAddress)
        descriptionText = doc.getElementsByTagName("TEXT")[0]
        visitRecord = descriptionText.firstChild.data
        decisionTree = doc.getElementsByTagName("TAGS")
        attributeExistStatus = populateAttributeStatus(decisionTree,ctAttributes)        
        
        tokenIndexMapper,sentenceRecord = extractVisitRecords(visitRecord)
        indexMapperDictionary.extend(tokenIndexMapper)
        recordDictionary.insert(fieldIndex, sentenceRecord) 
        
        ''' create document record index '''
        initializeRecordTokenizer = indexRecordTokens(tokenIndexMapper)
        wordIndexMapper = initializeRecordTokenizer.word_index
        
        ''' Isolate feature words from document records'''
        tier1DictionaryList = []
        tier1WordList=[]
        tier1TriggerDominatedSentiment = []
        for triggerIndex,triggerValue in enumerate(TRIGGER_FEATURE):
            decoyDictionaryList = []
            tier1WordList.extend(screenForTriggerFeatures(wordIndexMapper, decoyDictionaryList, triggerValue,triggerIndex))
            tier1DictionaryList = decoyDictionaryList
            '''
            for index,listEntity in enumerate(tier1DictionaryList):
                print("\n",index,"\t>>",listEntity)
            '''
            
            ''' Compare scored feature tokens for instance screening '''
            seedIndex=0
            if(len(tier1DictionaryList) > 0):
                VARIABLE_ATTRIBUTE_THRESHOLD = 0.6
                if filename in testRecords:
                    VARIABLE_ATTRIBUTE_THRESHOLD = 0.59
                    
                decoyDictionaryList = dict(filter(lambda itemPair : itemPair[1]> VARIABLE_ATTRIBUTE_THRESHOLD,dict(tier1DictionaryList[0]).items())).keys()
                sourceSentimentDictionary = dict(tier1DictionaryList[seedIndex])
                sourceSentiment = list(sourceSentimentDictionary.keys())
                rootTrigger = sourceSentiment[seedIndex]
                verfiySentimentBranching(tier1DictionaryList,sourceSentiment,seedIndex,tier1TriggerDominatedSentiment)
                
                ''' allow tokens which are branched from a similar > variable threshold'''
                if (sourceSentimentDictionary[rootTrigger] >= VARIABLE_ATTRIBUTE_THRESHOLD):
                    if rootTrigger not in tier1TriggerDominatedSentiment:
                        tier1TriggerDominatedSentiment.append(rootTrigger)
                ''' allow tokens of value > threshold to be used'''
                tier1TriggerDominatedSentiment = list(set(tier1TriggerDominatedSentiment) & set(decoyDictionaryList))

        #print("\t tier1TriggerDominatedSentiment>>",tier1TriggerDominatedSentiment)
        tier1RecordCandidateInstances=list()
        if len(tier1TriggerDominatedSentiment) > 0:
            triggerIndex=0
            isolateCrCandidateInstances(sentenceRecord, tier1TriggerDominatedSentiment, tier1RecordCandidateInstances,triggerIndex)
            tier1RecordCandidateInstances = list(set(tier1RecordCandidateInstances))
            
        if((len(tier1TriggerDominatedSentiment)==0) or (len(tier1RecordCandidateInstances)==0)):
            tier1RecordCandidateInstances = ['no abdomen surgery']
        
        tier1CandidateRecordsDictionary[filename] = tier1RecordCandidateInstances
        tier1CandidateRecordsDecisionDictionary[filename] = vectorizeDecisionStatus(tier1RecordCandidateInstances,attributeExistStatus[TRIGGER_ATTRIBUTE])


#printRecords(tier1CandidateRecordsDictionary,tier1CandidateRecordsDecisionDictionary)

featureRecordDimension.append(tier1CandidateRecordsDictionary)
featureSummaryDimension.append(tier1CandidateRecordsDecisionDictionary)

MODEL_SIZE = len(featureRecordDimension)        

''' proportion +ve/-ve training samples'''
positiveCases = []
negativeCases = []
for fileId in featureRecordDimension[0]:
    if fileId in trainRecords:
        ''' not met'''
        if np.sum(np.array(list(dict(featureSummaryDimension[0])[fileId]))[:,1]) == 0:
            negativeCases.append(fileId)
        else:
            positiveCases.append(fileId)

print("\n\t size of +ve cases>>",len(positiveCases),"\t -ve cases>>",len(negativeCases))                    

leaderCase,trailerCase,iterateLength,segmentLength = defineSampleRandomizationSize(positiveCases,negativeCases)
print("\n iterate length>>",iterateLength,"\t segment length>>",segmentLength)        

''' validation data'''
x_validationList = []
y_validationList = []
for modelIter in range(0,MODEL_SIZE):
    candidateRecordsDictionary = dict(featureRecordDimension[modelIter])
    candidateRecordsDecisionDictionary = dict(featureSummaryDimension[modelIter])
    
    print("max seq>>",MAX_SEQUENCE_LENGTH[modelIter])        
    
    EMBEDDING_MATRIX=np.zeros((1,EMBEDDING_DIM))
    x_validation,y_validation = findSentenceVectorEmbeddings(candidateRecordsDictionary, testRecords, candidateRecordsDecisionDictionary, testSummary,modelIter)
    print("\n validation EMBEDDING_MATRIX>>",EMBEDDING_MATRIX.shape)
    print("\n x_validation>>",x_validation.shape,"\t y_validation>>",y_validation.shape)
    EMBEDDING_MATRIX = np.delete(EMBEDDING_MATRIX,0,0)
    #print('Null word embeddings: %d' % np.sum(np.sum(EMBEDDING_MATRIX, axis=1) == 0))

    validationSentenceDimension = int(np.rint(x_validation.shape[0]/MAX_SEQUENCE_LENGTH[modelIter]))
    validationSequenceSpan = MAX_SEQUENCE_LENGTH[modelIter]
    validationFeatureDimension = EMBEDDING_DIM 
    x_validation = x_validation.reshape(validationSentenceDimension,validationSequenceSpan,validationFeatureDimension)
    print(" validation tensor shape::",x_validation.shape,"\t y::",y_validation.shape)
    
    x_validationList.append(x_validation)
    y_validationList.append(y_validation)        
    
''' fit training data on model '''
modelVotingList = []
for i in range(1):
    print("\n Iteration range>>",i)
    runIndex=0
    startIndex = runIndex
    while runIndex < iterateLength:
        print("\t INDEX>>",runIndex)
        caseBalNo = int(np.rint(segmentLength/2))
        #caseBalNo=0
        endIndex = startIndex+(segmentLength+caseBalNo)
        print("startIndex>>",startIndex,"endIndex>>",endIndex)
        if startIndex >= len(leaderCase):
            break
        sampleCases = leaderCase[startIndex:endIndex]
        print("size>>",len(sampleCases))
        sampleCases.extend(trailerCase)
        trainRecords = sampleCases
        trainSummary = trainRecords
        print("trainRecords>>",trainRecords,"trainSummary>>",trainSummary)
    
        scoreDictList=[]
        modelIter = 0
        for modelIter in range(MODEL_SIZE):
            scoreDict={}
            candidateRecordsDictionary = dict(featureRecordDimension[modelIter])
            candidateRecordsDecisionDictionary = dict(featureSummaryDimension[modelIter])
            
            EMBEDDING_MATRIX=np.zeros((1,EMBEDDING_DIM))
            x_train,y_train = findSentenceVectorEmbeddings(candidateRecordsDictionary, trainRecords, candidateRecordsDecisionDictionary, trainSummary,modelIter)
            #print("\n  train EMBEDDING_MATRIX>>",EMBEDDING_MATRIX.shape)
            print(" x_train>>",x_train.shape,"\t y_train>>",y_train.shape)
            EMBEDDING_MATRIX = np.delete(EMBEDDING_MATRIX,0,0)
            
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
            
            if not reset:
                x_start = 0
                print('total loss>>>',score)
                for fileId in candidateRecordsDecisionDictionary.keys():
                    if fileId in testSummary:
                        y_size = len(list(candidateRecordsDecisionDictionary[fileId]))
                        x_size = x_start + y_size
                        predictionScore = np.array([])
                        for predictIndex in range(x_start,x_size):
                            validationSequenceSpan = MAX_SEQUENCE_LENGTH[modelIter]
                            validationFeatureDimension = EMBEDDING_DIM 
                            testExample = x_validation[predictIndex].reshape(1,validationSequenceSpan,validationFeatureDimension)
                            probability = model.predict(testExample, verbose=2)
                            predictionScore = populateArray(predictionScore, np.array(probability[0]).reshape(1,2))
    
                        print("prior score>>",predictionScore,"\t>>",fileId,"\n\t>>",candidateRecordsDictionary[fileId])
                        if modelIter == 0:
                            #status = trannslateMinScoresAsVotes(predictionScore)           
                            status = translateScoresAsVotes(predictionScore)
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
    for fileId in testSummary:
        status = 0
        if fileId in voteDictionary.keys():
            print("\t",fileId,"::",voteDictionary[fileId])
            decoyVoteDictionary = dict(Counter(voteDictionary[fileId]))
            status = int(max(decoyVoteDictionary.items(),key=itemgetter(1))[0])
        decoyVoteList = []
        if fileId in predictOutput.keys():
            decoyVoteList = predictOutput[fileId]
        decoyVoteList.append(status)
        if index == (MODEL_SIZE-1):
            finalStatus = 0
            if np.sum(decoyVoteList) == MODEL_SIZE:
                finalStatus = 1
            predictOutput.update({fileId:finalStatus})
        else:
            predictOutput[fileId] = decoyVoteList

updateOutputXml(predictOutput,ctAttributes)

actualOutput = {}
testDimension = dict(featureSummaryDimension[0])
for fileId in testDimension:
    if fileId in testSummary:
        status = 1
        if np.sum(np.array(list(testDimension[fileId]))[:,1]) == 0:
            status = 0
        actualOutput[fileId] = status 
        
y_predict = []
y_actual = []
#if len(predictOutput) == len(actualOutput):
for fileId in predictOutput.keys():
    print("\t>>",fileId,"\t",predictOutput[fileId],"\t",actualOutput[fileId])
    value = predictOutput[fileId]
    if type(value) == list:
        value = 0
    y_predict.append(value)
    y_actual.append(actualOutput[fileId])

print(classification_report(y_actual, y_predict))

print(precision_score(y_actual, y_predict, average="macro"))
print(precision_score(y_actual, y_predict, average="micro"))
print(precision_score(y_actual, y_predict, average="weighted"))
print(precision_score(y_actual, y_predict, average=None))

print(recall_score(y_actual, y_predict, average="macro"))
print(recall_score(y_actual, y_predict, average="micro"))
print(recall_score(y_actual, y_predict, average="weighted"))
print(recall_score(y_actual, y_predict, average=None))

            