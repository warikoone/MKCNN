'''
Created on Mar 30, 2018

@author: neha
'''

import numpy as np
import re
import nltk
from nltk import word_tokenize
from nltk.corpus import wordnet
from operator import itemgetter,attrgetter
from array import array
from gensim.models import KeyedVectors


EMBEDDING_FILE = "/home/neha/Disk_R/Bio_NLP/N2C2_Task/Data/embedding/GoogleNews-vectors-negative300.bin"
#EMBEDDING_FILE = "/home/neha/Disk_R/Bio_NLP/N2C2_Task/Data/embedding/wiki.en.vec"

a = np.array([1,2,3])
print("a>>",a.shape)
a= np.append(a, [6,6,8], 0)

print("a>>",a)

string = " while "

print('>>>>',not(string.isspace()))

a = nltk.pos_tag(word_tokenize('The patient appears awake , alert , speech clear fluent receptive language function essentially intact .'))
for token in a:
    b = str(token)
    print("st>>",b)
    b = re.sub('^\(|\)$','',b)
    print("new >",b)
    tagMatcher = re.search('\'(VB|NN).{0,1}\'', str(token))
    if tagMatcher:
        print("yes")
        print(">>",tagMatcher.group(0))

l1 = [0.10679435174965578,0.10654161953891239,0.10605976558830403]
d = 0.10679435174965578
d = np.around(d, decimals=2)
print(">>>",d)
print("mean>>",np.mean(l1),'\t>>',np.average(l1))
#d1 = {1:'z',2:'b',3:'c',6:'g'}
d1 = {1:0.8,2:1.0,3:0.5,6:0.1}
print(">>",d1)
d1 = dict(sorted(d1.items(),key=itemgetter(0)))
l2 = list(d1.values())
print("sorted>>",np.percentile(l2,80))
for val in d1.items():
    print("\nchk this>>>",val[1])
b = dict(filter((lambda x: (x[1]>0.2)),d1.items()))
print(b)

str1 = 'The patient appears awake , alert , speech clear fluent receptive language function essentially intact .'
pattern = ['language','fluent','awake','intact']
tokens = list(word_tokenize(str1))
start=[];end=[]
for samplePattern in pattern:
    for index,word in enumerate(tokens, start=0):
        #print(">>word",word)
        m = re.match(samplePattern,str(word), flags=re.RegexFlag.IGNORECASE)
        if m:
            currIndex = index-4
            if currIndex < 0:
                currIndex = index
            start.append(currIndex)
            currIndex = index+4
            if currIndex > len(tokens)-1:
                currIndex = index
            end.append(currIndex)
print("start before>>",start)            
print("end before>>",end)
start = list(filter(lambda index : index>=0,start))
end = list(filter(lambda index : index<len(tokens),end))
print("start aftr>>",start)
print("end aftr>>",end)
startIndex = min(start)
endIndex = max(end)
print("final token>>",tokens[startIndex:endIndex+1])



print(nltk.corpus.wordnet.synsets('FISTULA'))
pattern = 'BIOPSY'
patternSynonyms = nltk.corpus.wordnet.synsets(pattern)
print(">>>>>",patternSynonyms)
hypernymList = []
for wordSynonym in patternSynonyms:
    wordLemma = wordSynonym.lemma_names()[0]
    if(re.match(pattern, wordLemma, flags = re.RegexFlag.IGNORECASE)):
        for hypernymWord in wordSynonym.hypernyms(): 
            hypernymList.append(hypernymWord.lemma_names()[0])

print(">>>",hypernymList)

'''
word2Vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE,binary=True)
if 'ASA' in word2Vec:
    print("yes")
else:
    print("no wasn't present")

print("similarity>>",word2Vec.similarity('ASA','aspirin'))    
'''

text = "November 17, 2083 RACINE, MAINE Name:  Quiana Justus NIHC #: 981-40-48 PROGRESS"
word='asa'
m = re.finditer(word, text, flags=re.RegexFlag.IGNORECASE)
for ma in m:
    print("\n\t>>",ma.group())













