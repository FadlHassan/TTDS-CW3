import re
from nltk.stem import PorterStemmer
from xml.etree import cElementTree as ET
import math

stemmer = PorterStemmer()

xml_input = open("CW1collection/trec.5000.xml", encoding = 'utf8').read() #loading the TREC 5000 sample
input_root = ET.fromstring(xml_input)

stopWords = open("englishST.txt").readlines() #loading stop words
stopWords = [x.strip("\n") for x in stopWords]
connectingStopWords = [x for x in stopWords if x != "and" and x != "or" and x != "not"] #removing connectors from stop words

def preProcessing(document):
    tokens = re.sub(r"[^\w]+", " ", document).lower() #tokenisation
    tokenisation = tokens.split(" ")
    stoppedWords = [word for word in tokenisation if word not in stopWords and word != ""] #stop word removal
    terms = [stemmer.stem(word) for word in stoppedWords] #stemming
    return terms

def invertedIndex(root):
    documents = {}
    for page in list(root):
        num = page.find('DOCNO').text
        headline = page.find('HEADLINE').text
        content = page.find('TEXT').text
        documents[num] = headline + " " + content
    total_num = len(documents) 
    invertedIndex = {} #dictionary of positional inverted index
    for num in documents:
        doc = documents[num]
        terms = preProcessing(doc)
        count = 0
        #store inverted index
        for term in terms:
            if term not in invertedIndex:
                invertedIndex[term] = {}
            if num not in invertedIndex[term]:
                invertedIndex[term][num] = []
                invertedIndex[term][num].append(str(count))
            else:
                invertedIndex[term][num].append(str(count))
            count += 1
    f1 = "index.txt"
    for term in invertedIndex:
        with open(f1, "a+") as file:
            file.write(term + ": " + str(len(invertedIndex[term])) + "\n")
            for doc in invertedIndex[term]:
                file.write("\t" + doc + ": " + ",".join(invertedIndex[term][doc]) + "\n")
    return invertedIndex,total_num

def booleanSearch(l1, l2, andFlag, orFlag, notFlag, totalList):
    queryList = []
    setL1 = set(l1)
    setL2 = set(l2)
    setTotalList = set(totalList)
    if andFlag == True and notFlag == False:
        queryList = [x for x in l1 if x in l2]
    elif andFlag == True and notFlag == True:
        queryList = [x for x in l1 if x not in l2]
    elif orFlag == True and notFlag == False:
        queryList = list(setL1.union(setL2))
    elif orFlag == True and notFlag == True:
        queryList = list(setL1.union(setTotalList.difference(setL2))) 
    return queryList

def phraseAndProximitySearch(t1,t2,dictionary,distance):
    dict1 = dictionary[t1]
    dict2 = dictionary[t2]
    pages = dict1.keys() & dict2.keys()
    queryList = []
    for page in pages:
        flag = True
        l1 = [int(x) for x in dict1[page]]
        l2 = [int(x) for x in dict2[page]]
        for l in l1:
            if distance > 1: #proximity search
                for i in list(range(-1*distance, distance+1)):
                    if (l + i) in l2:
                        flag = False
                        queryList.append(int(page))
                if not flag:
                    break
            else: #phrase search
                if l+1 in l2:
                    flag = False
                    queryList.append(int(page))
                    break
                if not flag:
                    break
    return queryList

def queryPreProcessing(query): #preprocessing the boolean queries
    chars = '[!"$%&\()*+,-./:;<=>?@[\\]^_{|}~]+|\r|\t'
    query = re.sub(chars, " ", query).lower().split(" ")
    stopWordRemoval = [word for word in query if word not in connectingStopWords and word !=""]
    stemming = [stemmer.stem(word) for word in stopWordRemoval]
    return stemming

def queryProcessing(query,totalNumberOfDocuments,invertedIndex):
    def wordIndexing(word,dictionary): #indexing for when the query has only one word
        queryList = []
        for key in dictionary[word]:
            queryList.append(int(key))
        return queryList
    
    preprocessedQuery = queryPreProcessing(query)
    andFlag = False
    orFlag = False
    notFlag = False
    l1 = []
    l2 = []
    for query in range(len(preprocessedQuery)):
        if preprocessedQuery[query] == "and" and "not" not in preprocessedQuery: #A and B
            andFlag = True
            l1 = preprocessedQuery[:query]
            l2 = preprocessedQuery[query+1:]
        elif preprocessedQuery[query] == "or" and "not" not in preprocessedQuery: #A or B
            orFlag = True
            l1 = preprocessedQuery[:query]
            l2 = preprocessedQuery[query+1:]
        elif preprocessedQuery[query] == "not" and query == 0: 
            notFlag = True
            if "and" in preprocessedQuery:#A and not B
                andIndex = preprocessedQuery.index("and")
                andFlag = True
                l1 = preprocessedQuery[andIndex+1:]
                l2 = preprocessedQuery[1:andIndex]
            elif "or" in preprocessedQuery:#A and or B
                orIndex = preprocessedQuery.index("or")
                orFlag = True
                l1 = preprocessedQuery[orIndex+1:]
                l2 = preprocessedQuery[1:orIndex]
            else:
                l1 = preprocessedQuery[1:]
        elif preprocessedQuery[query] == "not" and query != 0:
            notFlag = True
            if preprocessedQuery[query - 1] == "and": #not A and B
                andFlag = True
                l1 = preprocessedQuery[:query - 1]
                l2 = preprocessedQuery[query+1:]
            elif preprocessedQuery[query - 1] == "or": #not A or B
                orFlag = True
                l1 = preprocessedQuery[:query - 1]
                l2 = preprocessedQuery[query+1:]
            else: #not A
                l1 = preprocessedQuery[1:]
    totalList = [x for x in range(1,len(invertedIndex))]
    totalSet = set(totalList)
    if andFlag == False and orFlag == False and notFlag == False:
        if len(preprocessedQuery) == 1:
            queryList = wordIndexing(preprocessedQuery[0],invertedIndex)
        elif len(preprocessedQuery) == 2: 
            queryList = phraseAndProximitySearch(preprocessedQuery[0],preprocessedQuery[1],invertedIndex,1)
        elif len(preprocessedQuery) == 3: 
            distance = int(preprocessedQuery[0].strip("#"))
            queryList = phraseAndProximitySearch(preprocessedQuery[1],preprocessedQuery[2],invertedIndex,distance)
    else:
        if andFlag == False and orFlag == False and notFlag == True:
            if len(l1) == 1:
                tempQueryList = wordIndexing(l1[0],invertedIndex)
                tempQuerySet = set(tempQueryList)
                queryList = list(tempQuerySet.difference(totalSet))
            elif len(l1) == 2:
                tempQueryList = phraseAndProximitySearch(l1[0],l1[1],invertedIndex,1)
                tempQuerySet = set(tempQueryList)
                queryList = list(tempQuerySet.difference(totalSet))
        else:
            if len(l1) == 1:
                tempQueryList = wordIndexing(l1[0],invertedIndex)
            else:
                tempQueryList = phraseAndProximitySearch(l1[0],l1[1],invertedIndex,1)
            if len(l2) == 1:
                tempQueryList2 = wordIndexing(l2[0],invertedIndex)
            else:
                tempQueryList2 = phraseAndProximitySearch(l2[0],l2[1],invertedIndex,1)
            queryList = booleanSearch(tempQueryList,tempQueryList2,andFlag,orFlag,notFlag,totalList)
    return sorted(queryList)

def processingBooleanQueries(totalNumberOfDocuments,invertedIndex):
    file1 = "results.boolean.txt"
    booleanQueries = open("CW1collection/queries.boolean.txt").readlines() #loading boolean queries
    queryTerms = []
    for i in range(len(booleanQueries)):
        words = " ".join(booleanQueries[i].split(" ")[1:])
        words = words.strip("\n")
        results = queryProcessing(words,totalNumberOfDocuments,invertedIndex) #processing the queries
        for j in results:
            with open(file1, "a+") as file:
                file.write(str(i+1) + "," + str(j) + "\n") #writing results to file

def processingRankQueries(totalNumberOfDocuments,invertedIndex):
    file1 = "results.ranked.txt"
    rankQueries = open("CW1collection/queries.ranked.txt").readlines() #loading ranked queries
    queryTerms = []
    for query in rankQueries:
        queryTerms.append(preProcessing(query))
    documentDictionary = {}
    for query in range(len(queryTerms)):
        documentDictionary[query] = {}
        for q in queryTerms[query][1:]:
            documentNumber = query + 1
            documentList = queryProcessing(q,totalNumberOfDocuments,invertedIndex) #processing the queries
            df = len(invertedIndex[q])
            for document in documentList:
                if document not in documentDictionary[query]:
                    documentDictionary[query][document] = 0
                tf = len(invertedIndex[q][str(document)])
                score = (1 + math.log10(tf)) * math.log10(totalNumberOfDocuments/df) #calculating TFIDF score
                documentDictionary[query][document] += score
    file1 = "results.ranked.txt"
    for query in range(len(queryTerms)):
        for rank in sorted(documentDictionary[query].items(), key = lambda x:(x[1],x[0]) , reverse = True)[:150]:
            with open(file1, "a+") as file:
                file. write(str(query+1) + "," + str(rank[0]) + "," + "{:.4f}".format(rank[1]) + "\n") #writing the top 150 results to file

#function calls to execute the program
invertedIndex,totalNumberOfDocuments = invertedIndex(input_root)
# processingBooleanQueries(totalNumberOfDocuments,invertedIndex)
processingRankQueries(totalNumberOfDocuments,invertedIndex)
    





                




