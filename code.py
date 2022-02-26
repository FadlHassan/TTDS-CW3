import re
from nltk.stem import PorterStemmer
import random
import pandas as pd
import numpy as np 
import math
from gensim.models.ldamodel import LdaModel
from gensim.corpora.dictionary import Dictionary
from collections import Counter
import string
from scipy.sparse import dok_matrix
import csv
import sklearn
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

stemmer = PorterStemmer()
stopWords = open("englishST.txt").readlines() #loading stop words
stopWordsList = [x.replace("\n", "") for x in stopWords]

systemResults = open('system_results.csv', 'r')
Qrels = open('qrels.csv','r')
system_number,query_number,doc_number,rank_of_doc,score = ([] for i in range(5))
query_id,doc_id,relevance = ([] for i in range(3))
system_results = []
qrels = []
#reading the system_results file
file = csv.DictReader(systemResults)
for col in file:
    system_number.append(col['system_number'])
    query_number.append(col['query_number'])
    doc_number.append(col['doc_number'])
    rank_of_doc.append(col['rank_of_doc'])
    score.append(col['score'])
system_number = [ int(x) for x in system_number ]
system_results.append(system_number)
query_number = [ int(x) for x in query_number ]
system_results.append(query_number)
doc_number = [ int(x) for x in doc_number ]
system_results.append(doc_number)
rank_of_doc = [ int(x) for x in rank_of_doc ]
system_results.append(rank_of_doc)
score = [ int(x) for x in score ]
system_results.append(score)

#reading the qrels file
file = csv.DictReader(Qrels)
for col in file:
    query_id.append(col['query_id'])
    doc_id.append(col['doc_id'])
    relevance.append(col['relevance'])
query_id = [ int(x) for x in query_id ]
qrels.append(query_id)
doc_id = [ int(x) for x in doc_id ]
qrels.append(doc_id)
relevance = [ int(x) for x in relevance ]
qrels.append(relevance)

system_result = np.array(system_results).T.tolist()
qrels = np.array(qrels).T.tolist()

punctuations = re.compile(f'[{string.punctuation}]')
trainData = pd.read_csv("train_and_dev.tsv",header = None,sep = "\t",quoting = csv.QUOTE_NONE)
test = pd.read_csv("test.tsv",header = None,sep = "\t",quoting = csv.QUOTE_NONE)

def ldaModel(quranCorpus,ntCorpus,otCorpus): #calculates the LDA scores
    corpusList = [quranCorpus,ntCorpus,otCorpus]
    totalCorpus = quranCorpus + ntCorpus + otCorpus
    listOfDics = []
    corpusDic = Dictionary(totalCorpus)
    corpusDic.filter_extremes(no_below = 50, no_above = 0.1)
    corpus = [corpusDic.doc2bow(c) for c in totalCorpus]
    lda = LdaModel(corpus , num_topics = 20, id2word = corpusDic, random_state = 1)
    for c in corpusList:
        dic = Dictionary(c)
        dic.filter_extremes(no_below=50, no_above=0.1)
        c1 = [dic.doc2bow(text) for text in c]
        cTopics = lda.get_document_topics(c1)
        tempDic = {}
        for doc in cTopics:
            for t in doc:
                if t[0] not in tempDic:
                    tempDic[t[0]] = t[1]
                else:
                    tempDic[t[0]] += t[1]
        listOfDics.append(tempDic)
    topicDicQuran = listOfDics[0]
    topicDicNT = listOfDics[1]
    topicDicOT = listOfDics[2]
    for i, j in topicDicQuran.items():
        topicDicQuran[i] = j / len(quranCorpus)
    for i, j in topicDicNT.items():
        topicDicNT[i] = j / len(ntCorpus)
    for i, j in topicDicOT.items():
        topicDicOT[i] = j / len(otCorpus)
    return lda , topicDicQuran, topicDicNT, topicDicOT

def preprocessForSVM(data): #preprocesses data for task classiciation by removing punctuations and converting to lower case
    vocab = set([])
    docs =[]
    cats = []
    for index,row in data.iterrows():
        corpus,text = row[0] , row[1]
        words = punctuations.sub("",text).lower().split()
        for w in words:
            vocab.add(w)
        docs.append(words)
        cats.append(corpus)
    return docs,cats,vocab

def splitTrainAndDev(categories,preprocessedData): #splits the data into train(90%) and dev(10%) sets
    trainData = []
    trainCategories = []
    devData = []
    devCategories = []
    random.seed(0)
    devIndex = [random.randint(0, len(preprocessedData)) for i in range(round(len(preprocessedData) * 0.1) )]
    for i in devIndex:
        devData.append(preprocessedData[i])
        devCategories.append(categories[i])
    trainIndex = [i for i in range(len(preprocessedData)) if i not in devIndex]
    for i in trainIndex:
        trainData.append(preprocessedData[i])
        trainCategories.append(categories[i])
    return trainData,trainCategories,devData,devCategories,devIndex

def calculateMI(N,n00,n01,n10,n11):
    mi = n11 / N * math.log2(float(N * n11) / float((n10 + n11) * (n01 + n11))) + n01 / N * math.log2(float(N * n01) / float((n00 + n01) * (n01 + n11))) + n00 / N * math.log2(float(N * n00) / float((n00 + n01) * (n00 + n10)))
    return mi

def scoreCorpus(totalDic,corpusDic,quranCorpus,ntCorpus,otCorpus): #returns the MI and chi-square scores of each corpus
    MI = {}
    chiSquare = {}
    if corpusDic == ntDic:
        X = Counter(quranDic)
        Y = Counter(otDic)
        xyDic = dict(X + Y)
        lenCorpus = len(ntCorpus)
        lenOther = len(quranCorpus) + len(otCorpus)
    elif corpusDic == quranDic:
        X = Counter(otDic)
        Y = Counter(ntDic)
        xyDic = dict(X + Y)
        lenCorpus = len(quranCorpus)
        lenOther = len(ntCorpus) + len(otCorpus)
    else:
        X = Counter(quranDic)
        Y = Counter(ntDic)
        xyDic = dict(X + Y)
        lenCorpus = len(otCorpus)
        lenOther = len(ntCorpus) + len(quranCorpus)
    N = lenCorpus + lenOther
    for key in totalDic.keys():
        if key in corpusDic.keys():
            n11 = corpusDic[key]
            n01 = lenCorpus - n11
            if key not in xyDic:
                n00 = lenOther
                n10 = 0
                mi = calculateMI(N,n00,n01,n10,n11)
            else:
                n10 = xyDic[key]
                n00 = lenOther - n10
                mi = calculateMI(N,n00,n01,n10,n11)
        else:
            n01 = lenCorpus
            n11 = 0
            n10 = xyDic[key]
            n00 = lenOther - n10
            mi = n01/N*math.log2(float(N*n01) / float((n00+n01)*(n01+n11))) + n10/N*math.log2(float(N*n10) / float((n10+n11)*(n00+n10))) + n00/N*math.log2(float(N*n00) / float((n00+n01)*(n00+n10)))
        MI[key] = mi

        chi = ((n11 + n10 + n01 + n00) * math.pow((n11 * n00 - n10 * n01),2)) / ((n11 + n01) * (n11 + n10) * (n10 + n00) * (n01 + n00))
        chiSquare[key] = chi
    return MI,chiSquare

def incorrect(predVals,trueVals): #returns a list of incorrect predictions
    incorrectVals = []
    for i in range(len(predVals)):
        if predVals[i]  != trueVals[i]:
            incorrectVals.append(i)
    return incorrectVals

def precisionRecallF1(y,yHat,category,cat2id): #calculates the precission,recall and f1 scores
    truePositive = np.sum(np.logical_and(np.equal(y, cat2id[category]), np.equal(yHat, cat2id[category])))
    falsePositive = np.sum(np.logical_and(np.not_equal(y, cat2id[category]), np.equal(yHat, cat2id[category])))
    trueNegative = np.sum(np.logical_and(np.not_equal(y, cat2id[category]), np.not_equal(yHat, cat2id[category])))
    falseNegative = np.sum(np.logical_and(np.equal(y, cat2id[category]), np.not_equal(yHat, cat2id[category])))
    precision = round((truePositive/(truePositive+falsePositive)),3)
    recall = round((truePositive/(truePositive+falseNegative)),3)
    f1 = (2 * precision * recall) / (precision+recall)
    return precision , recall , f1

def accuracy(y,yHat,cat2id): #calculates the model accuracies
    precisionQuran,recallQuran,f1Quran = precisionRecallF1(y,yHat,"Quran",cat2id)
    precisionOT,recallOT,f1OT = precisionRecallF1(y,yHat,"OT",cat2id)
    precisionNT,recallNT,f1NT = precisionRecallF1(y,yHat,"NT",cat2id)
    precissionMacro = round((precisionQuran + precisionOT + precisionNT) / 3 , 3)
    precissionMacro = round((precisionQuran + precisionOT + precisionNT) / 3 , 3)
    recallMacro = round((recallQuran + recallOT + recallNT) / 3 , 3)
    f1Macro = round((f1Quran + f1OT + f1NT) / 3 , 3)
    accuracies = [str(precisionQuran), str(recallQuran), str(f1Quran),
                  str(precisionNT), str(recallNT), str(f1NT),
                  str(precisionOT), str(recallOT), str(f1OT),
                  str(precissionMacro), str(recallMacro), str(f1Macro)]
    return accuracies

def preprocessCorpus(data): #splits the corpora into Quran, NT and OT
    quranDic = {}
    newTestamentDic = {}
    oldTestamentDic = {}
    quranCorpus = []
    newTestamentCorpus = []
    oldTestamentCorpus = []
    for index,row in data.iterrows():
        corpus,text = row[0] , row[1]
        preprocessedTerms = preProcessing(text)
        if corpus == "OT":
            oldTestamentCorpus.append(preprocessedTerms)
            for t in set(preprocessedTerms):
                if t not in oldTestamentDic:
                    oldTestamentDic[t] = 1
                else:
                    oldTestamentDic[t] += 1
        elif corpus == "Quran":
            quranCorpus.append(preprocessedTerms)
            for t in set(preprocessedTerms):
                if t not in quranDic:
                    quranDic[t] = 1
                else:
                    quranDic[t] += 1
        else:
            newTestamentCorpus.append(preprocessedTerms)
            for t in set(preprocessedTerms):
                if t not in newTestamentDic:
                    newTestamentDic[t] = 1
                else:
                    newTestamentDic[t] += 1
    return quranDic,newTestamentDic,oldTestamentDic,quranCorpus,newTestamentCorpus,oldTestamentCorpus

def preProcessing(document): #preprocesses data (same as coursework 1)
    tokens = re.sub(r"[^\w]+", " ", document).lower() #tokenisation
    tokenisation = tokens.split(" ")
    stoppedWords = [word for word in tokenisation if word not in stopWordsList and word != ""] #stop word removal
    terms = [stemmer.stem(word) for word in stoppedWords] #stemming
    return terms

def Eval(system_results,qrels): #calculates the ir eval values for the different metrics
    irEval = pd.DataFrame(columns=["system_number","query_number","P@10","R@50","r-precision","AP","nDCG@10","nDCG@20"])
    for num in range(1,7):
        l = [i for i in system_results if i[0] ==num]
        qNo = [i[1] for i in l]
        qrDic = dict(Counter(qNo))
        subSystemResults = [i for i in system_results if i[0] ==num]
        irEvalDF = pd.DataFrame(columns=["system_number","query_number"])
        irEvalDF["query_number"] = np.array(range(1,11))
        irEvalDF["system_number"] = np.ones((10,1))*num
        P_10,R_50,r_precision,AP,nDCG_10,nDCG_20 = ([] for i in range(6))
        index = 0
        for i in range(1,11):
            q = [j[1] for j in qrels if i[0] == i]

            P10Result = [j[2] for j in range subSystemResults[index:index+10]]
            P10Result = set(P10Result)
            P_10.append(len(P10Result & set(q)) / 10)

            R50Result = [j[2] for j in range subSystemResults[index:index+50]]
            R50Result = set(R50Result)
            R_50.append(len(R50Result & set(q)) / len(q))

            rPresicionResult = [j[2] for j in range subSystemResults[index:index + len(q)]]
            rPresicionResult = set(rPresicionResult)
            r_precision.append(len(rPresicionResult & set(q)) / len(q))

            APresult = [j for j in range subSystemResults[index:index+qrDic[i]]
            tempList = []
            for doc in q:
                apr = [j[2] for j in APresult]
                if doc in apr:
                    tempList.append(int([j[3] for j in APresult if j[2] == doc][0]))
            c = 1
            total = 0
            for j in sorted(tempList):
                total += c / j
                c += 1
            AP.append(total/len(q))
            
            DCG10 = 0
            DCG10result = [j for j in range subSystemResults[index:index+10]]
            DCG20 = 0
            DCG20result = [j for j in range subSystemResults[index:index+20]]
            dcg = [10,20]
            for val in dcg:
                if val == 10:
                    for doc in q:
                        if doc in [j[2] for j in DCG10result]:
                            temp = [j[2] for j in qrels if j[0] == i and i[1] == doc][0]
                            if int([j[3] for j in DCG10result if j[2] == doc][0]) == 1
                                DCG10 += temp
                            else:
                                DCG10 += temp/math.log2(int([j[3] for j in DCG10result if j[2] == doc][0]))
                    relList = sorted([j[2] for j in qrels if j[0] == i]),reverse=True)
                    kDCG10 = relList[0]
                    if len(relList) <= 10:
                        for k in range(1,len(relList)):
                            kDCG10 += relList[k] / math.log2(k+1)
                    else:
                        for k in range(1,10):
                            kDCG10 += relList[k] / math.log2(k+1)
                    nDCG_10.append(DCG10/kDCG10)
                else:
                    for doc in q:
                        if doc in [j[2] for j in DCG10result]:
                            temp = [j[2] for j in qrels if j[0] == i and i[1] == doc][0]
                            if int([j[3] for j in DCG10result if j[2] == doc][0]) == 1:
                                DCG20 += temp
                            else:
                                DCG20 += temp/math.log2(int([j[3] for j in DCG10result if j[2] == doc][0]))
                    relList = sorted([j[2] for j in qrels if j[0] == i]),reverse=True)
                    kDCG20 = relList[0]
                    if len(relList) <= 20:
                        for k in range(1,len(relList)):
                            kDCG20 += relList[k] / math.log2(k+1)
                    else:
                        for k in range(1,20):
                            kDCG20 += relList[k] / math.log2(k+1)
                    nDCG_20.append(DCG20/kDCG20)
            index += qrDic[i] 
        irEvalDF["P@10"] = np.array(P_10)
        irEvalDF["R@50"] = np.array(R_50)
        irEvalDF["r-precision"] = np.array(r_precision)
        irEvalDF["AP"] = np.array(AP)
        irEvalDF["nDCG@10"] = np.array(nDCG_10)
        irEvalDF["nDCG@20"] = np.array(nDCG_20)
        irEvalDF.index = irEvalDF.index + 1
        irEvalDF.loc["mean"] = irEvalDF.mean()
        irEval = pd.concat([irEval,irEvalDF],axis = 0)
    return irEval

def writeToEval(system_results,qrels): #creates the ir_eval.csv file
    irEval = Eval(system_results,qrels)
    fileName = "ir_eval.csv"
    with open(fileName,"w") as file:
        file.write("system_number,query_number,P@10,R@50,r-precision,AP,nDCG@10,nDCG@20" + "\n")
    for index, row in irEval.iterrows():
        with open(fileName,"a") as file:
            file.write(str(int(row["system_number"])) + "," + str(index) + "," + "{:.3f}".format(row["P@10"]) + "," "{:.3f}".format(row["R@50"]) + "," + "{:.3f}".format(row["r-precision"]) + "," "{:.3f}".format(row["AP"]) + "," + "{:.3f}".format(row["nDCG@10"]) + "," + "{:.3f}".format(row["nDCG@20"]) + "\n")

def makeWord2id(vocab): #creates the words to id dictionary
    word2id = {}
    for wId, w in enumerate(vocab):
        word2id[w] = wId
    return word2id

def makeCat2id(categories): #creates the categories to id dictionary
    cat2id = {}
    for cId, c in enumerate(set(categories)):
        cat2id[c] = cId
    return cat2id

writeToEval(system_results,qrels)

quranDic,ntDic,otDic,quranCorpus,ntCorpus,otCorpus = preprocessCorpus(trainData)

def calculateMIandCHI(): #prints the top 10 MI and chi-square scores
    X , Y , Z = Counter(quranDic) , Counter(ntDic) , Counter(otDic)
    total = dict(X + Y + Z)
    quranMI,quranCHI = scoreCorpus(total,quranDic,quranCorpus,ntCorpus,otCorpus)
    ntMI,ntCHI = scoreCorpus(total,ntDic,quranCorpus,ntCorpus,otCorpus)
    otMI,otCHI = scoreCorpus(total,otDic,quranCorpus,ntCorpus,otCorpus)
    print("-----Quran MI------ ")
    print(sorted(quranMI.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)[:10])
    print("-----Quran CHI------ ")
    print(sorted(quranCHI.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)[:10])
    print("-----NT MI------ ")
    print(sorted(ntMI.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)[:10])
    print("-----NT CHI------ ")
    print(sorted(ntCHI.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)[:10])
    print("-----OT MI------ ")
    print(sorted(otMI.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)[:10])
    print("-----OT CHI------ ")
    print(sorted(otCHI.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)[:10])

ldaVal,topicDicQuran,topicDicNT,topicDicOT = ldaModel(quranCorpus,ntCorpus,otCorpus)

def LDA(): #prints the LDA topic scores
    topicQuran = sorted(topicDicQuran.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)[:5]
    topicNT = sorted(topicDicNT.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)[:5]
    topicOT = sorted(topicDicOT.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)[:5]
    print("topic_id: " + str(topicQuran[0][0]) + ", score: " + str(topicQuran[0][1]))
    print(ldaVal.print_topic(topicQuran[0][0]))
    print("topic_id: " + str(topicNT[0][0]) + ", score: " + str(topicNT[0][1]))
    print(ldaVal.print_topic(topicNT[0][0]))
    print("topic_id: " + str(topicOT[0][0]) + ", score: " + str(topicOT[0][1]))
    print(ldaVal.print_topic(topicOT[0][0]))
    for i in range(5):
        print("Top "+str(i)+" id in Quran "+str(topicQuran[i][0]))
        print("Top "+str(i)+" id in NT "+str(topicNT[i][0]))
        print("Top "+str(i)+" id in OT "+str(topicOT[i][0]))

calculateMIandCHI()
LDA()
#Preprocessing data for the models
docs,cats,vocab = preprocessForSVM(trainData)
testDocs,testCats,testVocab = preprocessForSVM(test)
#Creating the train and dev datasets
trainingData,trainingCategories,develData,develCategories,devIndex = splitTrainAndDev(cats,docs)
word2id = makeWord2id(vocab)
cat2id = makeCat2id(cats)

def generateBOW(preprocessedData): #Generates the BOW values
    matrix = (len(preprocessedData), len(word2id) + 1)
    outOfVocab = len(word2id)
    M = dok_matrix(matrix)
    for docId,doc in enumerate(preprocessedData):
        for d in doc:
            M[docId, word2id.get(d,outOfVocab)] += 1
    return M

def convertToBOW(xData,category): #Converts given data to BOW
    X = generateBOW(xData)
    Y = [cat2id[cat] for cat in category]
    return X,Y

#Create BOW values for the train,dev and test sets
xTrain , yTrain = convertToBOW(trainingData,trainingCategories)
xDev,yDev = convertToBOW(develData,develCategories)
xTest,yTest = convertToBOW(testDocs,testCats)

def svmModel(c,k): #Train SVM model 
    model = svm.SVC(C=c,kernel = k)
    model.fit(xTrain,yTrain)
    yTrainPred = model.predict(xTrain)
    yDevPred = model.predict(xDev)
    yTestPred = model.predict(xTest)
    trainError = accuracy(yTrain,yTrainPred,cat2id)
    devError = accuracy(yDev,yDevPred,cat2id)
    testError = accuracy(yTest,yTestPred,cat2id)
    return yDevPred,trainError,devError,testError

def improveAccuracy(): #Decision Tree Classifier Model
    model = DecisionTreeClassifier(random_state=0)
    model.fit(xTrain,yTrain)
    yTrainPred = model.predict(xTrain)
    yDevPred = model.predict(xDev)
    yTestPred = model.predict(xTest)
    trainError = accuracy(yTrain,yTrainPred,cat2id)
    devError = accuracy(yDev,yDevPred,cat2id)
    testError = accuracy(yTest,yTestPred,cat2id)
    print(trainError)
    print(devError)
    print(testError)

def improveAccuracyLR(): #Logistic Regression on OneVsRestClassifierModel
    model = OneVsRestClassifier(LogisticRegression(random_state = 0))
    clf = model.fit(xTrain,yTrain)
    clf.fit(xTrain,yTrain)
    yTrainPred = clf.predict(xTrain)
    yDevPred = clf.predict(xDev)
    yTestPred = clf.predict(xTest)
    trainError = accuracy(yTrain,yTrainPred,cat2id)
    devError = accuracy(yDev,yDevPred,cat2id)
    testError = accuracy(yTest,yTestPred,cat2id)
    print(trainError)
    print(devError)
    print(testError)

yDevPredBase,trainErrorBase,devErrorBase,testErrorBase = svmModel(1000,"linear") #Baseline Model
svmModel(100,"linear")
svmModel(2000,"linear")
yDevPredImp,trainErrorImp,devErrorImp,testErrorImp = svmModel(1000,"rbf") #Most Improved Model

improveAccuracy()
improveAccuracyLR()

def writeClassification(): #Write to classification.csv
    with open("classification.csv","w") as file:
        file.write("system,split,p-quran,r-quran,f-quran,p-ot,r-ot,f-ot,p-nt,r-nt,f-nt,p-macro,r-macro,f-macro" + "\n")
        file.write("baseline,train," + ",".join(trainErrorBase) + "\n")
        file.write("baseline,dev," + ",".join(devErrorBase) + "\n")
        file.write("baseline,test," + ",".join(testErrorBase) + "\n")
        file.write("improved,train," + ",".join(trainErrorImp) + "\n")
        file.write("improved,dev," + ",".join(devErrorImp) + "\n")
        file.write("improved,test," + ",".join(testErrorImp) + "\n")

def countIncorrect():#Count incorrect predictions by the baseline model
    incorrectVals = incorrect(yDevPredBase,yDev)
    for i in incorrectVals[:3]:
        print(docs[devIndex[i]])
        print("predicted category: " + list(cat2id.keys())[list(cat2id.values()).index(yDevPredBase[i])])
        print("true category: " + list(cat2id.keys())[list(cat2id.values()).index(yDev[i])])

writeClassification()
countIncorrect()
print(sum(x.count("muhammad") for x in quranCorpus))
print(sum(x.count("muhammad") for x in ntCorpus))
print(sum(x.count("muhammad") for x in otCorpus))
""" print("---------")
print(sum(x.count("deliv") for x in quranCorpus))
print(sum(x.count("deliv") for x in ntCorpus))
print(sum(x.count("deliv") for x in otCorpus))
print("---------")
print(sum(x.count("set") for x in quranCorpus))
print(sum(x.count("set") for x in ntCorpus))
print(sum(x.count("set") for x in otCorpus)) """




                