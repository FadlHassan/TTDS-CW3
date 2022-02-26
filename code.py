import os
import csv
import math
import sys, re
import numpy as np
from scipy.sparse import dok_matrix
import random
from collections import Counter
from stemming.porter2 import stem
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel
# from sklearn.svm import LinearSVC
from sklearn.svm import SVC, LinearSVC

# ----------- METHODS START -----------

# Method for case folding
def caseFolding(text):
    return re.sub(r'[^\w\s]|_', " ", text).lower().split()

# Method for stopping
def stopping(lst, stopWords):
    for stopWord in stopWords:
            if stopWord in lst:
                lst = list(filter((stopWord).__ne__, lst))
    return lst

# Method for stemming
def stemming(lst):
    return [stem(token) for token in lst]

# Method to calculate precision
def calcPricision(q, docsReturned, relDocs, cutOff):
    count = sum([1 for doc in docsReturned if doc in relDocs[q]])
    return count/cutOff

# Method to calculate recall
def calcRecall(q, docsReturned, relDocs):
    count = sum([1 for doc in docsReturned if doc in relDocs[q]])
    return count/len(relDocs[q])

# Method to calculate DCG@k
def calcDCG_k(docsReturned, gainPerDocReturned):
    dg = 0
    dcg_k = 0
    for i in range(len(docsReturned)):
        doc = docsReturned[i]
        gain = gainPerDocReturned[doc]
        
        if(i == 0 or i == 1):
            dg = gain
        else:
            dg = gain/math.log2(i+1)
        
        dcg_k += dg
        # print('DG: ', dg, ' DCG@k: ', dcg_k)

    return dcg_k

# Method to calculate nDCG
def calcNDCG(docsReturned, relevancePerDocs):
    # print('--------- Inside DCG ---------')
    # print('Docs returned: ', docsReturned)
    # print('Relevance per doc: ', relevancePerDocs)

    gainPerDocReturned = {}
    relevantDocs = {}
    
    # creating a dictionary for relevant docs and their gains
    for (a,b) in relevancePerDocs:
        relevantDocs[a] = b

    for doc in docsReturned:
        if(doc in relevantDocs):
            gainPerDocReturned[doc] = relevantDocs[doc]
        else:
            gainPerDocReturned[doc] = 0

    # print('Table: ', gainPerDocReturned)

    dg = 0
    dcg_k = 0

    # print("DCG --")
    dcg_k = calcDCG_k(docsReturned, gainPerDocReturned)
    # print('DCG@10: ', dcg_k)

    idg = 0
    idcg_k = 0

    # score (or gains) of the relevant docs
    gains = [rel for (doc, rel) in relevancePerDocs]
    gains.sort(reverse = True)

    l = len(docsReturned)
    if(len(gains) < l):
        gains = gains + [0]*(l-len(gains))
    elif(len(gains) > l):
        gains = gains[:l]
    # print('Gains: ', gains)

    for i in range(len(gains)):
        gain = gains[i]
        if(i == 0):
            idg = gain
        else:
            idg = gain/math.log2(i+1)
        idcg_k += idg
        # print('Rank: ', i, ' Gain: ', gain, 'iDG: ', idg, ' iDCG@k: ', idcg_k)

    # print('iDCG@10: ', idcg_k)

    ndcg_k = 0
    if(idcg_k != 0):
        ndcg_k = dcg_k/idcg_k
    # print('nDCG@10: ', ndcg_k)
    return ndcg_k

# Method to calculate the mean of a list
def calcMean(lst):
    return sum(lst)/len(lst)

# Method to calculate means for all the systems
def calcMeans(a, b, c, d, e, f):
    means = []
    low = 0
    high = 10
    for i in range(6):
        lst = [round(calcMean(a[low:high]), 3), round(calcMean(b[low:high]), 3), round(calcMean(c[low:high]), 3), round(calcMean(d[low:high]), 3), round(calcMean(e[low:high]), 3), round(calcMean(f[low:high]), 3)]
        means.append(lst)
        low += 10
        high += 10
    return means

# Method to generate the ir_eval.csv file
def generateIR_EVAL_File(a, b, c, d, e, f):
    print('---- Generating the file -----')
    i = 0
    systems = 6 # number of systems
    queries = 11 # number of queries
    means = calcMeans(a, b, c, d, e, f)
    rows = [['system_number', 'query_number', 'P@10', 'R@50', 'r-precision', 'AP', 'nDCG@10', 'nDCG@20']]
    
    # creating the rows
    for systemNum in range(1, (systems + 1)):
        for queryNum in range(1, (queries + 1)):
            if(queryNum == 11):
                row = [systemNum, 'mean'] + means[systemNum-1]
            else:
                row = [systemNum, queryNum, round(a[i], 3), round(b[i], 3), round(c[i], 3), round(d[i], 3), round(e[i], 3), round(f[i], 3)]
                i += 1
            rows.append(row)

    # open the file in the write mode
    with open('ir_eval.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        # writing to the file
        writer.writerows(rows)

# Method for Task 1 - IR EVALUATION
def EVAL(loc_qrels, loc_system_results):
    print('----- Starting IR Evaluation -----')

    relDocs = {}
    relevancePerQueryPerDoc = {}
    for q in range(1, 11):
        relDocs[q] = []
        relevancePerQueryPerDoc[q] = []


    # reading the qrels.csv file
    with open(loc_qrels, mode='r') as csv_file:
        qrels = csv.DictReader(csv_file)
        for row in qrels:
            q = int(row['query_id'])
            doc = int(row['doc_id'])
            relevance = int(row['relevance'])

            relDocs[q].append(doc)
            relevancePerQueryPerDoc[q].append((doc, relevance))


    resultsPrecision = []
    resultsRecall = []
    resultsRPrecision = []

    precisions = []
    resultsAP = []

    resultsNDCG_10 = []
    resultsNDCG_20 = []

    dgs = []
    resultsDCG = []

    precisionAllowedRanks = [i for i in range(1, 11)]
    recallAllowedRanks = [i for i in range(1, 51)]
    rPrecisionAllowedRanks = []

    dcg10AllowedRanks = [i for i in range(1, 11)]
    # print('DCG 10 allowed ranks: ', dcg10AllowedRanks)
    dcg20AllowedRanks = [i for i in range(1, 21)]

    readDocsCount, relDocsCount = 0, 0

    # reading the systems_results.csv file
    with open(loc_system_results, mode='r') as csv_file:
        systemResults = csv.DictReader(csv_file)

        s = 1
        currLstPrecision = []
        currLstRecall = []
        currLstRPrecision = []
        origQ = 1
        currLstnDCG_10 = []
        currLstnDCG_20 = []

        # Simultaneously calculating all the metrics
        for row in systemResults:

            rank = int(row['rank_of_doc'])
            currQ = int(row['query_number'])
            docReturned = int(row['doc_number'])

            rPrecisionAllowedRanks = [i for i in range(1, len(relDocs[currQ])+1)]
            # print('------------------------------------')
            # print('Curr rank: ', rank)
            # print('Curr Query: ', currQ)

            # print('Len of rel docs: ', len(relDocs[currQ]))
            # print('Doc returned: ', docReturned)

            # PRECISION@10
            if(rank in precisionAllowedRanks):
                currLstPrecision.append(docReturned)
            elif(rank == 11):
                resultsPrecision.append(calcPricision(currQ, currLstPrecision, relDocs, 10))
                currLstPrecision = []

            # RECALL@50
            if(rank in recallAllowedRanks):
                currLstRecall.append(docReturned)
            elif(rank == 51):
                resultsRecall.append(calcRecall(currQ, currLstRecall, relDocs))
                currLstRecall = []

            # R-PRECISION
            if(rank in rPrecisionAllowedRanks):
                # print('Add doc --')
                currLstRPrecision.append(docReturned)
            elif(rank == (rPrecisionAllowedRanks[-1]+1)):
                # print('Clear lst --')
                resultsRPrecision.append(calcPricision(currQ, currLstRPrecision, relDocs, len(rPrecisionAllowedRanks)))
                currLstRPrecision = []
            
            # AP
            readDocsCount += 1
            if(currQ == origQ):
                if(docReturned in relDocs[currQ]):
                    relDocsCount += 1
                    precisions.append(relDocsCount/readDocsCount)
            else:
                # print('Query: ', origQ)
                # print('Precisions: ', precisions)

                if(len(precisions) != 0):
                    resultsAP.append(sum(precisions)/len(relDocs[origQ]))
                    # print('AP: ', sum(precisions)/len(relDocs[origQ]))
                else:
                    resultsAP.append(0.0)
                
                origQ = currQ
                precisions = []
                readDocsCount, relDocsCount = 1, 0
                if(docReturned in relDocs[currQ]):
                    relDocsCount += 1
                    precisions.append(relDocsCount/readDocsCount)

            # nDCG@10
            if(rank in dcg10AllowedRanks):
                currLstnDCG_10.append(docReturned)
            elif(rank == 11):
                resultsNDCG_10.append(calcNDCG(currLstnDCG_10, relevancePerQueryPerDoc[currQ]))
                currLstnDCG_10 = []

            # nDCG@20
            if(rank in dcg20AllowedRanks):
                currLstnDCG_20.append(docReturned)
            elif(rank == 21):
                resultsNDCG_20.append(calcNDCG(currLstnDCG_20, relevancePerQueryPerDoc[currQ]))
                currLstnDCG_20 = []

        # Calculating AP for the last query of the last system
        if(len(precisions) != 0):
            resultsAP.append(sum(precisions)/len(relDocs[origQ]))
        else:
            resultsAP.append(0)

    # print("Results precision: ", resultsPrecision)
    # print("Len: ", len(resultsPrecision))

    # print("Results recall: ", resultsRecall)
    # print("Len: ", len(resultsRecall))

    # print("Results R precision: ", resultsRPrecision)
    # print("Len: ", len(resultsRPrecision))

    # print("Results AP: ", resultsAP)
    # print("Len: ", len(resultsAP))

    # print("Results nDCG@10: ", resultsNDCG_10)
    # print("Len: ", len(resultsNDCG_10))

    # print("Results nDCG@20: ", resultsNDCG_20)
    # print("Len: ", len(resultsNDCG_20))

    generateIR_EVAL_File(resultsPrecision, resultsRecall, resultsRPrecision, resultsAP, resultsNDCG_10, resultsNDCG_20)

# Method to count the frequency of term in verses
def countTerms(verses, term):
    count = 0
    for verse in verses:
        if(term in verse):
            count += 1
    return count

# Method to get the remaining corpuses
def getOther2Corpus(targetCorpus):
    return [c for c in ['ot', 'nt', 'quran'] if c != targetCorpus]
    
# Method read and process the data
def readData(loc_training_corpora, stopWords):
    
    terms = set()
    data = {'ot': [], 'nt': [], 'quran': []}
    
    # Using readlines()
    file1 = open(loc_training_corpora, 'r')
    lines = file1.readlines()
    N = len(lines)

    # Strips the newline character
    for line in lines:
        lst = caseFolding(line) # Case Folding
        lst = stopping(lst, stopWords) # Stopping
        lst = stemming(lst) # Stemming
        if(lst[0] == 'ot'):
            data['ot'].append(lst[1:])
        elif(lst[0] == 'nt'):
            data['nt'].append(lst[1:])
        elif(lst[0] == 'quran'):
            data['quran'].append(lst[1:])
        for term in lst[1:]:
            terms.add(term)
    
    # print('Terms: ', terms)
    # print('Len: ', len(terms))
    return data, terms, N

# Method read and process the data
def readDataTextClassification(loc_training_corpora):
    
    terms = set()
    data = []
    
    # Using readlines()
    file1 = open(loc_training_corpora, 'r')
    lines = file1.readlines()
    N = len(lines)

    # Strips the newline character
    for line in lines:
        lst = caseFolding(line) # Case Folding
        data.append(lst)
        for term in lst[1:]:
            terms.add(term)
    
    # print('Len data: ', len(data))
    # print('Len unique terms: ', len(terms))
    return data, list(terms)

# Method to calculate the Corpus score
def calcScore(probabilities):
    scores = [0]*20
    for lst in probabilities:
        for (x, y) in lst:
            scores[x] += y
    
    for i in range(len(scores)):
        scores[i] = scores[i] / len(probabilities)
    
    return scores

# Method for Task 2 - TEXT ANALYSIS
def textAnalysis(data, terms, N):
    print('----- Starting Text Analysis -----')

    corpuses = ['quran', 'ot', 'nt']
    MIs = [] # final MIs for all the corpuses for all the terms
    CHIs = [] # final CHIs for all the corpuses for all the terms
    
    for corpus in corpuses:
        ec_1 = corpus
        mis = [] # MIs for 1 the corpus
        chis = [] # CHIs for 1 the corpus
        print('Target corpus: ', ec_1)

        for term in terms:

            N11 = countTerms(data[ec_1], term)

            # print('term: ', term)
            # print('Len of target: ', len(data[ec_1]))
            N01 = len(data[ec_1]) - N11

            # calculating N10 and N00

            otherCorpuses = getOther2Corpus(ec_1)
            N10 = 0
            for c in otherCorpuses:
                N10 += countTerms(data[c], term)
            N00 = sum([len(data[c]) for c in otherCorpuses]) - N10

            # print('N : ', N)
            # print('N11: ', N11)
            # print('N10: ', N10)
            # print('N01: ', N01)
            # print('N00: ', N00)
            
            N1D = N11 + N10
            ND1 = N11 + N01
            N0D = N01 + N00
            ND0 = N10 + N00

            p1 = ((N11 / N) * math.log2((N * N11) / (N1D * ND1))) if N*N11 != 0 and N1D*ND1 != 0 else 0
            p2 = ((N01 / N) * math.log2((N * N01) / (N0D * ND1))) if N*N01 != 0 and N0D*ND1 != 0 else 0
            p3 = ((N10 / N) * math.log2((N * N10) / (N1D * ND0))) if N*N10 != 0 and N1D*ND0 != 0 else 0
            p4 = ((N00 / N) * math.log2((N * N00) / (N0D * ND0))) if N*N00 != 0 and N0D*ND0 != 0 else 0
            
            MI = p1 + p2 + p3 + p4
            temp = (N11 * N00) - (N10 * N01)
            numerator = (N11 + N10 + N01 + N00) * np.square(temp)
            denominator = (N11 + N01) * (N11 + N10) * (N10 + N00) * (N01 + N00)
            CHI = numerator / denominator if denominator != 0 else 0

            # print('MI: ', MI)
            # print('CHI: ', CHI)

            mis.append((term, MI))
            chis.append((term, CHI))

        # print('Len mis: ', len(mis))
        # print('Len chis: ', len(chis))

        # sorting in descending order
        mis.sort(key=lambda x: x[1], reverse=True)
        chis.sort(key=lambda x: x[1], reverse=True)
        # print(chis[:10])

        MIs.append(mis)
        CHIs.append(chis)

        # print('Len MIs: ', len(MIs))
        # print('Len CHIs: ', len(CHIs))

        print('----------------')


    print("1st 10 MI values, order - Quran -> OT -> NT")
    [print('Values: ', mi[:10], '\n') for mi in MIs]

    print("1st 10 CHI values, order - Quran -> OT -> NT")
    [print('Values: ', chi[:10], '\n') for chi in CHIs]

# Method for Latent Dirichlet Allocation
def latentDirichletAllocation(data):

    print('----- Starting LDA -----')

    verses = data['ot'] + data['nt'] + data['quran']

    dictionary = Dictionary(verses)

    corpus = [dictionary.doc2bow(verse) for verse in verses]

    lda = LdaModel(corpus, id2word=dictionary, num_topics=20)
    print('LDA: ', lda)

    overallProbabilities = [lda.get_document_topics(corpus[i]) for i in range(len(verses))]
    # print('Len Overall Topic Probabilities: ', len(overallProbabilities))

    corpusLenghts = [0] + [len(data[key]) for key in ['ot', 'nt', 'quran']] # length of corpuses
    # print('Lens: ', corpusLenghts)
    
    start = 0
    end = 0
    for i in range(len(data.keys())):
        print('Corpus: ', list(data.keys())[i])
        start = end
        end = start + corpusLenghts[i+1]
        # print('Start: ', start)
        # print('End: ', end)
        avg = calcScore(overallProbabilities[start:end])
        idx = [i for i in range(0, len(avg))]
        avg, idx = zip(*sorted(zip(avg, idx), reverse=True)) # sorting in descending order, top topic first
        top5Topics = idx[:5] # top 5 topics
        
        topTopic = idx[0]
        topWords = lda.print_topic(topTopic, 10)
        print('Top 10 tokens for top topic: ', topWords)
        print('----')

        # for topTopic in top5Topics:
        #     topWords = lda.print_topic(topTopic, 10)
        #     print('Top 10 tokens: ', topWords)
        #     print('----')
        
        # print('-----------------------------------------')


    # for topic in lda.print_topics(num_topics=20, num_words=10):
    #     print(topic)

# Method to generate the bag of words matrix
def generateSparseMatrix(verses, terms):
    X = dok_matrix((len(verses), len(terms)))
    verses = [verse[1:] for verse in verses]

    for i in range(len(verses)):
        doc = verses[i]
        freq = Counter(doc) # frequency of each word in the document

        for (word, count) in freq.items():
            if(word in terms): # checking if word appears in class terms
                X[i, terms.index(word)] = count

    return X

# Method to calculate the mean
def calcMean(lst):
    return sum(lst)/len(lst)

# Method to calculate the metrics like precision, recall, r-precison, etc.
def calcMetrics(YPred, Ytrue, system, split):
    print('Writing in the file --')

    precisions = []
    recalls = []
    f1_scores = []
    corpuses = ['quran', 'ot', 'nt']

    for corpus in corpuses:

        indx_pred = [i for i in range(len(YPred)) if YPred[i] == corpus]
        indx_true = [i for i in range(len(Ytrue)) if Ytrue[i] == corpus]

        matchCount = sum([1 for idx in indx_true if YPred[idx] == corpus])
        
        # Precison
        precision = matchCount/len(indx_pred)
        precisions.append(round(precision, 3))
        
        # Recall
        recall = matchCount/len(indx_true)
        recalls.append(round(recall, 3))

        # F1-score
        F1_score = 2 * precision * recall / (precision + recall)
        f1_scores.append(round(F1_score, 3))

    macroP = round(calcMean(precisions), 3)
    macroR = round(calcMean(recalls), 3)
    macroF1 = round(calcMean(f1_scores), 3)

    row = [[system, split, precisions[0], recalls[0], f1_scores[0], precisions[1], recalls[1], f1_scores[1], precisions[2], recalls[2], f1_scores[2], macroP, macroR, macroF1]]

    # open the file in the write mode
    with open('classification.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        # writing to the file
        writer.writerows(row)

# Method to find 3 instances of wrong classification in the development dataset
def wrongClassifications(X, Ypred, Y):
    print('Finding wrong classifications in dev dataset --------')
    count = 0
    for i in range(len(Ypred)):
        if(count > 3):
            break
        elif(Ypred[i] != Y[i]): # wrong prediction
            print('Verse: ', X[i])
            print('Prediction: ', Ypred[i], ' Actual: ', Y[i])
            count += 1

# Method to do text classification
def textClassification(locFileTrn, locFileTst):

    print('----- Starting Text Classification -----')
    
    data, terms = readDataTextClassification(locFileTrn)
    random.shuffle(data) # shuffling the data randomly

    # Training dataset
    TrnVerses = data[ : int(0.9*len(data))] #splitting the dataset in the ratio (9:1)
    Ytrn = [verse[0] for verse in TrnVerses]
    
    # Development dataset
    DevVerses = data[int(0.9*len(data)) : ]
    Ydev = [verse[0] for verse in DevVerses]

    # Test dataset
    TstVerses, _ = readDataTextClassification(locFileTst)
    Ytst = [verse[0] for verse in TstVerses]

    Xtrn = generateSparseMatrix(TrnVerses, terms)
    Xdev = generateSparseMatrix(DevVerses, terms)
    Xtst = generateSparseMatrix(TstVerses, terms)

    # deleting the file incase it exists from before
    if os.path.exists('classification.csv'):
        os.remove('classification.csv')

    # generating the file
    header = [['system', 'split', 'p-quran', 'r-quran', 'f-quran', 'p-ot', 'r-ot', 'f-ot', 'p-nt', 'r-nt', 'f-nt', 'p-macro', 'r-macro', 'f-macro']]
    with open('classification.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(header) # writing to the file

    # ---------------- BaseLine ----------------
    svcModel = SVC(C = 1000)
    print('Model train: ', svcModel)
    svcModel.fit(Xtrn, Ytrn) # training the model with training dataset

    # Predictions
    YtrnPred = svcModel.predict(Xtrn)
    YdevPred = svcModel.predict(Xdev)
    YtstPred = svcModel.predict(Xtst)

    # finding the 3 wrong classifications in dev test set
    wrongClassifications(DevVerses, YdevPred, Ydev)

    # calculating the metrics and writing in the file
    calcMetrics(YtrnPred, Ytrn, 'baseline', 'train')
    calcMetrics(YdevPred, Ydev, 'baseline', 'dev')
    calcMetrics(YtstPred, Ytst, 'baseline', 'test')

    # ---------------- Improved ----------------
    svcModel = SVC(C = 10)
    print('Model dev: ', svcModel)
    svcModel.fit(Xtrn, Ytrn) # training the model with training dataset

    # Predictions
    YtrnPred = svcModel.predict(Xtrn)
    YdevPred = svcModel.predict(Xdev)
    YtstPred = svcModel.predict(Xtst)

    # calculating the metrics and writing in the file
    calcMetrics(YtrnPred, Ytrn, 'improved', 'train')
    calcMetrics(YdevPred, Ydev, 'improved', 'dev')
    calcMetrics(YtstPred, Ytst, 'improved', 'test')

    

# --------------------------- Method Calling (for testing) ---------------------------

# EVAL('qrels.csv', 'system_results.csv')

# reading the stop words
# f = open('stop_words.txt', "r")
# stopWords = f.read().lower().split()
# f.close()

# data, terms, N = readData('train_and_dev.txt', stopWords)
# textAnalysis(data, terms, N)
# latentDirichletAllocation(data)

# textClassification('train_and_dev.txt', 'test.txt')

# ------- Test the DCG method with values from the lecture --------

# docsReturned = [11, 22, 33, 44, 55, 66, 77, 88, 99, 1010]
# relevancePerDocs = [(11, 3), (22, 2), (33, 3), (44, 0), (55, 0), (66, 1), (77, 2), (88, 2), (99, 3), (1010, 0)]

# calcNDCG(docsReturned, relevancePerDocs)


# ------------------------------------------ Menu ------------------------------------------

flag = True
while(flag):

    choice = input('Enter 1: IR Evaluation, 2: Text Analysis, 3: Text Classification => ')
    if(choice == '1'):
        qrelsLoc = input('Provide location of qrels.csv file => ')
        system_resultsLoc = input('Provide location of system_results.csv file => ')
        EVAL(qrelsLoc, system_resultsLoc)
    elif(choice == '2'):
        dataLoc = input('Provide location of train_and_dev.txt file => ')
        stopWordsLoc = input('Provide location of stopwords.txt file => ')
        
        # reading the stop words
        f = open(stopWordsLoc, "r")
        stopWords = f.read().lower().split()
        f.close()

        data, terms, N = readData(dataLoc, stopWords)
        textAnalysis(data, terms, N)
        latentDirichletAllocation(data)
    elif(choice == '3'):
        dataLoc = input('Provide location of train_and_dev.txt file => ')
        testLoc = input('Provide location of test.txt file => ')
        textClassification(dataLoc, testLoc)

    runAgain = input('Enter 1: To run again, 2: Exit => ')
    if(runAgain != '1'):
        flag = False