import math
import re
# from nltk.stem.snowball import SnowballStemmer
from re import Pattern
import re

from nltk.stem import PorterStemmer
import xml.etree.ElementTree as ET

import numpy as np

table = []
collection_matrix = []
counter = []
ps = PorterStemmer()
doc_list2 = []
no_sw = []
document_ids = []
w_stop = []
boolean_flag = 0
unique_words =[]


# reading the stop words from the file
def stop_words():

    with open("stop_words.txt") as file_in:
        stop_words = []
        for line in file_in:
            stop_words.append(line.replace("\n", ""))

    return stop_words

# removing the stop words from the documents
def stopping(lst, w_stop):

    for stop_word in w_stop:
        if stop_word in lst:
            lst = list(filter((stop_word).__ne__, lst))
    return lst

# applying stemming on the words
def stemm(lst):
    return [ps.stem(demo) for demo in lst]


# method to read the docs from the xml file
def read_docs(string):

    # global doc_list2
    # global document_ids
    global unique_words

    documents = ET.parse(string).getroot()
    doc_list = []
    w_stop = stop_words()
    document_ids = []

    # loop to iterate over the documents
    for document in documents:
        temp = ""
        document_ids.append(document[0].text)
        for i in range(1, len(document)):
            if document[i].tag == "TEXT" or document[i].tag == "HEADLINE": # reading only the headlines and the text tags
                temp += " " + document[i].text

        words = re.sub(r'[^\w\s]|_', " ", temp).lower().split()
        words = stopping(words, w_stop)
        words = stemm(words)
        doc_list.append(words)

    x = ""

    number_rows = len(doc_list)  # 3 rows in your example

    doc_list2 = []
    all_words = ""

    for i in range(number_rows):
        for j in range(0, len(doc_list[i])):
            x += doc_list[i][j] + " "
            all_words += doc_list[i][j] + " "

        doc_list2.append(x)
        x = ""

    unique_words = set(all_words.split(' '))
    unique_words.remove("")

    return doc_list2, unique_words, document_ids

# method to find the inverted index of the docs
def inverted_index():

    for i in range(0, len(doc_list2)):
        doc_list2[i] = doc_list2[i].lower()

    for i in range(0, len(doc_list2)):
        for j in w_stop:
            if j in doc_list2[i]:
                doc_list2[i] = doc_list2[i].replace(j, "")

    result = {}

    for doc_id, doc in enumerate(doc_list2):

        for word_pos, word in enumerate(doc.split()):
            result.setdefault(word, []).append((doc_id, word_pos))

    # FINDING THE VECTOR NOW

    counter = []

    for i in range(0, len(doc_list2)):
        for j in no_sw:
            counter.append(doc_list2[i].split().count(j))

    for i in range(1, len(counter)+1):
        print(counter[i-1], "  ", end="")
        if i % len(no_sw) == 0:
            print("\n")

# method to create bi-grams
def bi_gram(query):

    idxs = []

    for i in range(len(query)):
        if query[i] == '"':
            idxs.append(i)

    new_query = ""

    track = 0
    for i in idxs[::2]:

        if track == 0:
            idx1 = i+1 # start
            idx2 = idxs[track+1] # stop
            temp = query[idx1:idx2].replace(" ", "_")

            query = query.replace(query[i : idxs[track+1]+1], temp)
            track = 2
        else:
            idx1 = i+1-2 # start
            idx2 = idxs[track+1] - 2 # stop
            temp = query[idx1:idx2].replace(" ", "_")
            # print("temp: ", temp)
            query = query.replace(query[i-2 : idxs[track+1]+1-2], temp)
            track = 2

    return query


def query_processing(query):

    query = bi_gram(query)
    query = query.split()

    connecting_words = []
    cnt = 1
    different_words = []

    f = 0
    for word in query: # iterating over the processed query
        if f == 1:
            f = 0
            continue

        # dealing with the NOT operations
        if word.lower() == "not":
            different_words.append("@" + ps.stem(query[query.index(word) + 1].lower()))  # to differentiate NOT words
            query[query.index(word)] = "Gaand marao"
            f = 1

        # dealing with the AND and OR operations
        elif word.lower() != "and" and word.lower() != "or" and word.lower() != "not":
            if "_" in word.lower():
                splitted = word.lower().split("_")
                different_words.append(ps.stem(splitted[0]) + "_" + ps.stem(splitted[1]))

            else:
                different_words.append(ps.stem(word.lower()))

        else:
            connecting_words.append(word.lower())

    return connecting_words, different_words

# method for boolean Search
def boolean_search(query):

    global boolean_flag
    ps = PorterStemmer()
    no_sw2 = []
    counter = []

    connecting_words, different_words = query_processing(query)

    for i in different_words:
        if "@" in i:
            no_sw2.append(i.replace("@", ""))
            continue
        no_sw2.append(i)

    # global table
    # global collection_matrix
    # global counter


    for i in range(0, len(doc_list2)):
        for j in no_sw2:
            if "_" in j:  # for bigrams

                if doc_list2[i].count(j.replace("_", " ")+" ") >= 1:
                    counter.append(1)

                else:
                    counter.append(0)

            else:

                if doc_list2[i].split().count(j) >= 1:
                    counter.append(1)
                else:
                    counter.append(0)

    collection_matrix = []
    row_cm = []
    collection_matrix = np.array(counter).reshape(len(doc_list2), len(no_sw2))
    collection_matrix = np.transpose(collection_matrix)

    table = collection_matrix.copy()
    table = np.array(collection_matrix).tolist()
    boolean_flag = 1

    zeroes_and_ones_of_all_words = []

    for word in different_words:

        if word[0] == "@":
            bitwise_op = [not w1 for w1 in table[no_sw2.index(word[1:])]]
            bitwise_op = [int(b == True) for b in bitwise_op]
            zeroes_and_ones_of_all_words.append(bitwise_op)
            continue

        zeroes_and_ones_of_all_words.append(table[no_sw2.index(word)])

    if connecting_words != []:

        for word in connecting_words:

            word_list1 = zeroes_and_ones_of_all_words[0]
            word_list2 = zeroes_and_ones_of_all_words[1]

            if word == "and":
                bitwise_op = [w1 & w2 for (w1, w2) in zip(word_list1, word_list2)]
                zeroes_and_ones_of_all_words.remove(word_list1)
                zeroes_and_ones_of_all_words.remove(word_list2)
                zeroes_and_ones_of_all_words.insert(0, bitwise_op)

            elif word == "or":
                bitwise_op = [w1 | w2 for (w1, w2) in zip(word_list1, word_list2)]
                zeroes_and_ones_of_all_words.remove(word_list1)
                zeroes_and_ones_of_all_words.remove(word_list2)
                zeroes_and_ones_of_all_words.insert(0, bitwise_op)

            elif word == "not":
                bitwise_op = [not w1 for w1 in word_list1]
                bitwise_op = [int(b == True) for b in bitwise_op]
                zeroes_and_ones_of_all_words.remove(word_list1)
                zeroes_and_ones_of_all_words.insert(0, bitwise_op)

        zeroes_and_ones_of_all_words.insert(0, bitwise_op)
        lis = zeroes_and_ones_of_all_words[0]

    else:
        lis = zeroes_and_ones_of_all_words[0]

    found_docs = []

    for i in range(len(lis)):
        if lis[i] == 1:
            found_docs.append(int (document_ids[i]))

    unique, counts = np.unique(lis, return_counts=True)

    print(dict(zip(unique, counts)))
    # print(sorted(found_docs))
    # found_docs.sort(key=lambda ele: ele[1])

    found_docs = sorted(found_docs)

    return found_docs

# method to preprocess the query
def query_processing2(flag, query):

    idx1 = query.index('(')
    idx2 = query.index(')')

    list_query_words = [ps.stem(item).replace(" ","") for item in query[idx1+1:idx2].split(",")]
    difference_num = int ((query.replace(query[idx1:], "")).replace("#",""))

    return list_query_words, difference_num

# method to run the proximity Search
def proximity_search(query):

    list_query_words, difference_num = query_processing2(1, query)

    print(list_query_words)

    counter = []
    ps = PorterStemmer()

    for i in range(0, len(doc_list2)):
        for j in list_query_words:

            if "_" in j:  # for bigrams

                if doc_list2[i].count(j.replace("_"," ")) >= 1:
                    counter.append(1)

                else:
                    counter.append(0)

            else:

                if doc_list2[i].split().count(j) >= 1:
                    counter.append(1)
                else:
                    counter.append(0)

    collection_matrix = []
    row_cm = []
    collection_matrix = np.array(counter).reshape(len(doc_list2), len(list_query_words))
    collection_matrix = np.transpose(collection_matrix)

    table = collection_matrix.copy()
    table = np.array(collection_matrix).tolist()

    # - - - - - - - - Check only for documents where both the words exist

    documents_to_check = []

    for i in range(len(table)-1):
        for j in range(len(table[i])):
            if (table[i][j] == 1 and table[i+1][j] == 1):
                documents_to_check.append(j)

    # - - - - - - - - Checked - - - - - - - - - -

    big_list = []

    for i in list_query_words:
        medium_list = []
        for j in documents_to_check:
            idx = 0
            for k in doc_list2[j].split():
                if i == k:
                    medium_list.append(tuple((int(document_ids[j]), idx)))

                idx += 1

        big_list.append(medium_list)

    diff_final = []

    for i in range(len(big_list[0])):
        for j in range(len(big_list[1])):
            if (big_list[0][i][0] == big_list[1][j][0]) and abs((big_list[0][i][1] - big_list[1][j][1])) <= difference_num:
                diff_final.append(big_list[0][i][0])

    diff_final = sorted(set(diff_final))

    # print("Docs found: ", diff_final)

    return diff_final


def query_proccessing3(query):

    query = stemm(stopping(query.split(" "), stop_words()))
    query.sort()

    return query

# method to compute the TDIFs
def tdif(query):

    query = query_proccessing3(query)
    # print(doc_list2)
    counter = []

    # counting the number of occurance of words in query in the documents
    for i in range(0, len(doc_list2)):
        for j in query:
            counter.append(doc_list2[i].count(j))


    collection_matrix = []
    row_cm = []
    collection_matrix = np.array(counter).reshape(len(doc_list2), len(query))
    collection_matrix = np.transpose(collection_matrix)

    table = collection_matrix.copy()
    table = np.array(collection_matrix).tolist()

    sum_rows = []
    formula_applied = []
    new_table = np.array(table)
    new_table = new_table.astype(float)

    for i in range(len(table)):
        ct = 0
        for j in range(len(table[i])):
            if table[i][j] != 0:
                ct = ct + 1

        if ct != 0:
           sum_rows.append(len(table[i])/ct)
        else:
            sum_rows.append(0)

    # print(sum_rows)

    # computing the final scores
    for i in range(len(new_table)):
        for j in range(len(new_table[i])):
            if new_table[i][j] > 0:
                new_table[i][j] = (1 + math.log10(new_table[i][j]))*math.log10(sum_rows[i])
            # else:
            #     new_table[i][j] = 0

    print(new_table)
    score_column_wise = new_table.sum(axis=0)

    tuple_list = ()

    for i in range(len(score_column_wise)):
        tuple_list = tuple_list + ((document_ids[i], score_column_wise[i]),)

    # sorting the tuple list in descending order
    tuple_list = sorted(tuple_list, key=lambda tup: tup[1])
    tuple_list = (tuple_list[::-1])[:150] # top 150 relevent documents
    # print((tuple_list)[:10])

    return tuple_list


# method to generate the ranked file
def generate_results_rankedFile():

    f = open("results.ranked.txt", "a")

    queries = ['income tax reduction', 'stock market in Japan', 'health industry', 'the Robotics industries',
               'the peace process in the middle east', 'information retrieval', 'Dow Jones industrial average stocks',
               'will be there a reduction in the income taxes?', 'the gold prices versus the dollar price',
               'FT article on the BBC and BSkyB deal']

    for i in range(len(queries)):

        result = tdif(queries[i])
        for res in result:
            f.write(str(i + 1) + ',' + str(res[0]) + ',' + str(round(res[1], 4)))
            f.write('\n')
    f.close()


# Method to generate the boolean file
def generate_results_booleanFile():
    f = open("results.boolean.txt", "a")

    queries = ['Happiness', 'Edinburgh AND SCOTLAND', 'income AND taxes', '"income taxes"', '#20(income, taxes)',
               '"middle east" AND peace', '"islam religion"', '"Financial times" AND NOT BBC',
               '"wall street" AND "dow jones"', '#15(dow,stocks)']

    for i in range(len(queries)):

        result = []

        if (queries[i][0] == '#'):  # Proximity Search
            result = proximity_search(queries[i])
        else:  # Boolean Search
            result = boolean_search(queries[i])

        for res in result:
            f.write(str(i + 1) + ',' + str(res))
            f.write('\n')
    f.close()


def generate_index_file():

    f = open("index.txt", "w")

    big_list = []
    d = {}
    for i in unique_words:
        medium_list = []
        for j in doc_list2:
            idx = 0
            for k in j.split():
                if i == k:
                    medium_list.append(tuple((int(document_ids[doc_list2.index(j)]), idx)))
                idx += 1

        big_list.append(medium_list)
        d[i] = medium_list


    for token in d:
        docs = []
        for (docid, index) in d[token]:
            docs.append(docid)
        docs = set(docs)
        length = len(docs)
        f.write(token + ':' + str(length) + '\n')
        indexesAt = d[token]

        for doc in docs:
            idxs = []
            for (docid, idx) in indexesAt:
                if (docid == doc):
                    idxs.append(idx)
            f.write(('\t' + str(doc) + ': ').rstrip('\n'))
            for i in range(len(idxs)):
                if i < len(idxs) - 1:
                    f.write((str(idxs[i]) + ',').rstrip('\n'))
                else:
                    f.write(str(idxs[i]).rstrip('\n'))
            f.write('\n')
    f.close()

    # print(d)
    # print(big_list)

    f.close()


if __name__ == '__main__':

    doc_list2, no_sw, document_ids = read_docs("CW1collection/trec.5000.xml")
    # doc_list2, no_sw, document_ids = read_docs("collections/sample.xml")
    no_sw = sorted(no_sw)
    w_stop = stop_words()

    # generate_results_booleanFile()
    # generate_results_rankedFile()
    generate_index_file()
    # print(doc_list2)
    # print(unique_words)
    # print("Document 8374: ", doc_list2[document_ids.index("8374")])
    # tdif("industry in scotland")

    # boolean_search("Window")
    # boolean_search("Scotland")