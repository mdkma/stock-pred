import os, time
import numpy as np
import pandas
import re
import string
import random
import math
from functools import reduce

class Dataset(object):
    def __init__(self, data_path, voc, batch_size, word_emb_dim, prev_wordinsent_cnt = 0):
        self.voc = voc

        training_set = pandas.read_csv(data_path)
        # training_set = data[data['Date'] <= '2014-12-31']
        # testing_set = data[data['Date'] >= '2015-01-02']

        training_date = training_set['Date']
        self.temp_label = training_set['Label']
        training_news_group = training_set.iloc[:, 2:27]
        # testing_date = testing_set['Date']
        # testing_label = testing_set['Label']
        # testing_news_group = testing_set.iloc[:, 2:27]

        remove = string.punctuation
        remove = remove.replace("-", "") # don't remove hyphens
        pattern = r"[{}]".format(remove) # create the pattern

        # Preprocess the news by using regex
        for title in training_news_group:
            for index in range(len(training_news_group[title])):
                news = re.sub(r"(^b)([\"\'])", "", str(training_news_group[title][index]))
                news = re.sub(r"([\"\']$)", "", str(news))
                news = re.sub(pattern, "", str(news)) 
                training_news_group[title][index] = news

        # print(training_news_group.head(5))
        self.num_docs = len(training_news_group['Top1'])
        docs = []
        wordinsent_cnt = 0
        for index in range(len(training_news_group['Top1'])):
            thisDoc = []
            for title in training_news_group:
                sen = training_news_group[title][index].split(' ')
                if (len(sen) > wordinsent_cnt):
                    wordinsent_cnt = len(sen)
                sen_lower = [x.lower() for x in sen] # lower case for all words
                thisDoc.append(sen_lower)
            #print(len(thisDoc)) #25
            docs.append(thisDoc)
        # print(docs[0][0])
        print("-> wordinsent_cnt_train: ", wordinsent_cnt)
        if prev_wordinsent_cnt > wordinsent_cnt:
            self.wordinsent_cnt = prev_wordinsent_cnt
        else:
            self.wordinsent_cnt = wordinsent_cnt

        # WORDS TO NUMBERS
        # non_word_emb = np.zeros(word_emb_dim)
        self.temp_docs = list(map(lambda doc: list(map(lambda sentence: list(filter(lambda wordid: wordid !=-1,list(map(lambda word: self.get_word_emb(word),sentence)))),doc)),docs))
        self.epoch = math.ceil(len(self.temp_docs) / batch_size)

        self.docs = []
        self.label = []
        # self.wordmask = []
        # self.sentencemask = []
        # self.maxsentencenum = []
        for i in range(self.epoch):
            docsbatch = self.genBatch(self.temp_docs[i * batch_size:(i + 1) * batch_size])
            # docsbatch = self.temp_docs[i * batch_size:(i + 1) * batch_size]
            self.docs.append(docsbatch)
            self.label.append(np.asarray(self.temp_label[i * batch_size:(i + 1) * batch_size], dtype=np.int32))


    def get_word_emb(self, word):
        try:
            return self.voc[word]
        except:
            return -1

    def genBatch(self, data):
        #print(len(data)) #32
        # m = 0
        # # maxsentencenum = len(data[0])
        # maxsentencenum = self.wordinsent_cnt
        # # maxsentencenum = max(data, key=len)
        # # print(maxsentencenum)
        # for doc in data:
        #     for sentence in doc:
        #         if len(sentence)>m:
        #             m = len(sentence)
        #     for i in range(maxsentencenum - len(doc)):
        #         doc.append([0])
        # print (m)
        m = self.wordinsent_cnt
        tmp = list(map(lambda doc: np.asarray(list(map(lambda sentence : sentence + [0]*(m - len(sentence)), doc)), dtype = np.int32).T, data))
        # print (type(tmp))
        # print (len(tmp))
        # print (tmp[0])
        tmp = reduce(lambda doc,docs : np.concatenate((doc,docs),axis = 1),tmp)
        # print(np.shape(tmp))
        return tmp