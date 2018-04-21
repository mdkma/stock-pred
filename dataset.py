import os, time
import numpy as np
import pandas
import re
import string
import random
import math

class Dataset(object):
    def __init__(self, data_path, voc, batch_size, word_emb_dim):
        self.voc = voc

        training_set = pandas.read_csv(data_path)
        # training_set = data[data['Date'] <= '2014-12-31']
        # testing_set = data[data['Date'] >= '2015-01-02']

        training_date = training_set['Date']
        training_label = training_set['Label']
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
            docs.append(thisDoc)
        # print(docs[0][0])
        print("-> wordinsent_cnt_train: ", wordinsent_cnt)
        self.wordinsent_cnt = wordinsent_cnt

        # WORDS TO NUMBERS
        non_word_emb = np.zeros(word_emb_dim)
        self.docs = list(map(lambda doc: list(map(lambda sentence: list(filter(lambda wordid: wordid !=-1,list(map(lambda word: self.get_word_emb(word),sentence)))),doc)),docs))
        self.epoch = math.ceil(len(self.docs) / batch_size)

    def get_word_emb(self, word):
        try:
            return self.voc[word]
        except:
            return -1