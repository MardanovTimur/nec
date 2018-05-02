#coding=utf-8
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from models.reference import Features

__author__ = 'Timur Mardanov'

class PipeLine(object):
    leftVec = None
    rightVec = None
    classifyer = None


    '''
        test_counts - barier for classification data.
        Example : fit data, which count less than test_counts = 200
        if None then classificator hasn't barier
    '''
    def __init__(self, app, test_counts = None):
        self.app = app
        self.test_counts = test_counts
        self.leftVec = self.get_vectorizer(app)
        self.rightVec = self.get_vectorizer(app)
        self.classifyer = LogisticRegression(n_jobs=4, solver='lbfgs')

    def get_vectorizer(self, app):
        return TfidfVectorizer(
            encoding=app.text_encoding,
            ngram_range=(app.n, ) * 2,
            max_df=app.unknown_word_freq or 1.0,
            max_features = 2 * app.references_count,
        )

    def fit(self, left_words, right_words, types):
        lData = self.leftVec.fit_transform(left_words).toarray()
        rData = self.rightVec.fit_transform(right_words).toarray()
        self.matrix = np.append(lData, rData, axis=1)
        if self.test_counts is not None:
            self.classifyer.fit(self.matrix[:self.test_counts], types[:self.test_counts])
        else:
            self.classifyer.fit(self.matrix, types)

    def transform(self, left_words, right_words):
        lData = self.leftVec.transform(left_words).toarray()
        rData = self.rightVec.transform(right_words).toarray()
        self.test_data = np.append(lData, rData, axis=1)

    def test(self,):
        return self.classifyer.predict(self.test_data)

    #A1
    def ref_in_one_cpos(self):
        #CPOS
        print 'CPOS calculation'
        new_matrix = np.array([calculate_cpos_in_one_sentence(self.app)]).T
        self.matrixA = np.hstack([self.matrix, new_matrix])

    #A2
    def ref_in_one_wvnull(self):
        #WVNULL
        print 'WVNULL calculation'
        self.matrixA = np.hstack([self.matrix, np.array(calculate_wvnull_in_one_sentence(self.app)).T])

    #A3
    def ref_in_one_wvfl(self):
        #WVFL
        print 'WVFL calculation'
        self.matrixA = np.hstack([self.matrix, np.array(calculate_wvfl_in_one_sentence(self.app)).T])

    #A4
    def ref_in_one_wbnull(self):
        #WBNULL
        print 'WBNULL calculation'
        self.matrixA = np.hstack([self.matrix, np.array(calculate_wbnull_in_one_sentence(self.app)).T])

    #A5
    def ref_in_one_wbfl(self):
        #WBFL
        print 'WBFL calculation'
        self.matrixA = np.hstack([self.matrix, np.array(calculate_wbfl_in_one_sentence(self.app)).T])


#------------------------------------------------------------------------
#                      3task (A part)
#------------------------------------------------------------------------

def verb_in_sentence(tagged):
    __doc__ = 'LIB'
    if len(tagged) == 0:
        return 0
    else:
        return int(u'VERB' in tagged[:,1])

def only_one_verb_in_sentence(tagged):
    __doc__ = 'LIB'
    if len(tagged) == 0:
        return 0
    else:
        return int(sum(map(lambda tag: u'VERB' == tag, tagged[:,1]))==1)

# CPOS (pos_tag of first entity)
def calculate_cpos_in_one_sentence(app):
    references = app.all_references
    return map(lambda reference: pos_tag(word_tokenize(reference.refAobj.value),
                         tagset='universal')[0][1] if reference.feature_type == Features.InOneSentence else -1,references)

# no verb between
def calculate_wvnull_in_one_sentence(app):
    references = app.all_references
    return map(lambda reference: \
               verb_in_sentence(np.array(pos_tag(reference.tokenized_text_between, tagset='universal')))\
                    if reference.feature_type == Features.InOneSentence else -1, references)

# only one verb
def calculate_wvfl_in_one_sentence(app):
    references = app.all_references
    return map(lambda reference: verb_in_sentence(np.array(pos_tag(reference.tokenized_text_between, tagset='universal')))\
               if reference.feature_type == Features.InOneSentence else -1,references)

def calculate_wbnull_in_one_sentence(app):
    references = app.all_references
    #INTBOOLLEN я вызываю тебя...........
    return map(lambda reference: int(bool(len(reference.text_between))) if reference.feature_type == Features.InOneSentence else -1,references)

def calculate_wbfl_in_one_sentence(app):
    references = app.all_references
    return map(lambda reference: int(len(reference.tokenized_text_between) == 1) if reference.feature_type == Features.InOneSentence else -1,references)

#------------------------------------------------------------------------
#                      End
#------------------------------------------------------------------------

