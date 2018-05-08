#coding=utf-8
import codecs
import pickle
from collections import defaultdict

import numpy as np
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression

from library.lib import SENTENCE_DIVIDERS
from models.relation import Features

__author__ = 'Timur Mardanov'


class PipeLine(object):
    leftVec = None
    rightVec = None
    classifier = None
    classifier_n_jobs=4
    classifier_solver='lbfgs'


    def __init__(self, app, test_counts = None):
        '''
            Init pipeline
        '''
        self.app = app
        self.leftVec = self.get_vectorizer(app)
        self.rightVec = self.get_vectorizer(app)
        self.classifier = LogisticRegression(n_jobs=self.classifier_n_jobs,
                                             solver=self.classifier_solver)

        self.pos_ids = defaultdict(lambda: len(self.pos_ids.items()))

    def save(self, path):
        pickle.dump(self, open(path, 'wb'))

    @classmethod
    def load(cls, path):
        return pickle.load(open(path, 'rb'))

    def get_vectorizer(self, app):
        '''
            Get TfidfVectorizer instance
        '''
        return TfidfVectorizer(
            encoding=app.text_encoding,
            ngram_range=(app.n, ) * 2,
            max_df=app.unknown_word_freq or 1.0,
            max_features=2 * len(app.relations),
            smooth_idf=app.laplace,
        )

    def fit(self, left_words, right_words, types):
        '''
            Fit model
        '''
        self.target = types
        lData = self.leftVec.fit_transform(left_words)
        rData = self.rightVec.fit_transform(right_words)
        self.matrix = hstack((lData, rData))

        # Convert to usable format after construction
        self.matrix = self.matrix.tocsr()
        self.classifier.fit(self.matrix, self.target)

    def refit(self, ):
        '''
            Refit model (with new features)
        '''
        self.classifier = LogisticRegression(n_jobs=self.classifier_n_jobs,
                                             solver=self.classifier_solver)
        self.classifier.fit(self.matrix, self.target)

    def transform(self, left_words, right_words):
        '''
            Get transformed left and right words from TfIdfVectorizer
        '''
        lData = self.leftVec.transform(left_words).toarray()
        rData = self.rightVec.transform(right_words).toarray()
        self.test_data = np.append(lData, rData, axis=1)

    def test(self,):
        return self.classifier.predict(self.test_data)

    #------------------------------------------------------------------------
    #                      3task (A part)
    #------------------------------------------------------------------------

    #A1
    def ref_in_one_cpos(self):
        #CPOS
        print 'CPOS calculation'
        cpos = np.array([self.pos_ids[pos] for pos in calculate_cpos_in_one_sentence(self.app)])
        self.matrix = hstack((self.matrix, cpos.T))

    #A2
    def ref_in_one_wvnull(self):
        #WVNULL
        print 'WVNULL calculation'
        self.matrix = hstack((self.matrix, np.array([calculate_wvnull_in_one_sentence(self.app)]).T))

    #A3
    def ref_in_one_wvfl(self):
        #WVFL
        print 'WVFL calculation'
        self.matrix = hstack([self.matrix, np.array([calculate_wvfl_in_one_sentence(self.app)]).T])

    #A4
    def ref_in_one_wbnull(self):
        #WBNULL
        print 'WBNULL calculation'
        self.matrix = hstack([self.matrix, np.array([calculate_wbnull_in_one_sentence(self.app)]).T])

    #A5
    def ref_in_one_wbfl(self):
        #WBFL
        print 'WBFL calculation'
        self.matrix = hstack([self.matrix, np.array([calculate_wbfl_in_one_sentence(self.app)]).T])


    #------------------------------------------------------------------------
    #                      3task ты что дурак?
    #------------------------------------------------------------------------
    #C1
    def ref_in_diff_sdist(self):
        #SDIST
        print 'SDIST calculation'
        self.matrixA = np.hstack([self.matrix,np.array([calculate_sdist_in_diff_sentence(self.app)]).T])
    #C2
    def left_entity_freq_in_doc(self):
        #CRFQ
        print 'CRFQ calculation'
        self.matrixA = np.hstack([self.matrix, np.array([calculate_entity_freq_in_doc_in_diff_sentence(self.app,left=True)]).T])

    #C3
    def right_entity_freq_in_doc(self):
        # DRFQ
        print 'CRFQ calculation'
        self.matrixA = np.hstack([self.matrix, np.array([calculate_entity_freq_in_doc_in_diff_sentence(self.app, left=False)]).T])

#------------------------------------------------------------------------
#                      3task (C part)
#------------------------------------------------------------------------



#=============================================================================
#                       3 task A realisation
#=============================================================================

def verb_in_sentence(tagged):
    if len(tagged) == 0:
        return 0
    else:
        return int(u'VERB' in tagged[:,1])

def only_one_verb_in_sentence(tagged):
    if len(tagged) == 0:
        return 0
    else:
        return int(sum(map(lambda tag: u'VERB' == tag, tagged[:,1]))==1)

# CPOS (pos_tag of first entity)
def calculate_cpos_in_one_sentence(app):
    rels = app.relations
    return map(
        lambda rel: pos_tag(word_tokenize(rel.refAobj.value), tagset='universal')[0][1]
        if rel.feature_type == Features.InOneSentence else 'NO',
        rels)

# no verb between
def calculate_wvnull_in_one_sentence(app):
    references = app.relations
    return map(lambda reference: \
               verb_in_sentence(np.array(pos_tag(reference.tokenized_text_between, tagset='universal')))\
                    if reference.feature_type == Features.InOneSentence else -1, references)

# only one verb
def calculate_wvfl_in_one_sentence(app):
    references = app.relations
    return map(lambda reference: verb_in_sentence(np.array(pos_tag(reference.tokenized_text_between, tagset='universal')))\
               if reference.feature_type == Features.InOneSentence else -1,references)

def calculate_wbnull_in_one_sentence(app):
    references = app.relations
    #INTBOOLLEN я вызываю тебя...........
    return map(lambda reference: int(bool(len(reference.text_between))) if reference.feature_type == Features.InOneSentence else -1,references)

def calculate_wbfl_in_one_sentence(app):
    references = app.relations
    return map(lambda reference: int(len(reference.tokenized_text_between) == 1) if reference.feature_type == Features.InOneSentence else -1,references)

#=============================================================================
#                       3 task B realisation( осторожно. снизу кривой код)
#=============================================================================

def calculate_sdist_in_diff_sentence(app):
    #column of matrix with SDIST feature
    sdist_feature = []
    references = app.relations
    for rel in references:
        if rel.feature_type == Features.InDifferentSentence:
            count_sentence_dividers = 0
            for divider in SENTENCE_DIVIDERS:
                count_sentence_dividers += rel.text_between.count(divider)-1
            sdist_feature.append(count_sentence_dividers)
        else:
            sdist_feature.append(-1)
    return sdist_feature

def calculate_entity_freq_in_doc_in_diff_sentence(app, left=True):
    #column of matrix with CRFQ feature
    cfrq_feature = []
    for doc in app.documents:
        doc_file = codecs.open(doc.text_path, 'r', encoding='utf-8')
        text_of_doc = doc_file.read()
        for rel in doc.references:
            if rel.feature_type == Features.InDifferentSentence:
                entity_text = rel.refAobj.value if left else rel.refBobj.value
                cfrq_feature.append(text_of_doc.count(entity_text))
            else:
                cfrq_feature.append(-1)
        doc_file.close()
    return cfrq_feature



#------------------------------------------------------------------------
#                      End
#------------------------------------------------------------------------

