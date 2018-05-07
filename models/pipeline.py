# coding=utf-8
import codecs

from sklearn.linear_model.logistic import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize

from library.lib import SENTENCE_DIVIDERS
from models.reference import Features

__author__ = 'Timur Mardanov'


class PipeLine(object):
    leftVec = None
    rightVec = None
    classifyer = None
    classifyer_n_jobs=4
    classifyer_solver='lbfgs'


    def __init__(self, app, ):
        '''
            Init pipeline
        '''

        self.app = app
        self.leftVec = self.get_vectorizer(app)
        self.rightVec = self.get_vectorizer(app)
        self.classifyer = LogisticRegression(n_jobs=self.classifyer_n_jobs,
                                             solver=self.classifyer_solver)

    def get_vectorizer(self, app):
        '''
            Get TfidfVectorizer instance
        '''
        return TfidfVectorizer(
            encoding=app.text_encoding,
            ngram_range=(app.n,) * 2,
            max_df=app.unknown_word_freq or 1.0,
            max_features=2 * app.references_count,
            smooth_idf=app.laplace,
        )

    def fit(self, left_words, right_words, types):
        '''
            Fit model
        '''
        self.target = types
        lData = self.leftVec.fit_transform(left_words).toarray()
        rData = self.rightVec.fit_transform(right_words).toarray()
        self.matrix = np.append(lData, rData, axis=1)
        self.classifyer.fit(self.matrix, self.target)

    def refit(self, ):
        '''
            Refit model (with new features)
        '''
        self.classifyer = LogisticRegression(n_jobs=self.classifyer_n_jobs,
                                             solver=self.classifyer_solver)
        self.classifyer.fit(self.matrix, self.target)

    def transform(self, left_words, right_words):
        '''
            Get transformed left and right words from TfIdfVectorizer
        '''
        lData = self.leftVec.transform(left_words).toarray()
        rData = self.rightVec.transform(right_words).toarray()
        self.test_data = np.append(lData, rData, axis=1)

    def test(self, ):
        return self.classifyer.predict(self.test_data)

    #------------------------------------------------------------------------
    #                      3task (A part)
    #------------------------------------------------------------------------

    # A1
    def ref_in_one_cpos(self):
        # CPOS
        print 'CPOS calculation'
        cpos = np.array([calculate_cpos_in_one_sentence(self.app)])
        self.matrix = np.append(self.matrix, cpos.T, axis=1)

    # A2
    def ref_in_one_wvnull(self):
        # WVNULL
        print 'WVNULL calculation'
        self.matrix = np.append(self.matrix, np.array([calculate_wvnull_in_one_sentence(self.app)]).T, axis=1)

    # A3
    def ref_in_one_wvfl(self):
        # WVFL
        print 'WVFL calculation'
        self.matrix = np.hstack([self.matrix, np.array([calculate_wvfl_in_one_sentence(self.app)]).T])

    # A4
    def ref_in_one_wbnull(self):
        # WBNULL
        print 'WBNULL calculation'
        self.matrix = np.hstack([self.matrix, np.array([calculate_wbnull_in_one_sentence(self.app)]).T])

    # A5
    def ref_in_one_wbfl(self):
        # WBFL
        print 'WBFL calculation'
        self.matrix = np.hstack([self.matrix, np.array([calculate_wbfl_in_one_sentence(self.app)]).T])

    # ------------------------------------------------------------------------
    #                      3task (C part)
    # ------------------------------------------------------------------------
    # C1
    def ref_in_diff_sdist(self):
        # SDIST
        print 'SDIST calculation'
        self.matrixA = np.hstack([self.matrix,np.array([calculate_sdist_in_diff_sentence(self.app)]).T])

    # C2, C3
    def entity_freq_in_doc(self):
        # CRFQ, DRFQ
        print 'CRFQ, DRFQ calculation'
        self.matrixA = np.hstack(
            [self.matrix, np.array([calculate_entity_freq_in_doc_in_diff_sentence(self.app)]).T])

    # C4 , C5
    def whether_type_of_entity_is_unique_in_doc(self):
        # WCO, WDO
        print 'WCO, WDO calculation'
        self.matrixA = np.hstack(
            [self.matrix, np.array([calculate_whether_type_of_entity_is_unique_in_doc(self.app)]).T])

    # D



#=============================================================================
#                       3 task A realisation
#=============================================================================

def verb_in_sentence(tagged):
    if len(tagged) == 0:
        return 0
    else:
        return int(u'VERB' in tagged[:, 1])


def only_one_verb_in_sentence(tagged):
    if len(tagged) == 0:
        return 0
    else:
        return int(sum(map(lambda tag: u'VERB' == tag, tagged[:, 1])) == 1)


# CPOS (pos_tag of first entity)
def calculate_cpos_in_one_sentence(app):
    references = app.all_references
    return map(lambda reference: pos_tag(word_tokenize(reference.refAobj.value),
                         tagset='universal')[0][1] if reference.feature_type == Features.InOneSentence else 'NO',references)


# no verb between
def calculate_wvnull_in_one_sentence(app):
    references = app.all_references
    return map(lambda reference: \
                   verb_in_sentence(np.array(pos_tag(reference.tokenized_text_between, tagset='universal'))) \
                       if reference.feature_type == Features.InOneSentence else -1, references)


# only one verb
def calculate_wvfl_in_one_sentence(app):
    references = app.all_references
    return map(
        lambda reference: verb_in_sentence(np.array(pos_tag(reference.tokenized_text_between, tagset='universal'))) \
            if reference.feature_type == Features.InOneSentence else -1, references)


def calculate_wbnull_in_one_sentence(app):
    references = app.all_references
    # INTBOOLLEN я вызываю тебя...........
    return map(lambda reference: int(
        bool(len(reference.text_between))) if reference.feature_type == Features.InOneSentence else -1, references)


def calculate_wbfl_in_one_sentence(app):
    references = app.all_references
    return map(lambda reference: int(
        len(reference.tokenized_text_between) == 1) if reference.feature_type == Features.InOneSentence else -1,
               references)

#=============================================================================
#                       3 task C realisation
#=============================================================================

def calculate_sdist_in_diff_sentence(app):
    # column of matrix with SDIST feature
    sdist_feature = []
    references = app.all_references
    for rel in references:
        if rel.feature_type == Features.InDifferentSentence:
            count_sentence_dividers = 0
            for divider in SENTENCE_DIVIDERS:
                count_sentence_dividers += rel.text_between.count(divider) - 1
            sdist_feature.append(count_sentence_dividers)
        else:
            sdist_feature.append(-1)
    return sdist_feature

#for C2, C3
def calculate_entity_freq_in_doc_in_diff_sentence(app):
    # column of matrix with CRFQ feature
    cfrq_feature_left = []
    cfrq_feature_right = []
    for doc in app.documents:
        doc_file = codecs.open(doc.text_path, 'r', encoding='utf-8')
        text_of_doc = doc_file.read()
        for rel in doc.references:
            if rel.feature_type == Features.InDifferentSentence:
                cfrq_feature_left.append(text_of_doc.count(rel.refAobj.value))
                cfrq_feature_right.append(text_of_doc.count(rel.refBobj.value))
            else:
                cfrq_feature_left.append(-1)
                cfrq_feature_right.append(-1)
        doc_file.close()
    cfrq_feature_left_right = [cfrq_feature_left,cfrq_feature_right]
    return cfrq_feature_left_right

#for C4, C5
def calculate_whether_type_of_entity_is_unique_in_doc(app):
    # column of matrix with WOC feature
    woc_feature_left = []
    woc_feature_right = []
    for doc in app.documents:
        doc_file = codecs.open(doc.text_path, 'r', encoding='utf-8')
        text_of_doc = doc_file.read()
        for rel in doc.references:
            if rel.feature_type == Features.InDifferentSentence:
                woc_feature_left.append(whether_type_is_unique_in_doc(rel.refAobj.type,doc))
                woc_feature_right.append(whether_type_is_unique_in_doc(rel.refBobj.type, doc))
            else:
                woc_feature_left.append(-1)
                woc_feature_right.append(-1)
    woc_feature_left_right = [woc_feature_left,woc_feature_right]
    return woc_feature_left_right


def whether_type_is_unique_in_doc(type, doc):
    type_counts = 0
    for rel in doc.references:
        types = map(lambda ent: ent.type, (rel.refAobj, rel.refBobj))
        if type in types:
            type_counts += 1
            if type_counts >= 2:
                return 0
    return 1

# ------------------------------------------------------------------------
#                      End
# ------------------------------------------------------------------------
