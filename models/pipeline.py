#coding=utf-8
import os

__author__ = 'Timur Mardanov'

import pickle
from collections import defaultdict

import numpy as np
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from library.lib import SENTENCE_DIVIDERS
from models.relation import Features
from stanfordcorenlp import StanfordCoreNLP


VECTOR_SIZE = 300


class CustomTfidfVectorizer(TfidfVectorizer):

    def __init__(self, left, encoding, ngram_range, max_df, max_features, smooth_idf):
        super(CustomTfidfVectorizer, self).__init__(encoding=encoding, ngram_range=ngram_range, max_df=max_df,
                                                    max_features=max_features, smooth_idf=smooth_idf)
        self.left = left

    def _get_words(self, raw_documents):
        if self.left:
            return [rel.refAobj.value for rel in raw_documents]
        else:
            return [rel.refBobj.value for rel in raw_documents]

    def fit(self, raw_documents, y=None):
        return super(CustomTfidfVectorizer, self).fit(self._get_words(raw_documents), y)

    def transform(self, raw_documents, copy=True):
        return super(CustomTfidfVectorizer, self).transform(self._get_words(raw_documents), copy)

    def fit_transform(self, raw_documents, y=None):
        return super(CustomTfidfVectorizer, self).fit_transform(self._get_words(raw_documents), y)


class ColumnTransformer(BaseEstimator):
    def __init__(self, mapper, mapper_args=None):
        self.mapper = mapper
        self.mapper_args = mapper_args

    def fit(self, *args):
        return self

    def transform(self, relations, y=None):
        res = self.mapper(relations, *self.mapper_args)
        if type(res[0]) != 'list':
            # If one column, wrap in array for np
            res = [res]
        return np.array(res).T


class PipeLine(object):
    classifier_n_jobs = 4
    classifier_solver = 'lbfgs'

    def __init__(self, app, advanced_features=True):
        '''
            Init pipeline
        '''
        self.pos_ids = defaultdict(lambda: len(self.pos_ids.items()))

        estimators = [
            ('left_vec', self.get_vectorizer(app, True)),
            ('right_vec', self.get_vectorizer(app, False)),
        ]

        if advanced_features:
            estimators += [
                ('cpos', ColumnTransformer(calculate_cpos_in_one_sentence)),
                ('wvnull', ColumnTransformer(calculate_wvnull_in_one_sentence)),
                ('wvfl', ColumnTransformer(calculate_wvfl_in_one_sentence)),
                ('wbnull', ColumnTransformer(calculate_wbnull_in_one_sentence)),
                ('wbfl', ColumnTransformer(calculate_wbfl_in_one_sentence)),
                ('sdist', ColumnTransformer(calculate_sdist_in_diff_sentence)),
                ('crfq', ColumnTransformer(crfq_left)),
                ('drfq', ColumnTransformer(crfq_right)),
                ('drp2c', ColumnTransformer(calculate_dpr2c_in_one_sentence, [self.dependency_core])),
                ('drp2d', ColumnTransformer(calculate_dpr2d_in_one_sentence, [self.dependency_core])),
                ('wco_wdo', ColumnTransformer(calculate_whether_type_of_entity_is_unique_in_doc))
            ]

        self.pipeline = Pipeline([
            ('all_features', FeatureUnion(estimators, n_jobs=self.classifier_n_jobs)),
            ('clf', LogisticRegression(n_jobs=self.classifier_n_jobs, solver=self.classifier_solver, verbose=1))
        ])

    def save(self, path):
        pickle.dump(self.pipeline, open(path, 'wb'))

    def load(self, path):
        self.pipeline = pickle.load(open(path, 'rb'))

    def get_vectorizer(self, app, left):
        '''
            Get TfidfVectorizer instance
        '''
        return CustomTfidfVectorizer(
            left=left,
            encoding=app.text_encoding,
            ngram_range=(app.n,) * 2,
            max_df=app.unknown_word_freq or 1.0,
            max_features=2 * len(app.relations),
            smooth_idf=app.laplace,
        )

    def fit(self, relations, types):
        '''
            Fit model
        '''
        self.pipeline.fit(relations, types)

    def init_stanford_dependency_searching(self, ):
        self.dependency_core = StanfordCoreNLP(os.path.abspath(os.path.join(self.app.BASE_PATH, 'stanford_nlp')))

    def test(self, test_relations):
        return self.pipeline.predict(test_relations)

    #------------------------------------------------------------------------
    #                      3task B
    #------------------------------------------------------------------------

    #B3
    def ref_in_one_wcdd(self):
        #WCDD
        print 'WCDD calculation'
        self.matrix = np.hstack([self.matrix, np.array([calculate_wcdd_in_one_sentence(self.app, self.dependency_core)]).T])

    #B4
    def ref_in_one_wrcd(self):
        #WRCD
        print 'WRCD calculation'
        self.matrix = np.hstack([self.matrix, np.array([calculate_wrcd_in_one_sentence(self.app, self.dependency_core)]).T])

    #B5
    def ref_in_one_wrdd(self):
        #WRDD
        print 'WRDD calculation'
        self.matrix = np.hstack([self.matrix, np.array([calculate_wrdd_in_one_sentence(self.app, self.dependency_core)]).T])

    # C4 , C5
    def whether_type_of_entity_is_unique_in_doc(self):
        # WCO, WDO
        print 'WCO, WDO calculation'
        self.matrixA = np.hstack(
            [self.matrix, np.array(calculate_whether_type_of_entity_is_unique_in_doc(self.app)).T])


    # D, не доделано
    def word_wectors(self):
        # WV
        print 'Word vectors calculation'
        self.matrixA = np.hstack([self.matrix, np.array(calculate_word_vectors(self.app))])

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


pos_ids = defaultdict(lambda: len(pos_ids.items()))


# CPOS (pos_tag of first entity)
def calculate_cpos_in_one_sentence(rels):
    tags = map(
        lambda rel: pos_tag(word_tokenize(rel.refAobj.value), tagset='universal')[0][1]
        if rel.feature_type == Features.InOneSentence else 'NO',
        rels)
    return map(lambda tag: pos_ids[tag], tags)


# no verb between
def calculate_wvnull_in_one_sentence(references):
    return map(lambda reference: \
                   verb_in_sentence(np.array(pos_tag(reference.tokenized_text_between, tagset='universal'))) \
                       if reference.feature_type == Features.InOneSentence else -1, references)


# only one verb
def calculate_wvfl_in_one_sentence(references):
    return map(lambda reference: verb_in_sentence(np.array(pos_tag(reference.tokenized_text_between, tagset='universal')))\
               if reference.feature_type == Features.InOneSentence else -1,references)


def calculate_wbnull_in_one_sentence(references):
    #INTBOOLLEN я вызываю тебя...........
    return map(lambda reference: int(bool(len(reference.text_between))) if reference.feature_type == Features.InOneSentence else -1,references)


def calculate_wbfl_in_one_sentence(references):
    return map(lambda reference: int(len(reference.tokenized_text_between) == 1) if reference.feature_type == Features.InOneSentence else -1,references)

#=============================================================================
#                       3 task B realisation
#=============================================================================


def build_path_from_root(elements, destination_index, path=[], current_ind = 0, previos_indexes = []):
    '''
        Get path from root to element
    '''
    path.append(elements[current_ind][0])
    previos_indexes.append(current_ind)
    if (current_ind == destination_index):
        return path
    if elements[current_ind][1] not in previos_indexes:
        if len(elements) != elements[current_ind][1]:
            return build_path_from_root(elements, destination_index, path, elements[current_ind][1] , previos_indexes)
    if elements[current_ind][2] not in previos_indexes:
        if len(elements) != elements[current_ind][2]:
            return build_path_from_root(elements, destination_index, path, elements[current_ind][2], previos_indexes)


def get_index_from_list(tokenized_text, ent,):
    for item, index in zip(tokenized_text, range(len(tokenized_text))):
        if item == ent.value:
            return index
    return 0


def calculate_dpr2c_in_one_sentence(references, dependency_core):
    for reference in references:
        tokenized_text = dependency_core.word_tokenize(reference.text)
        tree = dependency_core.dependency_parse(reference.text)
        index = get_index_from_list(tokenized_text, reference.refAobj)
        path = build_path_from_root(tree, index)
        if not path:
            reference.path = 'None'
        else:
            reference.path = reduce(lambda x,y: x+'/'+y,path)
    return map(lambda reference: reference.path if reference.feature_type == Features.InOneSentence else 'NO',references)


def calculate_dpr2d_in_one_sentence(references, dependency_core):
    for reference in references:
        tokenized_text = dependency_core.word_tokenize(reference.text)
        tree = dependency_core.dependency_parse(reference.text)
        index = get_index_from_list(tokenized_text, reference.refBobj)
        path = build_path_from_root(tree, index-1)
        if not path:
            reference.path = 'None'
        else:
            reference.path = reduce(lambda x,y: x+'/'+y,path)
    return map(lambda reference: reference.path if reference.feature_type == Features.InOneSentence else 'NO',references)


#=============================================================================
#                       3 task C realisation
#=============================================================================

def calculate_sdist_in_diff_sentence(references):
    #column of matrix with SDIST feature
    sdist_feature = []
    for rel in references:
        if rel.feature_type == Features.InDifferentSentence:
            count_sentence_dividers = 0
            for divider in SENTENCE_DIVIDERS:
                count_sentence_dividers += rel.text_between.count(divider) - 1
            sdist_feature.append(count_sentence_dividers)
        else:
            sdist_feature.append(-1)
    return sdist_feature


def calculate_entity_freq_in_doc_in_diff_sentence(rels, left=True):
    #column of matrix with CRFQ feature
    counts = defaultdict(lambda: defaultdict(lambda: 0))

    def ent(rel):
        return rel.refAobj if left else rel.refBobj

    for rel in rels:
        counts[ent(rel).doc_id][ent(rel).value] += 1
    return map(lambda rel: counts[ent(rel).doc_id][ent(rel).value] if rel.feature_type == Features.InDifferentSentence else -1, rels)

#for C4, C5
def calculate_whether_type_of_entity_is_unique_in_doc(rels):
    # column of matrix with WOC feature
    woc_feature_left = []
    woc_feature_right = []
    counts_left = defaultdict(lambda: 0)
    counts_right = defaultdict(lambda: 0)
    for rel in rels:
        counts_left[rel.refAobj.type] += 1
        counts_right[rel.refBobj.type] += 1

    for rel in rels:
        if rel.feature_type == Features.InDifferentSentence:
            woc_feature_left.append(int(counts_left[rel.refAobj.type] < 2))
            woc_feature_right.append(int(counts_right[rel.refBobj.type] < 2))
        else:
            woc_feature_left.append(-1)
            woc_feature_right.append(-1)

    woc_feature_left_right = [woc_feature_left,woc_feature_right]
    return woc_feature_left_right


#=============================================================================
#                       3 task D realisation
#=============================================================================

#не доделано
def calculate_word_vectors(references):
    model = None
    wc_feature = []
    for rel in references:
        words = rel.refAobj.value + " " + rel.refBobj.value
        wc_feature.append(makeFeatureVec(words,model,VECTOR_SIZE))
    return wc_feature


def makeFeatureVec(words, model, num_features=VECTOR_SIZE):
    featureVec = np.zeros((num_features,), dtype="float32")
    nwords = float(0)
    index2word_set = set(model.wv.index2word)
    for word in word_tokenize(words):
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec, model.wv[word])
    featureVec = np.divide(featureVec, nwords)
    return featureVec

def crfq_left(rels):
    return calculate_entity_freq_in_doc_in_diff_sentence(rels, True)


def crfq_right(rels):
    return calculate_entity_freq_in_doc_in_diff_sentence(rels, False)
