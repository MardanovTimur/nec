#coding=utf-8
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

__author__ = 'Timur Mardanov'


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
    def __init__(self, mapper):
        self.mapper = mapper

    def fit(self, *args):
        return self

    def transform(self, relations, y=None):
        return np.array([self.mapper(relations)]).T


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
                ('drfq', ColumnTransformer(crfq_right))
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
            ngram_range=(app.n, ) * 2,
            max_df=app.unknown_word_freq or 1.0,
            max_features=2 * len(app.relations),
            smooth_idf=app.laplace,
        )

    def fit(self, relations, types):
        '''
            Fit model
        '''
        self.pipeline.fit(relations, types)

    def test(self, test_relations):
        return self.pipeline.predict(test_relations)


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
               verb_in_sentence(np.array(pos_tag(reference.tokenized_text_between, tagset='universal')))\
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
#                       3 task B realisation( осторожно. снизу кривой код)
#=============================================================================

def calculate_sdist_in_diff_sentence(references):
    #column of matrix with SDIST feature
    sdist_feature = []
    for rel in references:
        if rel.feature_type == Features.InDifferentSentence:
            count_sentence_dividers = 0
            for divider in SENTENCE_DIVIDERS:
                count_sentence_dividers += rel.text_between.count(divider)-1
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


def crfq_left(rels):
    return calculate_entity_freq_in_doc_in_diff_sentence(rels, True)


def crfq_right(rels):
    return calculate_entity_freq_in_doc_in_diff_sentence(rels, False)
