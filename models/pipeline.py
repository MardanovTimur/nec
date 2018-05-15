#coding=utf-8
import logging

from sklearn.metrics import precision_score, recall_score, f1_score

__author__ = 'Timur Mardanov'

import codecs
import os
from gensim.models import KeyedVectors
from pymystem3 import Mystem

from joblib import Memory

from library.decorators import log_time

import pickle
from collections import defaultdict

import numpy as np
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from library.lib import SENTENCE_DIVIDERS, PROJECT_PATH
from models.relation import Features
from library.lib import SENTENCE_DIVIDERS, preprocess
from stanfordcorenlp import StanfordCoreNLP


VECTOR_SIZE = 300

memory = Memory('./cache')


@memory.cache(verbose=0)
def pos_tag_cached(*args, **kwargs):
    return pos_tag(*args, **kwargs)


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
    def __init__(self, mapper, mapper_args=list()):
        self.mapper = mapper
        self.mapper_args = mapper_args

    def fit(self, *args):
        return self

    def transform(self, relations, y=None):
        return np.array(self.mapper(relations, *self.mapper_args), ndmin=2).T


class PipeLine(object):
    classifier_n_jobs = 4
    classifier_solver = 'lbfgs'

    _dependency_core = None

    def __init__(self, app, advanced_features=True):
        '''
            Init pipeline
        '''
        estimators = self.get_baseline_features(app)

        if advanced_features:
            add = self.get_advanced_features(app)
            if hasattr(advanced_features, '__iter__'):
                add = filter(lambda t: t[0] in advanced_features, add)
            estimators += add

        self.pipeline = Pipeline([
            ('all_features', FeatureUnion(estimators)),
            ('clf', LogisticRegression(n_jobs=self.classifier_n_jobs, solver=self.classifier_solver, verbose=1))
        ], memory='./cache')

    def get_baseline_features(self, app):
        return [
            ('left_vec', PipeLine.get_vectorizer(app, True)),
            ('right_vec', PipeLine.get_vectorizer(app, False)),
        ]

    def get_advanced_features(self, app):
        return [
                # one sentence
                ('cpos', ColumnTransformer(calculate_cpos_in_one_sentence, [app.language])),
                ('wvnull', ColumnTransformer(calculate_wvnull_in_one_sentence, [app.language])),
                ('wvfl', ColumnTransformer(calculate_wvfl_in_one_sentence, [app.language])),
                ('wbnull', ColumnTransformer(calculate_wbnull_in_one_sentence)),
                ('wbfl', ColumnTransformer(calculate_wbfl_in_one_sentence)),

                # different sentences
                ('sdist', ColumnTransformer(calculate_sdist_in_diff_sentence)),
                ('crfq', ColumnTransformer(crfq_left)),
                ('drfq', ColumnTransformer(crfq_right)),

                # C4, C5
                ('wco_wdo', ColumnTransformer(calculate_whether_type_of_entity_is_unique_in_doc)),
            ] + ([
                # ('drp2c', ColumnTransformer(calculate_dpr2c_in_one_sentence, [self.dependency_core])),
                # ('drp2d', ColumnTransformer(calculate_dpr2d_in_one_sentence, [self.dependency_core])),
            ] if app.language != 'rus' else [
                ('wordvec', ColumnTransformer(calculate_word_vectors)),
            ])

    def save(self, path):
        pickle.dump(self.pipeline, open(path, 'wb'))

    def load(self, path):
        self.pipeline = pickle.load(open(path, 'rb'))

    @property
    def dependency_core(self):
        if not self._dependency_core:
            self._dependency_core = StanfordCoreNLP(os.path.abspath(os.path.join(PROJECT_PATH, 'stanford_nlp')))
        return self._dependency_core

    @staticmethod
    def get_vectorizer(app, left):
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

    def test(self, test_x, test_y):
        pred_y = self.pipeline.predict(test_x)
        logging.info('precision score: {}'.format(precision_score(test_y, pred_y)))
        logging.info('   recall score: {}'.format(recall_score(test_y, pred_y)))
        logging.info('       f1 score: {}'.format(f1_score(test_y, pred_y)))


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
@log_time
def calculate_cpos_in_one_sentence(rels, lang):
    tags = map(
        lambda rel: pos_tag_cached(word_tokenize(rel.refAobj.value), lang=lang, tagset='universal')[0][1]
        if rel.feature_type == Features.InOneSentence else 'NO',
        rels)
    return map(lambda tag: pos_ids[tag], tags)


# no verb between
@log_time
def calculate_wvnull_in_one_sentence(references, lang):
    return map(lambda reference: \
                   verb_in_sentence(np.array(pos_tag_cached(reference.tokenized_text_between, lang=lang, tagset='universal'))) \
                       if reference.feature_type == Features.InOneSentence else -1, references)


# only one verb
@log_time
def calculate_wvfl_in_one_sentence(references, lang):
    return map(lambda reference: verb_in_sentence(np.array(pos_tag_cached(reference.tokenized_text_between, lang=lang, tagset='universal')))\
               if reference.feature_type == Features.InOneSentence else -1,references)


@log_time
def calculate_wbnull_in_one_sentence(references):
    return map(lambda reference: int(bool(len(reference.text_between))) if reference.feature_type == Features.InOneSentence else -1,references)


@log_time
def calculate_wbfl_in_one_sentence(references):
    return map(lambda reference: int(len(reference.tokenized_text_between) == 1) if reference.feature_type == Features.InOneSentence else -1,references)

#=============================================================================
#                       3 task B realisation
#=============================================================================


def build_path_from_root(elements, destination_index, path=None, current_ind = 0, previos_indexes = None):
    '''
        Get path from root to element
    '''

    # Mutable defaults
    if path is None:
        path = []
    if previos_indexes is None:
        previos_indexes = []

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


@log_time
def calculate_dpr2c_in_one_sentence(references, dependency_core):
    print('DPR2C')
    for reference in references:
        if reference.feature_type == Features.InOneSentence:
            reference.path = 'NO'
            continue

        tokenized_text = dependency_core.word_tokenize(reference.text)
        tree = dependency_core.dependency_parse(reference.text)
        index = get_index_from_list(tokenized_text, reference.refAobj)
        path = build_path_from_root(tree, index)
        if not path:
            reference.path = 'None'
        else:
            reference.path = reduce(lambda x,y: x+'/'+y,path)
    return map(lambda reference: reference.path, references)


@log_time
def calculate_dpr2d_in_one_sentence(references, dependency_core):
    print('DPR2D')
    for reference in references:
        if reference.feature_type == Features.InOneSentence:
            reference.path = 'NO'
            continue

        tokenized_text = dependency_core.word_tokenize(reference.text)
        tree = dependency_core.dependency_parse(reference.text)
        index = get_index_from_list(tokenized_text, reference.refBobj)
        path = build_path_from_root(tree, index-1)
        if not path:
            reference.path = 'None'
        else:
            reference.path = reduce(lambda x,y: x+'/'+y,path)
    return map(lambda reference: reference.path, references)


#=============================================================================
#                       3 task C realisation
#=============================================================================
@log_time
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
@log_time
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

def calculate_word_vectors(references):
    fname = 'vec_models/ruwikiruscorpora-nobigrams_upos_skipgram_300_5_2018.vec'
    model = KeyedVectors.load_word2vec_format(fname,binary=False)
    wc_feature = []
    m = Mystem()
    for rel in references:
        words = rel.refAobj.value + " " + rel.refBobj.value
        words = preprocess(words,m)
        tagged_words = pos_tag(tokens=words,tagset='universal',lang='rus')
        wc_feature.append(makeFeatureVec(tagged_words,model,VECTOR_SIZE))
    return wc_feature


def makeFeatureVec(tagged_words, model, num_features=VECTOR_SIZE):
    featureVec = np.zeros((num_features,), dtype="float32")
    nwords = float(0)
    for word,pos in tagged_words:
        if word+'_'+pos in model.vocab.keys():
            nwords = nwords + 1.
            featureVec = np.add(featureVec, model[word+'_'+pos])
    featureVec = np.divide(featureVec, nwords)
    return featureVec

@log_time
def crfq_left(rels):
    return calculate_entity_freq_in_doc_in_diff_sentence(rels, True)


@log_time
def crfq_right(rels):
    return calculate_entity_freq_in_doc_in_diff_sentence(rels, False)
