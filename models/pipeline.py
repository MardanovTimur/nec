#coding=utf-8
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

__author__ = 'Timur Mardanov'

class PipeLine(object):
    leftVec = None
    rightVec = None
    classifyer = None

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
        values = np.append(lData, rData, axis=1)
        if self.test_counts is not None:
            self.classifyer.fit(values[:self.test_counts], types[:self.test_counts])
        else:
            self.classifyer.fit(values, types)

    def transform(self, left_words, right_words):
        lData = self.leftVec.transform(left_words).toarray()
        rData = self.rightVec.transform(right_words).toarray()
        self.test_data = np.append(lData, rData, axis=1)

    def test(self,):
        return self.classifyer.predict(self.test_data)

