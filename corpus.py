import fnmatch
import logging
import os

from sklearn.model_selection import KFold, cross_validate

from library.lib import relations_in_sentence, DynamicFields, WORD_TYPES, DATA_PATH
from models.pipeline import PipeLine


class Corpus(DynamicFields):
    docs = []
    relations = []

    a_paths = ()
    d_paths = ()

    # Default properies
    text_encoding = "utf-8"
    word_type = WORD_TYPES[0]
    fetures = False
    laplace = False
    unknown_word_freq = None  # 0.5

    doc_pattern = None
    ann_pattern = None

    test_path = None
    _test_docs = None

    model_path = None

    def __init__(self, *args, **kwargs):
        super(Corpus, self).__init__(*args, **kwargs)
        self.logger = logging.getLogger(self.__class__.__name__)

    def first(self, ):
        self.d_paths, self.a_paths = self.get_paths(self.train_path)
        logging.info('Parsing objects...')
        self.docs = self.parse_objects(self.d_paths, self.a_paths)
        self.print_statistics()

    def get_paths(self, basedir):
        docs, annotations = [], []
        for root, dirnames, filenames in os.walk(os.path.join(os.path.abspath(DATA_PATH), basedir)):
            for filename in fnmatch.filter(filenames, self.doc_pattern):
                docs.append(os.path.join(root, filename))
            for filename in fnmatch.filter(filenames, self.ann_pattern):
                annotations.append(os.path.join(root, filename))

        self.logger.info('Found {} document files and {} annotation files'.format(len(docs), len(annotations)))

        return docs, annotations

    def parse_objects(self, d_paths, a_paths):
        raise NotImplementedError()

    @property
    def test_rels(self):
        if not self.test_path:
            return None
        if self._test_docs:
            return self._test_docs

        d_paths, a_paths = self.get_paths(self.test_path)
        test_docs = self.parse_objects(d_paths, a_paths)
        return [rel for doc in test_docs for rel in doc.relations]

    def print_statistics(self):
        self.relations = [rel for doc in self.docs for rel in doc.relations]

        logging.info('Count of documents: {}'.format(len(self.docs)))

        logging.info('Count of relations [ALL]: {}'.format(len(self.relations)))

        references_sentences = relations_in_sentence(self.docs, self.text_encoding)
        logging.info('Count of relations [IN ONE SENTENCE]: {}\nCount of relations [IN DIFFERENT SENTENCES]: {}'. \
            format(len(references_sentences[0]), len(references_sentences[1])))
        del references_sentences

        all_relations = [rel.refAobj.value for rel in self.relations] + [rel.refBobj.value for rel in self.relations]

        logging.info('Count of entities in relations: {}'.format(len(all_relations)))
        logging.info('Count of UNIQUE entities in relations: {}'.format(len(set(all_relations))))

    def second(self):
        train_y = [rel.is_fictive for rel in self.relations]

        pipeline = PipeLine(self, False)

        logging.info('Evaluating baseline model')

        if self.test_rels:
            self.fit(pipeline, self.relations, train_y)
            pipeline.test(self.test_rels, [rel.is_fictive for rel in self.test_rels])
        else:
            self.cross_validate(pipeline, self.relations, train_y)

        return pipeline

    def third(self, ):
        train_y = [rel.is_fictive for rel in self.relations]

        pipeline = PipeLine(self, True)

        if self.test_rels:
            self.fit(pipeline, self.relations, train_y)
            pipeline.test(self.test_rels, [rel.is_fictive for rel in self.test_rels])
        else:
            self.cross_validate(pipeline, self.relations, train_y)

        self.pipeline = pipeline
        return pipeline

    def fourth(self):
        train_y = [rel.is_fictive for rel in self.relations]

        self.logger.info('CV start')

        feature_packs = [
            ('cpos', 'wvnull', 'wvfl', 'wbnull', 'wbfl'),
            ('drp2c', 'drp2d'),
            ('sdist', 'crfq', 'drfq', 'wco_wdo'),
            ('wordvec', )
        ]

        for i in range(1, len(feature_packs)):
            feature_packs[i] += feature_packs[i-1]

        for pack in feature_packs:
            logging.info('Testing {}'.format(pack))
            pipeline = PipeLine(self, pack)
            self.cross_validate(pipeline, self.relations, train_y)

        self.logger.info('CV end')

    def fit(self, pipeline, X, y):
        if self.model_path:
            pipeline.load(self.model_path)
        else:
            pipeline.fit(X, y)

    def cross_validate(self, pipeline, X, y, n_splits=5):
        scoring = ['precision', 'recall', 'f1']

        kf = KFold(n_splits=n_splits)

        cv_results = cross_validate(pipeline.pipeline, X, y, cv=kf,
                                    scoring=scoring, verbose=1, return_train_score=True)

        logging.info('Train results: ')
        for metric in scoring:
            res = cv_results['train_' + metric]
            logging.info('{:>10}: {}; mean: {}; stdev: {}'.format(metric, res, res.mean(), res.std()))

        logging.info('Test results: ')
        for metric in scoring:
            res = cv_results['test_' + metric]
            logging.info('{:>10}: {}; mean: {}; stdev: {}'.format(metric, res, res.mean(), res.std()))
