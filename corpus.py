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

    def print_statistics(self):
        self.relations = [rel for doc in self.docs for rel in doc.relations]

        print 'Count of documents: {}'.format(len(self.docs))

        print 'Count of relations [ALL]: {}'.format(len(self.relations))

        references_sentences = relations_in_sentence(self.docs, self.text_encoding)
        print 'Count of relations [IN ONE SENTENCE]: {}\nCount of relations [IN DIFFERENT SENTENCES]: {}'. \
            format(len(references_sentences[0]), len(references_sentences[1]))
        del references_sentences

        all_relations = [rel.refAobj.value for rel in self.relations] + [rel.refBobj.value for rel in self.relations]

        print 'Count of entities in relations: {}'.format(len(all_relations))
        print 'Count of UNIQUE entities in relations: {}'.format(len(set(all_relations)))

    def second(self):
        train_y = [rel.is_fictive for rel in self.relations]

        pipeline = PipeLine(self, False)
        pipeline.fit(self.relations, train_y)
        return pipeline

    def third(self, ):
        train_y = [rel.is_fictive for rel in self.relations]

        pipeline = PipeLine(self, True)
        pipeline.fit(self.relations, train_y)
        self.pipeline = pipeline
        return pipeline

    def fourth(self):
        train_y = [rel.is_fictive for rel in self.relations]
        scoring = ['precision', 'recall', 'f1']

        kf = KFold(n_splits=5)
        self.logger.info('CV start')
        cv_results = cross_validate(PipeLine(self, True).pipeline, self.relations, train_y, cv=kf,
                                    scoring=scoring, verbose=2, return_train_score=True)
        self.logger.info('CV end')

        print(cv_results)

        print('Train results: ')
        for metric in scoring:
            res = cv_results['train_'+metric]
            print('{}: {}; mean: {}; stdev: {}'.format(metric, res, res.mean(), res.std()))

        print('Test results: ')
        for metric in scoring:
            res = cv_results['test_'+metric]
            print('{}: {}; mean: {}; stdev: {}'.format(metric, res, res.mean(), res.std()))
