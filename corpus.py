import fnmatch
import logging
import os

from library.decorators import validate
from library.lib import count_unique_entites_in_relations, references_in_sentence, DynamicFields, WORD_TYPES, DATA_PATH
from models.pipeline import PipeLine


class Corpus(DynamicFields):
    docs = []
    ref_count = 0

    # Default properies
    text_encoding = "utf-8"
    word_type = WORD_TYPES[0]
    fetures = False
    laplace = False
    #  unknown_word_freq = 0.5

    doc_pattern = None
    ann_pattern = None

    logger = logging.getLogger('Corpus')

    def __init__(self, *args, **kwargs):
        kwargs = args[0].__dict__
        super(Corpus, self).__init__(*args, **kwargs)

    def first(self, ):
        self.count_items()
        self.parse_objects()
        self.print_statistics()

    def count_items(self):
        docs, annotations = [], []
        for root, dirnames, filenames in os.walk(os.path.join(os.path.abspath(DATA_PATH), self.path)):
            for filename in fnmatch.filter(filenames, self.doc_pattern):
                docs.append(os.path.join(root, filename))
            self.logger.info('get_filenames_and_count_of_documents EXECUTED, {} documents'.format(len(docs)))
            for filename in fnmatch.filter(filenames, self.ann_pattern):
                annotations.append(os.path.join(root, filename))
            self.logger.info('get_filenames_and_count_of_documents EXECUTED, {} annotations'.format(len(annotations)))

        self.d_paths = docs
        self.a_paths = annotations

    def parse_objects(self):
        raise NotImplementedError()

    @validate
    def print_statistics(self):
        print 'Count of documents: {}'.format(self.document_count)

        references_count = reduce(lambda initial, y: initial + len(y.references), self.documents, 0)
        self.ref_count = references_count
        print 'Count of references [ALL]: {}'.format(references_count)

        references_sentences = references_in_sentence(self.documents, self.text_encoding)
        self.set_refs_in_out(references_sentences[0], references_sentences[1])
        print 'Count of references [IN ONE SENTENSE]: {}\nCount of references [IN DIFERRENT SENTENCES]: {}'. \
            format(len(references_sentences[0]), len(references_sentences[1]))
        del references_sentences

        print 'Count of entities in relations: {}'.format(
            reduce(lambda c, doc: c + len(doc.references) * 2, self.documents, 0))
        print 'Count of UNIQUE entities in relations: {}'.format(
            count_unique_entites_in_relations(self.documents))

    '''
        The data param should be zipped from 2 lists of entyties
        Example:
            data = zip(('ledocaine', 'word1'),('anesthesia', 'word2'))
            ent0[0] = ledocaine; ent0[1] = anesthesia
            ent1[0] = word1; ent1[1] = word2

        Save current pipeline
    '''

    def second(self, data):
        self.pipeline = self.get_baseline_model(data)

    def get_baseline_model(self, data):
        left_test, right_test = (dict(data).keys(), dict(data).values())
        left_words, right_words, types = ([], [], [])
        for document in self.documents:
            for rel in document.references:
                left_words.append(rel.refAobj.value)
                right_words.append(rel.refBobj.value)
                types.append(rel.type)
        pipeline = PipeLine(self, test_counts=50)
        pipeline.fit(left_words, right_words, types)
        pipeline.transform(left_test, right_test)
        print pipeline.test()

        return pipeline

    def third(self, ):
        """
            Return self -> for choose required type
            TODO
        """
        return self

    def relation_in_one_sentence(self):
        """
        Features:
            Relation in sentence = [
                CPOS(part of speech in relation),
                WVNULL(when no verb in between),
                WVFL(when only verb in between),
                WBNULL(no words in between)],
                WBFL(when only one word in between),
            ]
        """
        self.pipeline.ref_in_one_cpos()
        self.pipeline.ref_in_one_wvnull()
        self.pipeline.ref_in_one_wvfl()
        self.pipeline.ref_in_one_wbnull()
        self.pipeline.ref_in_one_wbfl()

    def __getattr__(self, attr):
        try:
            return super(Corpus, self).__getattr__()
        except AttributeError as er:
            return None

    a_paths = ()

    @property
    def annotation_paths(self):
        return self.a_paths

    d_paths = ()

    @property
    def document_paths(self):
        return self.d_paths

    def get_references_from_documents(self):
        initial = list()
        map(lambda doc: initial.extend(doc.references), self.documents)
        return initial

    '''
        Setter for in || out references
    '''

    def set_refs_in_out(self, ref_in, ref_out):
        self.all_references = self.get_references_from_documents()
