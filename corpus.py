import fnmatch
import logging
import os

from library.lib import relations_in_sentence, DynamicFields, WORD_TYPES, DATA_PATH
from models.pipeline import PipeLine


class Corpus(DynamicFields):
    docs = []
    relations = []

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
        self.count_items()
        self.parse_objects()
        self.print_statistics()

    def count_items(self):
        docs, annotations = [], []
        for root, dirnames, filenames in os.walk(os.path.join(os.path.abspath(DATA_PATH), self.train_path)):
            for filename in fnmatch.filter(filenames, self.doc_pattern):
                docs.append(os.path.join(root, filename))
            for filename in fnmatch.filter(filenames, self.ann_pattern):
                annotations.append(os.path.join(root, filename))

        self.logger.info('Counting complete, total {} document files and {} annotation files'
                         .format(len(docs), len(annotations)))

        self.d_paths = docs
        self.a_paths = annotations

    def parse_objects(self):
        raise NotImplementedError()

    @validate
    def print_statistics(self):
        self.relations = [rel for doc in self.docs for rel in doc.relations]

        print 'Count of documents: {}'.format(len(self.docs))

        print 'Count of relations [ALL]: {}'.format(len(self.relations))

        references_sentences = relations_in_sentence(self.docs, self.text_encoding)
        print 'Count of relations [IN ONE SENTENCE]: {}\nCount of relations [IN DIFFERENT SENTENCES]: {}'. \
            format(len(references_sentences[0]), len(references_sentences[1]))
        del references_sentences

        all_relations = sum([[rel.refAobj.value, rel.refBobj.value] for rel in self.relations], [])

        print 'Count of entities in relations: {}'.format(len(all_relations))
        print 'Count of UNIQUE entities in relations: {}'.format(len(set(all_relations)))

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
        left_words, right_words, target_statements = ([], [], [])
        for document in self.docs:
            for rel in document.references:
                left_words.append(rel.refAobj.value)
                right_words.append(rel.refBobj.value)
                target_statements.append(rel.is_fictive)
        pipeline = PipeLine(self)
        pipeline.fit(left_words, right_words, target_statements)
        pipeline.transform(left_test, right_test)
        print pipeline.test()

        return pipeline

    def third(self, ):
        self.relation_in_one_sentence()

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

    a_paths = ()

    @property
    def annotation_paths(self):
        return self.a_paths

    d_paths = ()

    @property
    def document_paths(self):
        return self.d_paths

    '''
        Setter for in || out references
    '''
    def set_refs_in_out(self, ref_in, ref_out):
        # TODO
        pass
