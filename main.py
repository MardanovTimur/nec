import os, sys, logging
from library.lib import parse_args, get_filenames_and_count_of_documents, \
     WORD_TYPES, DynamicFields
from library.annotations import convert_to_objects
from library.lib import statistic_of_corpus
from library.lib import base_line_model
import numpy as np
import os

logging.basicConfig(filename="app.log",
                    level=logging.INFO,
                    format ='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger('main.py')

class App(DynamicFields):

    document_count = 0

    #Default properies
    text_encoding = "utf-8"
    word_type = WORD_TYPES[0]
    fetures=False
    laplace=False

    #Path of project
    BASE_PATH = os.path.dirname(__file__)
    #  unknown_word_freq = 0.5

    def __init__(self, *args, **kwargs):
        kwargs = args[0].__dict__
        super(App, self).__init__(*args, **kwargs)

    def first(self, ):
        self.document_count, self.d_paths, self.a_paths = get_filenames_and_count_of_documents(self.src_train_texts)
        self.documents = convert_to_objects(self.a_paths, self.src_train_texts, self.text_encoding, self.train_size)
        statistic_of_corpus(self)


    '''
        The data param should be zipped from 2 lists of entyties
        Example:
            data = zip(('ledocaine', 'word1'),('anesthesia', 'word2'))
            ent0[0] = ledocaine; ent0[1] = anesthesia
            ent1[0] = word1; ent1[1] = word2

        Save current pipeline
    '''
    def second(self, data):
        self.pipeline = base_line_model(self, data)

    def third(self,):
        '''
            Return self -> for choose required type
        '''
        return self

    def relation_in_one_sentence(self, ):
        '''
        Features:
            Relation in sentence = [
                CPOS(part of speech in relation),
                WVNULL(when no verb in between),
                WVFL(when only verb in between),
                WBNULL(no words in between)],
                WBFL(when only one word in between),
            ]
        '''
        #  self.pipeline.ref_in_one_cpos()
        #  self.pipeline.ref_in_one_wvnull()
        #  self.pipeline.ref_in_one_wvfl()
        #  self.pipeline.ref_in_one_wbnull()
        #  self.pipeline.ref_in_one_wbfl()
        self.pipeline.init_stanford_dependency_searching()
        try:
            self.pipeline.ref_in_one_dpr2c()
        except Exception as e:
            print(e.message)
        finally:
            self.pipeline.dependency_core.close()


    def __getattr__(self, attr):
        try:
            return super(App, self).__getattr__()
        except AttributeError as er:
            return None


    a_paths = ()
    @property
    def annotation_paths(self,):
        return self.a_paths


    d_paths = ()
    @property
    def document_paths(self,):
        return self.d_paths


    def get_references_from_documents(self, ):
        initial = list()
        map(lambda doc: initial.extend(doc.references), self.documents)
        return initial

    '''
        Setter for in || out references
    '''
    def set_refs_in_out(self, ref_in, ref_out):
        self.all_references = self.get_references_from_documents()

if __name__ == '__main__':
    args = parse_args()
    app = App(args)

#----------------------------------------------------------------------------
    logger.info("First task started : Find relations")
    app.first()

#----------------------------------------------------------------------------
    logger.info("Second task started : Baseline model")

    '''
        Baseline test_data
    '''
    ent2 = ('anesthesia', )
    ent1 = ('ledocaine',)
    app.second(zip(ent1, ent2))

#----------------------------------------------------------------------------
    logger.info("Third task started : Add extra features for data")
    #---------------------------A-------------------------------------------
    logger.info("Third task started : Extra features for relation in one sentence")
    app.third().relation_in_one_sentence()
    #-----------------------------------------------------------------------




#----------------------------------------------------------------------------

