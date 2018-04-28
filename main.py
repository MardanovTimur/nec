import os, sys, logging
from library.lib import parse_args, get_filenames_and_count_of_documents, \
     WORD_TYPES, DynamicFields
from library.annotations import convert_to_objects
from library.lib import statistic_of_corpus
from library.lib import base_line_model

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
    #  unknown_word_freq = 0.5

    def __init__(self, *args, **kwargs):
        kwargs = args[0].__dict__
        super(App, self).__init__(*args, **kwargs)

    def first(self, ):
        self.document_count, self.d_paths, self.a_paths = get_filenames_and_count_of_documents(self.src_train_texts)
        self.documents = convert_to_objects(self.a_paths, self.src_train_texts, self.text_encoding)
        statistic_of_corpus(self)

    def second(self, ):
        base_line_model(self)


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

    '''
        Setter for in || out references
    '''
    def set_refs_in_out(self, ref_in, ref_out):
        self.ref_in_one_sentence = ref_in
        self.ref_in_out_sentence = ref_out

if __name__ == '__main__':
    args = parse_args()
    app = App(args)
    logger.info("First task started : Find relations")
    app.first()
    app.second()

