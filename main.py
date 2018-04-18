import os, sys, logging
from library.lib import parse_args, get_filenames_and_count_of_documents, \
     WORD_TYPES
from library.annotations import convert_to_objects

logging.basicConfig(filename="app.log",
                    level=logging.INFO,
                    format ='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger('main.py')

class App(object):

    document_count = 0

    text_encoding = "utf-8"
    word_type = WORD_TYPES[0]
    fetures=False
    laplace=False
    unknown_word_freq = 0.5

    def __init__(self, args):
        map(lambda item: setattr(self, item[0], item[1]),dict(filter(lambda x: x[1] is not None,args.__dict__.items())).items())


    def first(self, ):
        self.document_count, self.d_paths, self.a_paths = get_filenames_and_count_of_documents(self.src_train_texts)
        self.documents = convert_to_objects(self.a_paths)
        print self.documents[0]
        print self.document_count


    a_paths = ()
    @property
    def annotation_paths(self,):
        return self.a_paths


    d_paths = ()
    @property
    def document_paths(self,):
        return self.d_paths

if __name__ == '__main__':
    args = parse_args()
    app = App(args)
    logger.info("First task started : Find relations")
    app.first()
