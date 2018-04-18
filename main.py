import os, sys, logging
from library.lib import parse_args
from library.lib import get_filenames_and_count_of_documents

logging.basicConfig(filename="app.log",
                    level=logging.INFO,
                    format ='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class App(object):

    def __init__(self, args):
        map(lambda item: setattr(self, item[0], item[1]),dict(filter(lambda x: x[1] is not None,args.__dict__.items())).items())


    def first(self, ):
        self.document_count, self.documents = get_filenames_and_count_of_documents(self.src_train_texts)
        print self.document_count



if __name__ == '__main__':
    args = parse_args()
    logger = logging.getLogger('main.py')
    logger.info("App started")
    app = App(args)
    app.first()
