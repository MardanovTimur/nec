import logging

from brat import BratCorpus
from chemprot import ChemprotCorpus
from library.lib import parse_args
from made import MadeCorpus

logging.basicConfig(filename="app.log",
                    level=logging.INFO,
                    format ='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger('main.py')

CORPUS_CLS = {
    u'MADE-1.0': MadeCorpus,
    u'corpus_release': BratCorpus,
    u'chemprot': ChemprotCorpus
}

if __name__ == '__main__':
    args = parse_args()
    corpus = CORPUS_CLS[args.SRC_TRAIN_TEXTS](args)

    #----------------------------------------------------------------------------
    logger.info("First task started : Find relations")
    corpus.first()

    #----------------------------------------------------------------------------
    logger.info("Second task started : Baseline model")

    '''
        Baseline test_data
    '''
    ent2 = ('anesthesia', )
    ent1 = ('ledocaine',)
    corpus.second(zip(ent1, ent2))

    #----------------------------------------------------------------------------
    logger.info("Third task started : Add extra features for data")
    #---------------------------A-------------------------------------------
    logger.info("Third task started : Extra features for relation in one sentence")
    corpus.third().relation_in_one_sentence()
    #-----------------------------------------------------------------------
